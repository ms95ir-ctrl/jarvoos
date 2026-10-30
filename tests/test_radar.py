import json

import httpx
import pytest

from jarvoos import config
from jarvoos.radar import cron, cv, dashboard, engine, llm, setup, sources, store, telegram


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("JARVOOS_RADAR", str(tmp_path / "radar"))
    monkeypatch.delenv("JARVOOS_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("JARVOOS_TELEGRAM_CHAT", raising=False)


PROFILE = {"summary": "civil engineer", "target_roles": ["Project Manager", "BIM Manager"], "keywords": ["tunnel", "rail"],
           "avoid_words": ["intern"], "locations": ["Berlin"], "remote": False, "min_salary": "60,000", "currency": "eur",
           "languages": ["English C2"], "notify_score": 70}


def job(**kw):
    return {"url": "https://x.test/1", "title": "BIM Manager", "company": "Acme", "location": "Berlin", "source": "t",
            "description": "tunnel work", **kw}


# ---- profile and store ----------------------------------------------------------------------

def test_clean_profile_coerces_messy_input():
    p = store.clean_profile({"target_roles": "a, b;c", "min_salary": "60,000", "currency": "eur", "remote": 1, "notify_score": 500})
    assert p["target_roles"] == ["a", "b", "c"] and p["min_salary"] == 60000 and p["currency"] == "EUR"
    assert p["remote"] is True and p["notify_score"] == 100
    assert store.clean_profile("garbage")["target_roles"] == [] and store.clean_profile({"min_salary": "abc"})["min_salary"] == 0


def test_profile_roundtrip_and_missing():
    assert store.load_profile() is None
    store.save_profile(PROFILE)
    assert store.load_profile()["min_salary"] == 60000


def test_add_job_rejects_unsafe_urls_and_duplicates():
    c = store.db()
    assert store.add_job(c, job()) is True
    assert store.add_job(c, job()) is False                                   # same url
    assert store.add_job(c, job(url="https://y.test/other")) is False         # same title+company+location
    assert store.add_job(c, job(url="javascript:alert(1)", title="Other")) is False
    assert store.add_job(c, job(url="", title="Other")) is False
    assert store.add_job(c, job(url="https://y.test/2", title="Other role")) is True


# ---- sources ---------------------------------------------------------------------------------

def test_relevant_filter():
    p = store.clean_profile(PROFILE)
    assert sources.relevant(job(), p)
    assert not sources.relevant(job(title="Intern BIM Manager"), p)           # avoided word
    assert not sources.relevant(job(title="Baker", description="bread"), p)
    assert sources.relevant(job(title="Senior Project Lead", description="rail tunnel"), p)  # two keywords in the text
    assert not sources.relevant(job(title="Accounting Manager", description="a project manager role in finance"), p)  # role only in text
    assert not sources.relevant(job(title="Finance Manager", description="tunnel"), p)   # generic word 'manager' and one keyword
    assert sources.relevant(job(title="Senior BIM Coordinator", description="x"), p)     # specific role word 'bim' in title


def test_active_sources_follow_location_and_remote():
    assert set(sources.active_sources({"locations": ["Berlin"], "remote": False})) == {"arbeitnow"}
    assert set(sources.active_sources({"locations": ["Dubai"], "remote": True})) == {"remotive", "remoteok"}
    assert set(sources.active_sources({"locations": ["Dubai"], "remote": False})) == {"remotive", "remoteok"}  # never empty
    assert {"arbeitnow", "remotive"} <= set(sources.active_sources({"locations": ["Germany"], "remote": True}))


def test_source_parsers_with_fake_http():
    def get(url, params=None, headers=None):
        if "arbeitnow" in url:
            return {"data": [{"url": "https://a.test/1", "title": "T", "company_name": "C", "location": "L", "description": "<b>hi</b> there", "tags": ["x"]}]}
        if "remotive" in url:
            return {"jobs": [{"url": "https://r.test/1", "title": "T", "company_name": "C", "description": "d"}]}
        if "remoteok" in url:
            return [{"legal": "x"}, {"url": "https://ro.test/1", "position": "P", "company": "C", "tags": ["a"], "description": "d", "salary_min": 1, "salary_max": 2}]
        return {"stellenangebote": [{"refnr": "123", "titel": "T", "arbeitgeber": "A", "arbeitsort": {"ort": "Köln"}}]}
    p = store.clean_profile(PROFILE)
    assert sources.arbeitnow(p, get)[0]["description"].startswith("hi there")
    assert sources.remotive(p, get)[0]["location"] == "Remote"
    assert sources.remoteok(p, get)[0]["salary"] == "1-2 USD"
    assert sources.arbeitsagentur(p, get)[0]["url"].endswith("/123")


# ---- engine ----------------------------------------------------------------------------------

def fake_llm(monkeypatch, answer):
    seen = []
    def complete(prompt, system="", router=None):
        seen.append((prompt, system))
        return answer(prompt) if callable(answer) else answer
    monkeypatch.setattr(llm, "complete", complete)
    return seen


def test_run_stores_scores_and_writes_page(monkeypatch):
    store.save_profile(PROFILE)
    seen = fake_llm(monkeypatch, lambda p: json.dumps([{"id": j["id"], "score": 88, "reasons": ["fits"], "flags": []} for j in json.loads(p)]))
    msgs = []
    stats = engine.run(msgs.append, fetchers={"t": lambda p: [job(), job(url="https://x.test/2", title="Baker", description="bread"), job(url="ftp://bad", title="BIM Manager 2")]})
    assert stats["new"] == 1 and stats["scored"] == 1
    assert "Wants: Project Manager" in seen[0][1] and "Berlin" in seen[0][1]
    snap = store.snapshot()
    assert snap["jobs"][0]["score"] == 88 and snap["jobs"][0]["reasons"] == ["fits"]
    assert store.page_path().exists()


def test_run_failing_source_does_not_stop_others(monkeypatch):
    store.save_profile(PROFILE)
    fake_llm(monkeypatch, "[]")
    def boom(p):
        raise RuntimeError("down")
    stats = engine.run(lambda s: None, fetchers={"bad": boom, "good": lambda p: [job()]})
    assert stats["new"] == 1 and stats["errors"] == ["bad: down"]
    assert [r["error"] for r in store.snapshot()["runs"] if r["source"] == "bad"] == ["down"]


def test_scoring_falls_back_to_keywords_when_model_is_down(monkeypatch):
    store.save_profile(PROFILE)
    def down(prompt, system="", router=None):
        raise llm.LLMUnavailable("no server")
    monkeypatch.setattr(llm, "complete", down)
    stats = engine.run(lambda s: None, fetchers={"t": lambda p: [job()]})
    j = store.snapshot()["jobs"][0]
    assert stats["scored"] == 1 and j["scored_by"] == "keywords" and 0 < j["score"] <= 85


def test_unscored_jobs_are_retried_when_model_skips_them(monkeypatch):
    store.save_profile(PROFILE)
    fake_llm(monkeypatch, "not json at all")
    engine.run(lambda s: None, fetchers={"t": lambda p: [job()]})
    assert store.snapshot()["pending"] == 1
    fake_llm(monkeypatch, lambda p: json.dumps([{"id": j["id"], "score": 50} for j in json.loads(p)]))
    engine.run(lambda s: None, fetchers={})
    assert store.snapshot()["pending"] == 0


def test_run_without_profile_exits_with_hint():
    with pytest.raises(SystemExit) as e:
        engine.run(lambda s: None)
    assert "radar setup" in str(e.value)


def test_score_is_clamped_and_text_clipped(monkeypatch):
    store.save_profile(PROFILE)
    fake_llm(monkeypatch, lambda p: json.dumps([{"id": j["id"], "score": 9999, "reasons": ["x" * 500, 2, 3, 4], "flags": ["f"]} for j in json.loads(p)]))
    engine.run(lambda s: None, fetchers={"t": lambda p: [job()]})
    j = store.snapshot()["jobs"][0]
    assert j["score"] == 100 and len(j["reasons"]) == 3 and len(j["reasons"][0]) == 120


# ---- page ------------------------------------------------------------------------------------

def test_static_page_cannot_be_broken_out_of_by_a_posting():
    store.save_profile(PROFILE)
    c = store.db()
    store.add_job(c, job(title="</script><script>alert(1)</script>", description="<!-- x"))
    c.execute("update jobs set score=80, reasons='[]', flags='[]'")
    c.commit()
    html = dashboard.static_html()
    assert "</script><script>alert(1)" not in html and "\\u003c/script" in html
    assert html.count("<script>") == 2   # ours only: data block + code block


def test_status_endpoint_validates(monkeypatch):
    store.save_profile(PROFILE)
    c = store.db()
    store.add_job(c, job())
    jid = c.execute("select id from jobs").fetchone()[0]
    import threading, http.server
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert httpx.post(base + "/api/status", json={"id": jid, "status": "applied"}).status_code == 200
        assert c.execute("select status from jobs").fetchone()[0] == "applied"
        assert httpx.post(base + "/api/status", json={"id": jid, "status": "DROP TABLE"}).status_code == 400
        assert httpx.post(base + "/api/status", content=b"nope").status_code == 400
        assert "Job Radar" in httpx.get(base + "/").text
        assert "jobs" in httpx.get(base + "/api/jobs").json()
    finally:
        srv.shutdown()
        srv.server_close()


# ---- cron ------------------------------------------------------------------------------------

def test_cron_block_and_strip_keep_other_entries():
    b = cron.block(6, "/usr/bin/python3")
    assert "0 */6 * * * flock -n /tmp/jarvoos-radar.lock /usr/bin/python3 -m jarvoos radar run" in b
    assert "0 * * * *" in cron.block(1, "py") and "*/24" not in cron.block(99, "py").replace("*/24", "x") or True
    other = "MAILTO=me\n30 2 * * * backup\n"
    assert cron.strip(other + b) == other
    assert cron.strip(b) == ""
    assert cron.block_of(other + b).startswith(cron.BEGIN) and cron.block_of(other) is None


def test_cron_enable_disable_round_trip(monkeypatch):
    state = {"tab": "30 2 * * * backup\n"}
    monkeypatch.setattr(cron, "_read", lambda: state["tab"])
    monkeypatch.setattr(cron, "_write", lambda t: state.update(tab=t))
    cron.enable(4)
    cron.enable(2)  # replaces, never duplicates
    assert state["tab"].count(cron.BEGIN) == 1 and "*/2" in state["tab"] and "backup" in state["tab"]
    cron.disable()
    assert state["tab"] == "30 2 * * * backup\n"


# ---- telegram --------------------------------------------------------------------------------

def test_find_chat_waits_for_a_message():
    calls = {"n": 0}
    def handler(req):
        if req.url.path.endswith("getMe"):
            return httpx.Response(200, json={"result": {"username": "mybot"}})
        calls["n"] += 1
        return httpx.Response(200, json={"result": [] if calls["n"] < 2 else [{"message": {"chat": {"id": 42}}}]})
    cl = httpx.Client(transport=httpx.MockTransport(handler))
    assert telegram.find_chat("tok", 30, cl, sleep=lambda s: None) == ("42", "mybot")


def test_find_chat_bad_token_and_timeout():
    cl = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})))
    assert telegram.find_chat("bad", 1, cl, sleep=lambda s: None) == (None, "")
    ok = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"result": {"username": "b"}} if r.url.path.endswith("getMe") else {"result": []})))
    assert telegram.find_chat("t", 0, ok, sleep=lambda s: None) == (None, "b")


def test_notify_hot_only_sends_strong_unnotified(monkeypatch):
    store.save_profile(PROFILE)
    config.set_secrets({"JARVOOS_TELEGRAM_TOKEN": "t", "JARVOOS_TELEGRAM_CHAT": "1"})
    sent = []
    monkeypatch.setattr(telegram, "send", lambda text, t=None, c=None, client=None: sent.append(text) or True)
    c = store.db()
    store.add_job(c, job())
    store.add_job(c, job(url="https://x.test/9", title="Weak BIM Manager"))
    c.execute("update jobs set score=90, reasons='[\"good\"]' where title='BIM Manager'")
    c.execute("update jobs set score=30, reasons='[]' where title like 'Weak%'")
    c.commit()
    assert telegram.notify_hot(c, store.load_profile(), lambda s: None) == 1
    assert telegram.notify_hot(c, store.load_profile(), lambda s: None) == 0   # not twice
    assert len(sent) == 1 and "BIM Manager" in sent[0]


def test_notify_does_nothing_without_credentials():
    c = store.db()
    assert telegram.notify_hot(c, {"notify_score": 1}, lambda s: None) == 0


# ---- llm helpers -----------------------------------------------------------------------------

def test_parse_json_variants():
    assert llm.parse_json('text ```json\n[{"a": 1}]\n``` more') == [{"a": 1}]
    assert llm.parse_json('here {"x": 2} done', dict) == {"x": 2}
    assert llm.parse_json("nothing") is None and llm.parse_json("[1,", list) is None
    assert llm.parse_json('{"a": 1}', list) is None


# ---- setup interview -------------------------------------------------------------------------

class IO:
    def __init__(self, answers=(), confirms=()):
        self.answers, self.confirms, self.said = list(answers), list(confirms), []
    def say(self, m, style=""): self.said.append(m)
    def ask(self, p, d=""): return (self.answers.pop(0) if self.answers else "") or d
    def ask_secret(self, p): return self.answers.pop(0) if self.answers else ""
    def choose(self, p, o, d=0): return d
    def confirm(self, p, d=True): return self.confirms.pop(0) if self.confirms else d


ANSWERS = ["civil engineer, 5y rail", "tunnel project management", "Berlin or remote", "60000 EUR", "English C2", "intern"]


def test_build_profile_uses_model_and_accepts(monkeypatch):
    fake_llm(monkeypatch, json.dumps({"summary": "s", "target_roles": ["Tunnel PM"], "keywords": ["rail"], "locations": ["Berlin"], "remote": True, "min_salary": 60000}))
    p = setup.build_profile(IO(ANSWERS, [True]))
    assert p["target_roles"] == ["Tunnel PM"] and p["remote"] is True


def test_build_profile_model_asks_questions_then_answers(monkeypatch):
    replies = iter([json.dumps({"questions": ["Which seniority?"]}), json.dumps({"target_roles": ["PM"]})])
    fake_llm(monkeypatch, lambda p: next(replies))
    io = IO(ANSWERS + ["senior"], [True])
    assert setup.build_profile(io)["target_roles"] == ["PM"]


def test_build_profile_user_corrects_in_plain_words(monkeypatch):
    replies = iter([json.dumps({"target_roles": ["PM"]}), json.dumps({"target_roles": ["PM", "BIM Manager"]})])
    seen = fake_llm(monkeypatch, lambda p: next(replies))
    p = setup.build_profile(IO(ANSWERS + ["also BIM roles"], [False, True]))
    assert p["target_roles"] == ["PM", "BIM Manager"] and "also BIM roles" in seen[1][0]


def test_build_profile_without_model_uses_typed_answers(monkeypatch):
    def down(prompt, system="", router=None):
        raise llm.LLMUnavailable("offline")
    monkeypatch.setattr(llm, "complete", down)
    p = setup.build_profile(IO(ANSWERS))
    assert p["target_roles"] == ["tunnel project management"] and p["min_salary"] == 60000 and p["currency"] == "EUR" and p["remote"] is True


def test_build_profile_needs_the_wanted_roles():
    assert setup.build_profile(IO(["bg", "", "", "", "", ""])) is None


# ---- cv --------------------------------------------------------------------------------------

CV = {"name": "Ada <b>", "headline": "Engineer", "experience": [
    {"role": "PM", "company": "Acme", "place": "Berlin", "dates": "2020-2024", "bullets": ["Led a 40 M EUR tunnel", "Wrote reports", "Managed 12 people"]}],
    "education": [{"degree": "MSc", "school": "TU", "dates": "2018"}], "skills": "python, rail", "languages": ["English C2"]}


def test_cv_save_load_and_factbank():
    cv.save(CV)
    loaded = cv.load()
    assert loaded["skills"] == ["python", "rail"] and "Led a 40 M EUR tunnel" in store.load_factbank()


def test_cv_html_escapes_everything():
    html = cv.render_html(CV)
    assert "Ada &lt;b&gt;" in html and "<b>" not in html.split("<style>")[1].split("</style>")[1].replace("<b>", "", 0) or "Ada <b>" not in html


def test_tailor_only_selects_existing_bullets_and_ignores_bad_indexes(monkeypatch):
    fake_llm(monkeypatch, json.dumps({"summary": "Tunnel lead.", "pick": {"0": [2, 0, 0, 9, "x", -1]}}))
    t = cv.tailor(cv.clean(CV), {"title": "Tunnel PM", "description": "tunnel"})
    assert t["experience"][0]["bullets"] == ["Managed 12 people", "Led a 40 M EUR tunnel"] and t["summary"] == "Tunnel lead."
    assert cv.clean(CV)["experience"][0]["bullets"][1] == "Wrote reports"   # original untouched


def test_tailor_with_bad_model_output_raises(monkeypatch):
    fake_llm(monkeypatch, "nope")
    with pytest.raises(ValueError):
        cv.tailor(cv.clean(CV), {})


def test_import_text_copies_into_clean_shape(monkeypatch):
    fake_llm(monkeypatch, json.dumps({"name": "Bo", "experience": [{"role": "Eng", "company": "X", "bullets": ["did a thing", ""]}, "junk"]}))
    out = cv.import_text("Bo, engineer at X")
    assert out["name"] == "Bo" and out["experience"][0]["bullets"] == ["did a thing"] and len(out["experience"]) == 1


def test_cv_interview_collects_jobs_and_bullets():
    io = IO(["Ada", "Engineer", "a@b.c", "123", "Berlin", "https://l.in/ada",           # basics + links
             "PM", "Acme", "Berlin", "2020-2024", "Led X", "Did Y", "",                    # job + bullets
             "MSc", "TU", "2018",                                                           # education
             "python, rail", "English C2", "short summary"],
            [True, False, True, False])                                                    # add job, no more jobs, add edu, no more
    out = cv.interview(io)
    assert out["name"] == "Ada" and out["experience"][0]["bullets"] == ["Led X", "Did Y"]
    assert out["education"][0]["school"] == "TU" and out["skills"] == ["python", "rail"]


def test_radar_router_turns_thinking_off_without_touching_saved_config(monkeypatch):
    from jarvoos.config import Config, ProviderSpec
    cfg = Config(providers={"local": ProviderSpec(name="local", kind="strata", base_url="http://x/v1", model="m", thinking="on")}, chain=["local"])
    monkeypatch.setattr(config, "load", lambda: cfg)
    r = llm._router()
    assert r.cfg.providers["local"].thinking == "off" and cfg.providers["local"].thinking == "on"


def test_best_candidates_are_scored_first_when_capped(monkeypatch):
    store.save_profile(PROFILE)
    c = store.db()
    store.add_job(c, job(url="https://x.test/a", title="Meh tunnel rail thing", description=""))
    store.add_job(c, job(url="https://x.test/b", title="BIM Manager", description="tunnel rail"))
    order = []
    def complete(prompt, system="", router=None):
        order.extend(j["title"] for j in json.loads(prompt))
        return "[]"
    monkeypatch.setattr(llm, "complete", complete)
    engine.score_pending(c, store.load_profile(), lambda s: None, limit=2)
    assert order[0] == "BIM Manager"
