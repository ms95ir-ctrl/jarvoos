import sys
import types

import pytest

from jarvoos import config, detect, doctor
from test_detect import STRATA, fake_servers
from test_setup import GOOD, Err, Probe, or_routes

DDG = {"duckduckgo.com:443": {"/html/": (200, {})}}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    results = {}
    monkeypatch.setitem(sys.modules, "jarvoos.providers", types.SimpleNamespace(probe=lambda spec, key, *, with_tools=True: results.get(spec.name, Probe())))
    monkeypatch.setitem(sys.modules, "jarvoos.errors", types.SimpleNamespace(friendly=lambda e, spec: (f"{spec.name} is down", e.raw)))
    return types.SimpleNamespace(home=tmp_path, results=results)


def write(*specs, **kw):
    config.save(config.Config({s.name: s for s in specs}, [s.name for s in specs], **kw))


LOCAL = config.ProviderSpec("local", "strata", "http://127.0.0.1:8090/v1", "qwen3-30b", context_tokens=131072)
OR = config.ProviderSpec("openrouter", "openrouter", detect.OPENROUTER_URL, "deepseek/deepseek-v4-flash", "OPENROUTER_API_KEY")


def by_name(checks):
    return {c.name: c for c in checks}


def test_unconfigured_exits_nonzero(env, monkeypatch, capsys):
    fake_servers(monkeypatch, {})
    assert doctor.main() == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "jarvoos setup" in out and "To fix:" in out


def test_healthy_setup_exits_zero(env, monkeypatch, capsys):
    write(LOCAL, OR)
    (env.home / "secrets.env").write_text(f"OPENROUTER_API_KEY={GOOD}\n")
    (env.home / "secrets.env").chmod(0o600)
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA, **DDG}, other=or_routes())
    checks = doctor.run_checks(config.load())
    c = by_name(checks)
    assert c["local chat"].level == "ok" and "310ms" in c["local chat"].detail
    assert c["local tools"].level == "ok" and c["openrouter key"].level == "ok" and c["search"].level == "ok"
    assert c["secrets file"].level == "ok" and c["config"].level == "ok"
    assert all(x.level != "fail" for x in checks)
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert "0 problem(s)" in out or "All good" in out


def test_refused_provider_fails_with_fix(env, monkeypatch):
    write(LOCAL)
    fake_servers(monkeypatch, {})
    c = by_name(doctor.run_checks(config.load()))
    assert c["local reachable"].level == "fail" and "start the server" in c["local reachable"].fix
    assert "local chat" not in c
    assert doctor.render(doctor.run_checks(config.load())) == 1


def test_chat_failure_uses_friendly_error(env, monkeypatch):
    write(LOCAL)
    env.results["local"] = Probe(chat_ok=False, tools_ok=False, error=Err(raw="boom"))
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA, **DDG})
    c = by_name(doctor.run_checks(config.load()))
    assert c["local chat"].level == "fail" and c["local chat"].detail == "local is down" and c["local chat"].fix


def test_tools_failing_is_a_warning_with_jinja_hint(env, monkeypatch):
    write(config.ProviderSpec("local", "llamacpp", "http://127.0.0.1:8080/v1", "m"))
    env.results["local"] = Probe(tools_ok=False)
    fake_servers(monkeypatch, {"127.0.0.1:8080": {"/v1/models": (200, {"data": [{"id": "m"}]})}, **DDG})
    c = by_name(doctor.run_checks(config.load()))
    assert c["local tools"].level == "warn" and "--jinja" in c["local tools"].fix
    assert doctor.render([x for x in doctor.run_checks(config.load())]) == 0


def test_missing_key_and_rejected_key(env, monkeypatch):
    write(OR)
    fake_servers(monkeypatch, {}, other=or_routes())
    c = by_name(doctor.run_checks(config.load()))
    assert c["openrouter key"].level == "fail" and "OPENROUTER_API_KEY" in c["openrouter key"].detail
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong")
    c = by_name(doctor.run_checks(config.load()))
    assert c["openrouter key"].level == "fail" and "rejected" in c["openrouter key"].detail


def test_model_missing_from_models_list_warns(env, monkeypatch):
    write(config.ProviderSpec("local", "llamacpp", "http://127.0.0.1:8080/v1", "ghost"))
    fake_servers(monkeypatch, {"127.0.0.1:8080": {"/v1/models": (200, {"data": [{"id": "real"}]})}, **DDG})
    assert by_name(doctor.run_checks(config.load()))["local model"].level == "warn"


def test_ollama_context_warning(env, monkeypatch):
    write(config.ProviderSpec("ollama", "ollama", "http://127.0.0.1:11434/v1", "llama3", context_tokens=32000))
    fake_servers(monkeypatch, {"127.0.0.1:11434": {"/v1/models": (200, {"data": [{"id": "llama3"}]})}, **DDG})
    c = by_name(doctor.run_checks(config.load()))
    assert c["ollama context"].level == "warn" and "OLLAMA_CONTEXT_LENGTH" in c["ollama context"].fix


def test_secrets_permissions_and_fix(env, monkeypatch, capsys):
    write(LOCAL)
    sp = env.home / "secrets.env"
    sp.write_text("X=1\n")
    sp.chmod(0o644)
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA, **DDG})
    c = by_name(doctor.run_checks(config.load()))
    assert c["secrets file"].level == "fail" and c["secrets file"].fix == f"chmod 600 {sp}"
    assert doctor.main() == 1
    assert doctor.main(fix=True) == 0
    assert sp.stat().st_mode & 0o777 == 0o600
    assert "chmod 600" in capsys.readouterr().out


def test_fix_creates_dirs(env):
    sub = env.home / "new" / "home"
    import os
    os.environ["JARVOOS_HOME"] = str(sub)
    try:
        doctor.main(fix=True)
        assert sub.is_dir() and (sub / "state").is_dir()
    finally:
        os.environ["JARVOOS_HOME"] = str(env.home)


def test_lite_skips_probe_and_extras(env, monkeypatch):
    write(LOCAL)
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    names = set(by_name(doctor.run_checks(config.load(), lite=True)))
    assert "local reachable" in names and "local chat" not in names and "search" not in names and "git" not in names


def test_not_live_does_not_touch_network(env, monkeypatch):
    write(LOCAL)
    fake_servers(monkeypatch, {})
    c = by_name(doctor.run_checks(config.load(), live=False))
    assert c["local"].level == "ok" and "not tested" in c["local"].detail
