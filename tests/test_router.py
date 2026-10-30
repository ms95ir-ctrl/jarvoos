import json
import threading
import time

import httpx
import pytest

from jarvoos.config import Config, ProviderSpec
from jarvoos.errors import Interrupted, ProviderError, friendly
from jarvoos.providers import Router, build_body, probe, request_headers, retry_delay, thinking_params


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path))
    monkeypatch.setenv("OR_KEY", "sk-or-test-1234567890")


def sse(*deltas, finish="stop", usage=None):
    out = [f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n" for d in deltas]
    out.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}]})}\n\n")
    if usage:
        out.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def ok(text="hello", **kw):
    return httpx.Response(200, content=sse({"content": text}, **kw), headers={"content-type": "text/event-stream"})


def err(status, message="boom", **extra):
    return httpx.Response(status, json={"error": {"message": message, **extra}})


def spec(name="local", kind="strata", url=None, **kw):
    host = "or.test" if kind == "openrouter" else f"{name}.test"
    kw.setdefault("api_key_env", "OR_KEY" if kind == "openrouter" else "")
    return ProviderSpec(name, kind, url or f"http://{host}/v1", kw.pop("model", "m"), **kw)


def cfg(*specs):
    return Config({s.name: s for s in specs}, [s.name for s in specs])


class Net:
    """Fake network: handlers by host, every request recorded."""

    def __init__(self, **handlers):
        self.handlers, self.calls = handlers, []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request):
        host = request.url.host.split(".")[0] if request.url.host.endswith(".test") else request.url.host
        body = json.loads(request.content) if request.content else {}
        self.calls.append((host, request.url.path, body, request.headers))
        h = self.handlers[host]
        r = h(request, body) if callable(h) else h
        if isinstance(r, Exception):
            raise r
        return r

    def hosts(self):
        return [c[0] for c in self.calls if c[1].endswith("/chat/completions")]


def router(net, *specs, **kw):
    return Router(cfg(*specs), transport=net.transport, sleep=kw.pop("sleep", lambda d, c: None), **kw)


def chat(r, **kw):
    got, reasoning, events = [], [], []
    turn = r.chat([{"role": "user", "content": "hi"}], [], got.append, reasoning.append, events.append, **kw)
    return turn, got, reasoning, events


def test_streams_text_tool_calls_and_usage():
    deltas = [{"content": "hi "}, {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "x", "arguments": '{"a"'}}]},
              {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}]
    net = Net(local=httpx.Response(200, content=sse(*deltas, finish="tool_calls", usage={"prompt_tokens": 12, "completion_tokens": 3})))
    turn, got, _, _ = chat(router(net, spec()))
    assert turn.text == "hi " and got == ["hi "]
    assert [(c.id, c.name, c.args) for c in turn.tool_calls] == [("c1", "x", '{"a":1}')]
    assert (turn.usage.prompt_tokens, turn.usage.completion_tokens) == (12, 3)
    assert turn.finish == "tool_calls" and turn.provider == "local"


def test_two_tool_calls_with_same_index_and_distinct_ids():
    d = [{"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}]},
         {"tool_calls": [{"index": 0, "id": "b", "function": {"name": "g", "arguments": "{}"}}]}]
    net = Net(local=httpx.Response(200, content=sse(*d, finish="tool_calls")))
    turn, *_ = chat(router(net, spec()))
    assert [c.name for c in turn.tool_calls] == ["f", "g"]


FAILS = [
    ("network", httpx.ConnectError("refused")),
    ("timeout", httpx.ConnectTimeout("slow")),
    ("server", err(503)),
    ("rate_limit", err(429)),
    ("auth", err(401)),
    ("billing", err(402)),
    ("not_found", err(404)),
]


@pytest.mark.parametrize("kind,failure", FAILS, ids=[k for k, _ in FAILS])
def test_failover_on_each_error_kind(kind, failure):
    net = Net(local=failure, **{"or": ok("from cloud")})
    r = router(net, spec(), spec("cloud", "openrouter"))
    turn, got, _, events = chat(r)
    assert turn.text == "from cloud" and turn.provider == "cloud"
    assert net.hosts() == ["local", "or"]
    ev = [e for e in events if e.kind == "failover"]
    assert len(ev) == 1 and ev[0].from_ == "local" and ev[0].to == "cloud" and ev[0].error.kind == kind
    assert r.active == "cloud" and r.failed_over


def test_not_found_on_local_tries_models_listing_only_once():
    net = Net(local=lambda req, body: err(404) if req.url.path.endswith("chat/completions") else httpx.Response(200, json={"data": []}),
              **{"or": ok("cloud")})
    r = router(net, spec(), spec("cloud", "openrouter"))
    assert chat(r)[0].provider == "cloud"


def test_context_overflow_does_not_fail_over():
    net = Net(local=err(400, "This model's maximum context length is 8192 tokens"), **{"or": ok()})
    with pytest.raises(ProviderError) as ei:
        chat(router(net, spec(), spec("cloud", "openrouter")))
    assert ei.value.kind == "context" and net.hosts() == ["local"]


def test_bad_request_does_not_fail_over():
    net = Net(local=err(400, "unsupported parameter"), **{"or": ok()})
    with pytest.raises(ProviderError) as ei:
        chat(router(net, spec(), spec("cloud", "openrouter")))
    assert ei.value.kind == "bad_request" and net.hosts() == ["local"]


def test_failover_after_partial_output_emits_restart_and_drops_partial_text():
    def cut(req, body):
        raw = b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\ndata: {"choices":[{"delta":{"content":"par"}}]}\n\n'
        return httpx.Response(200, content=raw)  # stream ends without [DONE] or finish_reason

    net = Net(local=cut, **{"or": ok("full answer")})
    turn, got, reasoning, events = chat(router(net, spec(), spec("cloud", "openrouter")))
    assert [e.kind for e in events] == ["failover", "restart"]
    assert got == ["par", "full answer"] and reasoning == ["thinking"]
    assert turn.text == "full answer"


def test_error_chunk_inside_stream_is_classified():
    raw = b'data: {"error": {"code": 429, "message": "slow down", "metadata": {"retry_after_seconds": 500}}}\n\n'
    net = Net(local=httpx.Response(200, content=raw))
    with pytest.raises(ProviderError) as ei:
        chat(router(net, spec()), )
    assert ei.value.kind == "rate_limit" and ei.value.retry_after == 60


def blocking(first=b"", release=None):
    def gen():
        if first:
            yield first
        (release or threading.Event()).wait(5)
    return httpx.Response(200, content=gen())


def test_cancel_mid_stream_raises_interrupted_fast():
    release = threading.Event()
    first = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
    net = Net(local=lambda req, body: blocking(first, release))
    cancel = threading.Event()
    got = []

    def on_text(s):
        got.append(s)
        threading.Timer(0.1, cancel.set).start()

    r = router(net, spec())
    t0 = time.monotonic()
    with pytest.raises(Interrupted):
        r.chat([{"role": "user", "content": "x"}], [], on_text, cancel=cancel)
    release.set()
    assert got == ["a"] and time.monotonic() - t0 < 1.0
    assert isinstance(Interrupted(), KeyboardInterrupt)


def test_cancel_before_start_raises():
    c = threading.Event()
    c.set()
    with pytest.raises(Interrupted):
        chat(router(Net(local=ok()), spec()), cancel=c)


def test_first_token_timeout_and_idle_timeout():
    release = threading.Event()
    net = Net(local=lambda req, body: blocking(b"", release))
    with pytest.raises(ProviderError) as ei:
        chat(router(net, spec(first_token_timeout_s=0.3)))
    assert ei.value.kind == "timeout" and "first token" in ei.value.raw
    first = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
    net = Net(local=lambda req, body: blocking(first, release))
    with pytest.raises(ProviderError) as ei:
        chat(router(net, spec(first_token_timeout_s=5, idle_timeout_s=0.3)))
    assert ei.value.kind == "timeout" and "mid-reply" in ei.value.raw
    release.set()


def test_retry_with_backoff_on_sole_provider_only_before_first_token():
    sleeps, n = [], [0]

    def flaky(req, body):
        n[0] += 1
        return err(503) if n[0] <= 3 else ok("finally")

    net = Net(local=flaky)
    turn, _, _, events = chat(router(net, spec(), sleep=lambda d, c: sleeps.append(d)))
    assert turn.text == "finally" and n[0] == 4
    assert [e.kind for e in events] == ["retry"] * 3
    for got, base in zip(sleeps, (1, 2, 4)):
        assert base * 0.79 <= got <= base * 1.21


def test_retries_give_up_after_three():
    sleeps = []
    net = Net(local=err(500))
    with pytest.raises(ProviderError) as ei:
        chat(router(net, spec(), sleep=lambda d, c: sleeps.append(d)))
    assert ei.value.kind == "server" and len(net.calls) == 4 and len(sleeps) == 3


def test_no_retry_after_text_was_streamed():
    sleeps = []
    cut = httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
    with pytest.raises(ProviderError) as ei:
        chat(router(Net(local=cut), spec(), sleep=lambda d, c: sleeps.append(d)))
    assert ei.value.kind == "network" and sleeps == []


def test_retry_after_is_honoured_and_clamped():
    assert retry_delay(0, 7) == 7 and retry_delay(0, 500) == 60
    assert retry_delay(10) <= 30 * 1.2 + 1e-9


def test_no_retry_when_a_fallback_is_available():
    sleeps = []
    net = Net(local=err(500), **{"or": ok()})
    chat(router(net, spec(), spec("cloud", "openrouter"), sleep=lambda d, c: sleeps.append(d)))
    assert sleeps == [] and net.hosts() == ["local", "or"]


def test_cooldown_skips_failed_primary_then_recovers():
    now = [1000.0]
    mode = {"down": True}
    net = Net(local=lambda req, body: httpx.ConnectError("x") if mode["down"] else ok("local again"), **{"or": ok("cloud")})
    r = router(net, spec(), spec("cloud", "openrouter"), clock=lambda: now[0])
    chat(r)
    net.calls.clear()
    assert chat(r)[0].provider == "cloud" and net.hosts() == ["or"]  # primary not even tried
    mode["down"], now[0] = False, now[0] + 301
    turn, _, _, events = chat(r)
    assert turn.provider == "local" and [e.kind for e in events] == ["recovered"] and not r.failed_over


def test_reset_cooldowns_and_pin():
    net = Net(local=lambda req, body: httpx.ConnectError("x"), **{"or": ok("cloud")})
    r = router(net, spec(), spec("cloud", "openrouter"), clock=lambda: 5.0)
    chat(r)
    net.calls.clear()
    chat(r)
    assert net.hosts() == ["or"]
    r.reset_cooldowns()
    net.calls.clear()
    chat(r)
    assert net.hosts() == ["local", "or"]
    r.pin("cloud")
    net.calls.clear()
    chat(r)
    assert net.hosts() == ["or"] and not r.failed_over
    r.pin(None)
    with pytest.raises(KeyError):
        r.pin("nope")


def test_thinking_param_per_kind():
    off = lambda kind, model="m": thinking_params(spec(kind=kind, thinking="off", model=model))
    assert off("strata") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert off("llamacpp") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert off("openrouter", "deepseek/deepseek-v4-flash") == {"reasoning": {"enabled": False}}
    assert off("openrouter", "deepseek/deepseek-r1") == {}  # reasoning is mandatory there
    assert off("ollama") == {} and off("lmstudio") == {} and off("openai") == {}
    assert thinking_params(spec(thinking="default")) == {}
    assert thinking_params(spec(thinking="on")) == {"chat_template_kwargs": {"enable_thinking": True}}
    assert thinking_params(spec(kind="openrouter", thinking="on", model="deepseek/deepseek-r1")) == {"reasoning": {"enabled": True}}


def test_request_body_and_headers_reach_the_server():
    net = Net(**{"or": ok()})
    chat(router(net, spec("cloud", "openrouter", thinking="off", model="deepseek/deepseek-v4-flash")))
    _, _, body, headers = net.calls[0]
    assert body["reasoning"] == {"enabled": False} and body["stream"] is True and "tools" not in body
    assert headers["authorization"] == "Bearer sk-or-test-1234567890"
    assert headers["http-referer"] and headers["x-title"] == "Jarvoos"
    assert "http-referer" not in request_headers(spec(), None)


def test_tools_false_sends_no_tools():
    tools = [{"type": "function", "function": {"name": "t"}}]
    net = Net(local=ok())
    r = router(net, spec(tools=False))
    r.chat([{"role": "user", "content": "x"}], tools, lambda s: None)
    assert "tools" not in net.calls[0][2]
    assert build_body(spec(), "m", [], tools)["tools"] == tools


def test_empty_reply_after_only_thinking_is_an_empty_error_with_hint():
    raw = sse({"reasoning_content": "hmm " * 20}, finish="length")
    with pytest.raises(ProviderError) as ei:
        chat(router(Net(local=httpx.Response(200, content=raw)), spec(thinking="default")))
    e = ei.value
    assert e.kind == "empty"
    msg, _ = friendly(e, spec())
    assert "only thought" in msg and "thinking off" in msg


def test_missing_openrouter_key_is_an_auth_error(monkeypatch):
    monkeypatch.delenv("OR_KEY")
    with pytest.raises(ProviderError) as ei:
        chat(router(Net(**{"or": ok()}), spec("cloud", "openrouter")))
    assert ei.value.kind == "auth" and "OR_KEY" in ei.value.raw


def test_sole_model_fallback_when_configured_id_is_unknown():
    def local(req, body):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "the-only-one"}]})
        return ok("served") if body["model"] == "the-only-one" else err(404, "model not found")

    net = Net(local=local)
    r = router(net, spec(kind="ollama", model="wrong"))
    assert chat(r)[0].text == "served"
    assert chat(r)[0].text == "served"
    assert [c[2]["model"] for c in net.calls if c[1].endswith("completions")] == ["wrong", "the-only-one", "the-only-one"]


def test_context_tokens_uses_active_provider_or_default():
    r = Router(cfg(spec(context_tokens=131000), spec("cloud", "openrouter", context_tokens=64000)))
    assert r.context_tokens() == 131000
    r.active = "cloud"
    assert r.context_tokens() == 64000
    assert Router(cfg(spec())).context_tokens() == 32000


# ---- probe -----------------------------------------------------------------------------------------

WEATHER = [{"tool_calls": [{"index": 0, "id": "w", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}]


def probe_handler(tool_response):
    def h(req, body):
        assert body["max_tokens"] == 80
        if body.get("tools"):
            return tool_response(body) if callable(tool_response) else tool_response
        return ok("OK")
    return h


def test_probe_passes_for_strata():
    net = Net(local=probe_handler(httpx.Response(200, content=sse(*WEATHER, finish="tool_calls"))))
    res = probe(spec(thinking="on"), None, transport=net.transport)
    assert res.ok and res.chat_ok and res.tools_ok and res.error is None and res.ttft_ms >= 0
    assert net.calls[0][2]["chat_template_kwargs"] == {"enable_thinking": False}  # thinking forced off


def test_probe_llamacpp_without_jinja_hints_at_the_flag():
    net = Net(local=probe_handler(err(500, "tools param requires --jinja flag")))
    res = probe(spec(kind="llamacpp"), None, transport=net.transport)
    assert res.chat_ok and not res.tools_ok and not res.ok and "--jinja" in res.hint


def test_probe_model_that_ignores_tools():
    net = Net(local=probe_handler(ok("The weather is nice")))
    res = probe(spec(kind="ollama"), None, transport=net.transport)
    assert res.chat_ok and not res.tools_ok and "tool support" in res.hint


def test_probe_openrouter_ok_and_sends_key():
    net = Net(**{"or": probe_handler(httpx.Response(200, content=sse(*WEATHER, finish="tool_calls")))})
    res = probe(spec("cloud", "openrouter"), "sk-or-xyz", transport=net.transport)
    assert res.ok and net.calls[0][3]["authorization"] == "Bearer sk-or-xyz"


def test_probe_chat_failure_reports_the_error_and_url():
    net = Net(local=httpx.ConnectError("refused"))
    res = probe(spec(), None, transport=net.transport)
    assert not res.ok and not res.chat_ok and res.error.kind == "network"
    assert "isn't answering" in res.hint and "local.test" in res.hint


def test_probe_bad_key():
    net = Net(**{"or": err(401, "No auth credentials found")})
    res = probe(spec("cloud", "openrouter"), "bad", transport=net.transport)
    assert not res.chat_ok and res.error.kind == "auth" and "openrouter.ai/keys" in res.hint


def test_probe_chat_only_skips_tool_check():
    net = Net(local=ok("OK"))
    res = probe(spec(tools=False), None, with_tools=False, transport=net.transport)
    assert res.ok and res.chat_ok and len(net.calls) == 1
