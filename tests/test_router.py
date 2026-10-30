import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvoos import config
from jarvoos.agent import Agent
from jarvoos.providers import AllProvidersFailed, Provider, Router
from jarvoos.skills import SkillStore
from jarvoos.tools import Registry, build_registry

FAILOVER = {"connect_timeout": 1.0, "first_token_base": 3.0, "first_token_per_1k": 0.0,
            "fallback_timeout": 3.0, "cooldown": 60.0}


def sse(*deltas, finish="stop"):
    out = [f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n" for d in deltas]
    out.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}]})}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def serve(handler_fn):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): ...
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, payload = handler_fn(body)
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload)
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVENTS_LOG", tmp_path / "events.log")
    monkeypatch.setattr("jarvoos.providers.EVENTS_LOG", tmp_path / "events.log")
    monkeypatch.setattr("jarvoos.providers.STATE_DIR", tmp_path)


def mk(url, name="p", thinking="default"):
    return Provider(name, url, "m", thinking=thinking, connect_timeout=1.0)


def test_streams_text_and_tool_calls():
    def h(body):
        return 200, sse({"content": "hi "}, {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "x", "arguments": '{"a"'}}]},
                        {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}, finish="tool_calls")
    srv, url = serve(h)
    got = []
    turn = mk(url).stream([{"role": "user", "content": "q"}], [], first_token_timeout=3, on_text=got.append)
    assert turn.text == "hi " and got == ["hi "]
    assert turn.tool_calls == [{"id": "c1", "name": "x", "args": '{"a":1}'}]


def test_thinking_off_sends_template_kwarg():
    seen = {}
    def h(body):
        seen.update(body)
        return 200, sse({"content": "ok"})
    srv, url = serve(h)
    mk(url, thinking="off").stream([{"role": "user", "content": "q"}], [], first_token_timeout=3)
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_failover_on_connection_refused_and_recovery():
    srv, good = serve(lambda b: (200, sse({"content": "from fallback"})))
    events = []
    r = Router(mk("http://127.0.0.1:1/v1", "primary"), mk(good, "fallback"), FAILOVER, [events.append])
    resets = []
    turn = r.complete([{"role": "user", "content": "q"}], [], on_reset=lambda *a: resets.append(a))
    assert turn.text == "from fallback" and turn.provider == "fallback"
    assert [e["kind"] for e in events] == ["failover"] and resets
    # second call: primary is in cooldown, goes straight to the fallback with no new event
    r.complete([{"role": "user", "content": "q"}], [])
    assert len(events) == 1


def test_failover_on_http_error_and_empty_reply():
    srv_bad, bad = serve(lambda b: (500, b"boom"))
    srv_empty, empty = serve(lambda b: (200, sse({"reasoning_content": "thinking…"}, finish="length")))
    srv_good, good = serve(lambda b: (200, sse({"content": "ok"})))
    for primary in (bad, empty):
        r = Router(mk(primary, "primary"), mk(good, "fallback"), FAILOVER)
        assert r.complete([{"role": "user", "content": "q"}], []).text == "ok"


def test_primary_recovers_after_cooldown():
    srv, good = serve(lambda b: (200, sse({"content": "fb"})))
    srv2, primary_ok = serve(lambda b: (200, sse({"content": "primary"})))
    events = []
    r = Router(mk("http://127.0.0.1:1/v1", "primary"), mk(good, "fallback"), FAILOVER, [events.append])
    r.complete([{"role": "user", "content": "q"}], [])
    r.primary = mk(primary_ok, "primary")
    r.down_until = 0  # cooldown over
    assert r.complete([{"role": "user", "content": "q"}], []).provider == "primary"
    assert [e["kind"] for e in events] == ["failover", "recovered"]


def test_all_failed_raises():
    r = Router(mk("http://127.0.0.1:1/v1", "a"), mk("http://127.0.0.1:2/v1", "b"), FAILOVER)
    with pytest.raises(AllProvidersFailed):
        r.complete([{"role": "user", "content": "q"}], [])


class QuietUI:
    def __init__(self, answer="yes"):
        self.answer, self.asked = answer, []
    def text(self, c): ...
    def reasoning(self, c): ...
    def notice(self, m): ...
    def tool(self, n, a, r): ...
    def confirm(self, n, a, r):
        self.asked.append(n)
        return self.answer


def test_agent_runs_tool_loop_and_asks_for_exec(tmp_path):
    calls = []
    def h(body):
        calls.append(body)
        if not any(m["role"] == "tool" for m in body["messages"]):
            return 200, sse({"tool_calls": [{"index": 0, "id": "c1", "function": {
                "name": "run_shell", "arguments": json.dumps({"command": "echo hello-from-tool"})}}]}, finish="tool_calls")
        return 200, sse({"content": "done"})
    srv, url = serve(h)
    cfg = config.load_config()
    ui = QuietUI("yes")
    agent = Agent(Router(mk(url), None, FAILOVER), build_registry(cfg, SkillStore([])), "", cfg, ui)
    assert agent.run("run it") == "done"
    assert ui.asked == ["run_shell"]
    tool_msg = [m for m in agent.messages if m["role"] == "tool"][0]
    assert "hello-from-tool" in tool_msg["content"]


def test_declined_tool_is_not_run(tmp_path):
    target = tmp_path / "nope.txt"
    def h(body):
        if not any(m["role"] == "tool" for m in body["messages"]):
            return 200, sse({"tool_calls": [{"index": 0, "id": "c1", "function": {
                "name": "write_file", "arguments": json.dumps({"path": str(target), "content": "x"})}}]}, finish="tool_calls")
        return 200, sse({"content": "ok"})
    srv, url = serve(h)
    cfg = config.load_config()
    agent = Agent(Router(mk(url), None, FAILOVER), build_registry(cfg, SkillStore([])), "", cfg, QuietUI("no"))
    agent.run("write")
    assert not target.exists()


def test_fetch_refuses_private_addresses():
    cfg = config.load_config()
    reg = build_registry(cfg, SkillStore([]))
    assert "private" in reg.tools["fetch_url"].fn("http://127.0.0.1:8080/")
