import http.server
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

from jarvoos import tools
from jarvoos.errors import Interrupted
from jarvoos.tools import ToolContext, run_shell, summarize_call, tool_schemas


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(tmp_path)


def test_run_shell_basic_and_exit_code(tmp_path):
    assert run_shell("echo hi", 5, cwd=tmp_path) == "exit 0\nhi"
    out = run_shell("echo oops >&2; exit 3", 5, cwd=tmp_path)
    assert out.startswith("exit 3") and "oops" in out


def test_run_shell_timeout_kills_the_whole_process_group(tmp_path):
    marker = tmp_path / "alive"
    t0 = time.monotonic()
    out = run_shell(f"(sleep 1.5; touch {marker}) & sleep 30", 1, cwd=tmp_path)
    assert time.monotonic() - t0 < 5 and "timed out" in out and "killed" in out
    time.sleep(2)
    assert not marker.exists()  # the background child died with the group


def test_run_shell_cancel_kills_and_raises(tmp_path):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Interrupted) as ei:
        run_shell("echo started; sleep 30", 60, cancel, cwd=tmp_path)
    assert time.monotonic() - t0 < 3 and "started" in ei.value.partial


def test_run_shell_streams_output_live(tmp_path):
    seen = []
    run_shell("echo one; sleep 0.3; echo two", 5, cwd=tmp_path, on_output=seen.append)
    assert "".join(seen) == "one\ntwo\n" and len(seen) >= 2


def test_shell_cwd_persists_between_calls(ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    tools.TOOLS["run_shell"].fn(ctx, command="cd sub")
    assert ctx.cwd == (tmp_path / "sub").resolve()
    assert tools.TOOLS["run_shell"].fn(ctx, command="pwd").endswith(str((tmp_path / "sub").resolve()))


def test_shell_hides_secret_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "supersecret")
    monkeypatch.setenv("PLAIN", "visible")
    out = run_shell('echo "[$MY_API_KEY][$PLAIN]"', 5, cwd=tmp_path)
    assert "[][visible]" in out


def test_shell_long_output_is_clipped(tmp_path):
    out = run_shell("seq 1 20000", 10, cwd=tmp_path)
    assert len(out) < 32_000 and "chars cut" in out and out.rstrip().endswith("20000")


def test_read_file_offset_limit_and_more_hint(ctx, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("\n".join(f"l{i}" for i in range(1, 11)))
    assert tools.read_file(ctx, "a.txt", offset=3, limit=2) == "l3\nl4\n[more: call read_file again with offset=5]"
    assert tools.read_file(ctx, "a.txt", offset=9, limit=10) == "l9\nl10"
    assert "no lines at offset" in tools.read_file(ctx, "a.txt", offset=99)
    (tmp_path / "bin").write_bytes(b"\0\1\2")
    assert "binary" in tools.read_file(ctx, "bin")
    assert tools.read_file(ctx, "missing").startswith("error:")


def test_read_file_caps_huge_files(ctx, tmp_path):
    (tmp_path / "big").write_text("x" * 100 + "\n" and ("y" * 99 + "\n") * 5000)
    out = tools.read_file(ctx, "big")
    assert len(out) < 70_000 and "more: call read_file" in out


def test_grep_and_glob(ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "src" / "b.txt").write_text("alpha beta\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("def alpha(): pass\n")
    (tmp_path / ".env").write_text("alpha=secret\n")
    out = tools.grep(ctx, "alpha")
    assert "a.py" in out and "b.txt" in out and "node_modules" not in out and ".env" not in out
    out = tools.grep(ctx, "ALPHA", glob="*.py", ignore_case=True)
    assert "a.py" in out and "b.txt" not in out
    assert tools.grep(ctx, "zzz") == "no matches"
    assert tools.grep(ctx, "(").startswith("error: invalid regex")
    assert tools.glob_files(ctx, "*.py") == "src/a.py"
    assert tools.glob_files(ctx, "**/*.txt") == "src/b.txt"
    assert tools.glob_files(ctx, "*.rs") == "no files matched"


def test_grep_python_fallback(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda n: None)
    (tmp_path / "a.py").write_text("one\nneedle here\n")
    (tmp_path / "b.bin").write_bytes(b"needle\0")
    out = tools.grep(ctx, "needle")
    assert "a.py:2:needle here" in out and "b.bin" not in out


def test_edit_and_write_previews_and_errors(ctx, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("one\ntwo\nthree\n")
    t = tools.TOOLS["edit_file"]
    prev = t.preview(ctx, {"path": "f.txt", "old": "two", "new": "2"})
    assert "-two" in prev and "+2" in prev
    assert t.fn(ctx, path="f.txt", old="two", new="2") == f"edited {f}"
    assert f.read_text() == "one\n2\nthree\n"
    assert "not found" in t.fn(ctx, path="f.txt", old="zzz", new="y")
    f.write_text("a\na\n")
    assert "2 times" in t.fn(ctx, path="f.txt", old="a", new="b")
    assert t.preview(ctx, {"path": "f.txt", "old": "a", "new": "b"}).startswith("error:")
    w = tools.TOOLS["write_file"]
    assert w.preview(ctx, {"path": "new.txt", "content": "x\ny"}).startswith("new file, 2 lines")
    assert "-a" in w.preview(ctx, {"path": "f.txt", "content": "z\n"})
    w.fn(ctx, path="deep/er/new.txt", content="x")
    assert (tmp_path / "deep/er/new.txt").read_text() == "x"


def test_tool_schemas_by_mode():
    names = {t["function"]["name"] for t in tool_schemas("approve", True)}
    assert {"run_shell", "read_file", "grep", "glob", "edit_file", "write_file", "fetch_url", "web_search"} <= names
    assert not ({"mcp", "todo", "subagent"} & names)
    assert tool_schemas("chat", True) == [] and tool_schemas("auto", False) == []
    for t in tool_schemas("auto", True):
        assert t["function"]["parameters"]["type"] == "object"


def test_summarize_call_redacts_keys():
    assert summarize_call("run_shell", {"command": "curl -H 'Authorization: Bearer abcdef123456789' x"}) == "$ curl -H 'Authorization: Bearer ***' x"
    assert "sk-or-" not in summarize_call("run_shell", {"command": "echo sk-or-v1-abcdef123456"})
    assert summarize_call("run_shell", {"command": "OPENROUTER_API_KEY=abc123xyz run"}) == "$ OPENROUTER_API_KEY=*** run"
    assert summarize_call("read_file", {"path": "a/b.py"}) == "read a/b.py"
    assert len(summarize_call("run_shell", {"command": "x" * 500})) <= 140
    assert summarize_call("nope", {"a": 1}).startswith("nope")


# ---- fetch -----------------------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): ...

    def do_GET(self):
        if self.path == "/big":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"x" * 100_000)  # 10 MB offered
            except OSError:
                pass
        elif self.path == "/bin":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(b"\x89PNG")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><head><title>t</title></head><body><script>bad()</script><p>Hello world</p></body></html>")


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_fetch_refuses_private_addresses(ctx, server):
    assert "refusing" in tools.fetch_url(ctx, server + "/")
    assert "refusing" in tools.fetch_url(ctx, "http://localhost/")
    assert "refusing" in tools.fetch_url(ctx, "http://169.254.169.254/latest")
    assert "only http(s)" in tools.fetch_url(ctx, "file:///etc/passwd")


def test_fetch_html_to_text_and_paging(ctx, server):
    ctx.allow_private = True
    out = tools.fetch_url(ctx, server + "/")
    assert "Hello world" in out and "bad()" not in out and "untrusted" in out


def test_fetch_stops_at_2mb_and_pages(ctx, server):
    ctx.allow_private = True
    url, text, truncated = tools.fetch(server + "/big", True)
    assert truncated and len(text) == tools.FETCH_CAP
    out = tools.fetch_url(ctx, server + "/big", start=8000)
    assert "start=16000" in out


def test_fetch_rejects_binary_content_types(ctx, server):
    ctx.allow_private = True
    assert "not a text page" in tools.fetch_url(ctx, server + "/bin")


def test_fetch_follows_redirects_but_rechecks_each_hop():
    def handler(request):
        if request.url.host == "good.test":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        return httpx.Response(200, text="x")

    import socket
    orig = socket.getaddrinfo
    socket.getaddrinfo = lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34" if host == "good.test" else "127.0.0.1", 0))]
    try:
        with pytest.raises(ValueError, match="refusing"):
            tools.fetch("http://good.test/", transport=httpx.MockTransport(handler))
    finally:
        socket.getaddrinfo = orig


# ---- search ----------------------------------------------------------------------------------------

DDG = """<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x">Example <b>A</b></a></h2>
<a class="result__snippet" href="x">First <b>snippet</b> text</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a><a class="result__snippet">ad</a></div>
<div class="result"><a class="result__a" href="https://example.org/b">B title</a><a class="result__snippet">Second</a></div>"""


def test_parse_ddg_results_decodes_redirects_and_skips_ads():
    assert tools.parse_ddg(DDG) == [("Example A", "https://example.com/a", "First snippet text"),
                                    ("B title", "https://example.org/b", "Second")]


def test_web_search_is_keyless_by_default_and_has_no_setup_nag(ctx, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(tools, "_ddg", lambda q, n, transport=None: tools._fmt(tools.parse_ddg(DDG)[:n]))
    out = tools.web_search(ctx, "anything")
    assert "https://example.com/a" in out and "untrusted" in out


def test_web_search_failure_message_and_backend_order(ctx, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    calls = []

    def boom(q, n, transport=None):
        calls.append("ddg")
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(tools, "_ddg", boom)
    monkeypatch.setattr(tools, "_searxng", lambda base, q, n: calls.append("searx") or "searx result")
    ctx.search = {"provider": "ddg", "searxng_url": "http://s"}
    assert tools.web_search(ctx, "q") == "searx result" and calls == ["ddg", "searx"]
    ctx.search = {}
    monkeypatch.setattr(tools, "_searxng", lambda *a: "never")
    out = tools.web_search(ctx, "q")
    assert out.startswith("error: search failed") and "offline" in out and "setup" not in out
