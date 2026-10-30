import io
import os
import sys
from types import SimpleNamespace as NS

import pytest
from rich.console import Console

from jarvoos import ui as ui_mod
from jarvoos.ui import UI, ThinkFilter, sanitize


def make_ui(tty=False, **cfg):
    spec = NS(name="local", model="qwen3", context_tokens=131072)
    c = NS(providers={"local": spec, "openrouter": NS(name="openrouter", model="deepseek-v4-flash", context_tokens=0)},
           chain=["local", "openrouter"], mode="approve", verbose=False, **cfg)
    buf = io.StringIO()
    return UI(c, console=Console(file=buf, force_terminal=False, width=100), tty=tty), buf


def feed_all(chunks):
    f = ThinkFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_think_filter_basic_and_split_tags():
    assert feed_all(["<think>secret</think>\n\nHello"]) == "Hello"
    assert feed_all(["<thi", "nk>sec", "ret</th", "ink>Hel", "lo"]) == "Hello"
    assert feed_all(["a < b and <b>x</b>"]) == "a < b and <b>x</b>"
    assert feed_all(["text<think>never closed"]) == "text"
    assert feed_all(["ends with <thi"]) == "ends with <thi"


def test_sanitize_strips_escapes_and_control_chars():
    assert sanitize("a\x1b[31mred\x1b[0m\x07b\x00c\td\r\ne") == "aredbc\td\ne"
    assert sanitize("\x1b]0;title\x07ok") == "ok"
    assert sanitize("keep\nlines\tand tabs") == "keep\nlines\tand tabs"


def test_text_stream_filters_and_sanitizes():
    u, buf = make_ui()
    for chunk in ["<think>hm</think>Hel", "lo \x1b[2J", "world"]:
        u.text(chunk)
    u.end_turn()
    assert buf.getvalue() == "Hello world\n"


def test_tool_preview_truncates_unless_verbose():
    u, buf = make_ui()
    u.tool_end("read_file", "\n".join(f"line{i}" for i in range(10)), 0.42)
    out = buf.getvalue()
    assert "10 lines in 0.4s" in out and "line3" in out and "line4" not in out and "/verbose" in out
    u.verbose = True
    buf.truncate(0)
    u.tool_end("read_file", "\n".join(f"line{i}" for i in range(10)), 0.1)
    assert "line9" in buf.getvalue()


def test_error_renders_friendly_and_details():
    u, buf = make_ui()
    u.error("OpenRouter rejected the key.", "401 invalid")
    assert buf.getvalue().splitlines() == ["OpenRouter rejected the key.", "Details: 401 invalid"]


def test_router_events_render_failover_and_restart():
    u, buf = make_ui()
    u.router_event(NS(kind="failover", from_="local", to="openrouter", error=None))
    u.router_event(NS(kind="restart", from_="local", to="openrouter", error=None))
    out = buf.getvalue()
    assert "local unreachable -> using openrouter (deepseek-v4-flash) for this turn" in out
    assert "--- restarting on openrouter ---" in out


def test_restart_discards_partial_think_state():
    u, _ = make_ui()
    u.text("<think>unfinished")
    u.router_event(NS(kind="restart", from_="a", to="openrouter", error=None))
    u.text("fresh")
    assert u._answer == ["fresh"]


def fake_module(monkeypatch, name, **attrs):
    import sys, types
    import jarvoos
    mod = types.ModuleType(f"jarvoos.{name}")
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, f"jarvoos.{name}", mod)
    monkeypatch.setattr(jarvoos, name, mod, raising=False)
    return mod


REQ = NS(tool="shell", summary="rm x", reason="", risk="exec", preview="line1\nline2")


@pytest.fixture
def tty_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", NS(isatty=lambda: True, readline=lambda: "", fileno=lambda: 0))


@pytest.mark.parametrize("answer,expected", [("", "once"), ("o", "once"), ("s", "session"), ("A", "always"),
                                              ("d", "deny"), ("c", "cancel"), ("always", "always")])
def test_approval_parsing(tty_stdin, monkeypatch, answer, expected):
    u, _ = make_ui(tty=True)
    monkeypatch.setattr(u, "_read_line", lambda t: answer.lower())
    assert u.approval_prompt(REQ) == expected


def test_approval_reprompts_then_denies(tty_stdin, monkeypatch):
    u, _ = make_ui(tty=True)
    answers = iter(["huh", "what", "??"])
    monkeypatch.setattr(u, "_read_line", lambda t: next(answers))
    assert u.approval_prompt(REQ) == "deny"


def test_approval_timeout_fails_closed(tty_stdin, monkeypatch):
    u, buf = make_ui(tty=True)
    monkeypatch.setattr(u, "_read_line", lambda t: None)
    assert u.approval_prompt(REQ) == "deny"
    assert "denied" in buf.getvalue()


def test_approval_non_tty_denies():
    u, _ = make_ui(tty=False)
    assert u.approval_prompt(REQ) == "deny"


def test_read_line_real_timeout_and_input(monkeypatch):
    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r))
    u, _ = make_ui(tty=True)
    assert u._read_line(0.3) is None
    os.write(w, b"S\n")
    assert u._read_line(1) == "s"


def test_cancel_event_aborts_approval_wait(monkeypatch):
    import threading
    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r))
    u, _ = make_ui(tty=True)
    u.cancel = threading.Event()
    u.cancel.set()
    assert u._read_line(5) is None


def test_banner_and_tokens_format():
    u, buf = make_ui()
    u.banner(NS(provider="local", cwd="/w", id="20261030-1412"), "a tip", 3)
    out = buf.getvalue()
    assert "qwen3" in out and "ctx 131k" in out and "3 skills" in out and "a tip" in out and "Alt+Enter" in out
    assert ui_mod.fmt_tokens(999) == "999" and ui_mod.fmt_tokens(131072) == "131k"


def test_events_wires_callbacks(monkeypatch):
    from dataclasses import make_dataclass
    AE = make_dataclass("AgentEvents", ["on_text", "on_reasoning", "on_tool_start", "on_tool_end",
                                         "on_router", "on_notice", "confirm"])
    fake_module(monkeypatch, "agent", AgentEvents=AE)
    u, buf = make_ui()
    ev = u.events()
    ev.on_text("hi")
    ev.on_notice("[context compacted: 38k -> 11k tokens]")
    assert "hi" in buf.getvalue() and "compacted" in buf.getvalue() and ev.confirm == u.approval_prompt
