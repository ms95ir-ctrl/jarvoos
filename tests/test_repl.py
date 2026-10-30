import io
import os
import signal
import threading
from types import SimpleNamespace as NS

import pytest
from rich.console import Console

from jarvoos import config, repl as R
from jarvoos.sessions import SessionStore
from jarvoos.ui import UI


class FakeRouter:
    def __init__(self, cfg=None, events=None):
        self.active, self.failed_over, self.pins = "local", False, []

    def pin(self, name):
        self.pins.append(name)

    def reset_cooldowns(self):
        self.pins.append("reset")


class FakeAgent:
    def __init__(self, cfg=None, router=None, events=None, cwd=None, messages=None):
        self.messages = list(messages or [])
        self.events, self.prompt_tokens, self.script = events, 0, None

    def run(self, text, cancel):
        self.messages.append({"role": "user", "content": text})
        if self.script:
            return self.script(self, text, cancel)
        self.messages.append({"role": "assistant", "content": f"echo {text}"})
        return NS(text=f"echo {text}", stopped=None, error=None, steps=1)

    def undo(self):
        if not self.messages:
            return False
        while self.messages and self.messages.pop().get("role") != "user":
            pass
        return True

    def retry_text(self):
        return next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), None)

    def compact(self, focus=None):
        self.messages[:] = self.messages[-1:]

    def set_mode(self, mode):
        self.mode = mode

    def usage_snapshot(self):
        return NS(prompt_tokens=self.prompt_tokens, completion_tokens=10, context_tokens=1000, pct=0)


def make_cfg():
    return NS(providers={"local": NS(name="local", kind="strata", model="qwen3", context_tokens=1000, tools=True,
                                     api_key_env=""),
                         "openrouter": NS(name="openrouter", kind="openrouter", model="dsv4", context_tokens=0,
                                          tools=True, api_key_env="")},
              chain=["local", "openrouter"], mode="approve", max_steps=25, compact_at=0.75, verbose=False)


def fake_module(monkeypatch, name, **attrs):
    import sys, types
    import jarvoos
    mod = types.ModuleType(f"jarvoos.{name}")
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, f"jarvoos.{name}", mod)
    monkeypatch.setattr(jarvoos, name, mod, raising=False)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state", raising=False)
    monkeypatch.setattr(R, "load_skills", lambda: [NS(name="review", description="review code", body="BODY")])
    return tmp_path


def make_state(env, agent=None):
    cfg = make_cfg()
    buf = io.StringIO()
    ui = UI(cfg, console=Console(file=buf, force_terminal=False, width=120), tty=False)
    store = SessionStore(env / "sessions")
    meta = store.new(env, "local", "qwen3")
    st = R.ReplState(cfg, agent or FakeAgent(), FakeRouter(), store, ui, meta)
    st.skills = R.load_skills()
    st.buf = buf
    return st


def test_every_command_has_help_and_names_are_unique():
    names = [n for c in R.COMMANDS for n in (c.name, *c.aliases)]
    assert len(names) == len(set(names))
    assert all(c.help.strip() and callable(c.handler) for c in R.COMMANDS)
    for required in "help exit new clear history retry undo title sessions resume compress model mode status tools " \
                    "skills reload config doctor verbose".split():
        assert R.find_command(required), required


def test_help_lists_every_command_and_skills(env):
    st = make_state(env)
    R.handle_line(st, "/help")
    out = st.buf.getvalue()
    assert all(f"/{c.name}" in out for c in R.COMMANDS) and "/review" in out


def test_unknown_command_suggests_closest_and_sends_nothing(env):
    st = make_state(env)
    R.handle_line(st, "/sesions")
    assert "did you mean /sessions?" in st.buf.getvalue()
    R.handle_line(st, "/zzzzzzzz")
    assert "/help lists commands" in st.buf.getvalue()
    assert st.agent.messages == []


def test_path_like_input_is_a_normal_message(env):
    st = make_state(env)
    R.handle_line(st, "/etc/hosts what is this")
    assert st.agent.messages[0]["content"] == "/etc/hosts what is this"


def test_skill_command_runs_a_turn(env):
    st = make_state(env)
    R.handle_line(st, "/review src/x.py")
    text = st.agent.messages[0]["content"]
    assert "BODY" in text and text.endswith("src/x.py")


def test_turn_is_persisted_and_title_set(env):
    st = make_state(env)
    R.handle_line(st, "hello world")
    meta, msgs = st.store.load(st.session.id)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and meta.title == "hello world"


def test_undo_and_retry_rewrite_the_session(env):
    st = make_state(env)
    R.handle_line(st, "first")
    R.handle_line(st, "/retry")
    _, msgs = st.store.load(st.session.id)
    assert [m["content"] for m in msgs] == ["first", "echo first"]
    R.handle_line(st, "/undo")
    assert st.store.load(st.session.id)[1] == []


def test_compress_rewrites_session(env):
    st = make_state(env)
    R.handle_line(st, "a")
    R.handle_line(st, "b")
    R.handle_line(st, "/compress")
    assert len(st.store.load(st.session.id)[1]) == 1


def test_valid_len_cuts_unanswered_tool_calls():
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "x"}},
                                                               {"id": "2", "function": {"name": "y"}}]}
    msgs = [{"role": "user", "content": "q"}, call, {"role": "tool", "tool_call_id": "1", "content": "r"}]
    assert R._valid_len(msgs) == 1
    assert R._valid_len([*msgs, {"role": "tool", "tool_call_id": "2", "content": "r"}]) == 4


def test_model_pin_and_unknown(env):
    st = make_state(env)
    R.handle_line(st, "/model openrouter")
    assert st.router.pins == ["openrouter"] and st.store.load(st.session.id)[0].provider == "openrouter"
    R.handle_line(st, "/model opnrouter")
    assert "did you mean openrouter" in st.buf.getvalue()
    R.handle_line(st, "/model auto")
    assert st.router.pins[-2:] == [None, "reset"]


def test_mode_command(env):
    st = make_state(env)
    R.handle_line(st, "/mode auto")
    assert st.cfg.mode == "auto" and st.agent.mode == "auto"
    R.handle_line(st, "/mode bogus")
    assert st.cfg.mode == "auto"


def test_resume_restores_messages_and_provider_pin(env):
    st = make_state(env)
    other = st.store.new(env, "openrouter", "dsv4")
    hist = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "older"}]
    for m in hist:
        st.store.append(other.id, m)
    R.handle_line(st, f"/resume {other.id}")
    assert st.agent.messages == hist and st.session.id == other.id
    assert st.router.pins == ["openrouter"] and st.pinned == "openrouter"
    R.handle_line(st, "next")
    assert len(st.store.load(other.id)[1]) == 4


def test_toolbar_ladder_and_fallback_marker(env):
    st = make_state(env)
    def text(parts): return "".join(t for _, t in parts)
    st.agent.prompt_tokens = 100
    parts = R.toolbar(st)
    assert ("fg:ansigreen", " ctx 10% ") in parts and "FALLBACK" not in text(parts) and "local:qwen3" in text(parts)
    st.agent.prompt_tokens = 500
    assert any(s == "fg:ansiyellow" for s, _ in R.toolbar(st))
    st.agent.prompt_tokens = 900
    assert any(s == "fg:ansired" for s, _ in R.toolbar(st))
    st.router.failed_over, st.router.active = True, "openrouter"
    parts = R.toolbar(st)
    assert parts[0][1] == " FALLBACK: openrouter " and "bold" in parts[0][0]
    assert "openrouter:dsv4" in text(parts)


def test_ctrl_c_mid_turn_keeps_history_valid(env):
    def script(agent, text, cancel):
        os.kill(os.getpid(), signal.SIGINT)
        assert cancel.wait(2)
        agent.messages.append({"role": "assistant", "content": "partial"})
        return NS(text="partial", stopped="interrupted", error=None, steps=1)
    st = make_state(env)
    st.agent.script = script
    old = signal.getsignal(signal.SIGINT)
    R.handle_line(st, "do it")
    assert signal.getsignal(signal.SIGINT) is old
    assert st.cancel.is_set() and not st.done
    msgs = st.store.load(st.session.id)[1]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["content"] == "partial"
    assert "stopped" in st.buf.getvalue()
    st.agent.script = None
    R.handle_line(st, "again")  # next turn starts with a clean cancel flag
    assert not st.cancel.is_set()


def test_second_ctrl_c_mid_turn_force_exits(env, monkeypatch):
    class Exit(BaseException):
        pass
    def boom(st):
        raise Exit
    monkeypatch.setattr(R, "_force_exit", boom)
    def script(agent, text, cancel):
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        cancel.wait(1)
    st = make_state(env)
    st.agent.script = script
    with pytest.raises(Exit):
        R.handle_line(st, "x")


def test_ctrl_c_at_prompt_needs_two_presses(env):
    st = make_state(env)
    lines = iter([KeyboardInterrupt, "hi", KeyboardInterrupt, KeyboardInterrupt, "never"])
    def read():
        v = next(lines)
        if v is KeyboardInterrupt:
            raise v
        return v
    R.loop(st, read)
    assert [m["content"] for m in st.agent.messages if m["role"] == "user"] == ["hi"]
    assert next(lines) == "never"


def test_ctrl_d_exits_loop(env):
    st = make_state(env)
    def read(): raise EOFError
    R.loop(st, read)


def test_exit_command_stops_loop(env):
    st = make_state(env)
    R.loop(st, iter(["/exit", "nope"]).__next__)
    assert st.agent.messages == []


def test_tips_rotate_and_hints_show_once(env):
    seen = {R.next_tip() for _ in range(len(R.TIPS))}
    assert len(seen) == len(R.TIPS) and 8 <= len(R.TIPS) <= 10
    st = make_state(env)
    R.hint(st, "alt_enter")
    R.hint(st, "alt_enter")
    assert st.buf.getvalue().count("Alt+Enter inserts") == 1
    assert "alt_enter" in config.STATE_DIR.joinpath("seen.json").read_text()


def test_repl_end_to_end_with_continue(env, monkeypatch):
    from dataclasses import make_dataclass
    AE = make_dataclass("AgentEvents", ["on_text", "on_reasoning", "on_tool_start", "on_tool_end",
                                         "on_router", "on_notice", "confirm"])
    fake_module(monkeypatch, "agent", AgentEvents=AE, Agent=FakeAgent)
    fake_module(monkeypatch, "providers", Router=FakeRouter)
    monkeypatch.setattr(R, "UI", lambda cfg: UI(cfg, console=Console(file=io.StringIO()), tty=False))
    scripts = [["hello", "/exit"], ["/history", "/exit"]]
    captured = []
    def fake_reader(st):
        captured.append(st)
        it = iter(scripts.pop(0))
        return lambda: next(it)
    monkeypatch.setattr(R, "make_reader", fake_reader)
    assert R.repl(make_cfg()) == 0
    first = captured[0].session.id
    assert R.repl(make_cfg(), continue_last=True) == 0
    st2 = captured[1]
    assert st2.session.id == first
    assert [m["content"] for m in st2.agent.messages] == ["hello", "echo hello"]
    assert R.repl(make_cfg(), resume="nope") == 2
