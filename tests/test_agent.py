import json
import threading
from pathlib import Path

import pytest

from jarvoos import agent as A, tools
from jarvoos.agent import Agent, AgentEvents, INTERRUPT_NOTE, SUMMARY_PREFIX, hard_deny, assess, guard_path
from jarvoos.config import Config, ProviderSpec
from jarvoos.errors import Interrupted, ProviderError
from jarvoos.providers import RouterEvent, ToolCall, Turn, Usage


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path / "jhome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


class FakeRouter:
    """Scripted stand-in: each script item is a Turn, an exception to raise, or fn(on_text, cancel) -> Turn."""

    def __init__(self, *script, ctx=100_000):
        self.script, self.calls, self.ctx = list(script), [], ctx

    def context_tokens(self):
        return self.ctx

    def chat(self, messages, tools_, on_text, on_reasoning=None, on_event=lambda e: None, cancel=None):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools_})
        item = self.script.pop(0)
        if callable(item):
            item = item(on_text, cancel)
        if isinstance(item, BaseException):
            raise item
        if item.text:
            on_text(item.text)
        return item


def say(text, usage=None):
    return Turn(text=text, finish="stop", usage=usage, provider="local")


def call(name, args=None, id="c1", usage=None):
    return Turn(tool_calls=[ToolCall(id, name, json.dumps(args or {}))], finish="tool_calls", usage=usage, provider="local")


def make(tmp_path, *script, mode="approve", confirm=None, ctx=100_000, max_steps=25, **ev):
    spec = ProviderSpec("local", "strata", "http://x/v1", "m", context_tokens=ctx)
    cfg = Config({"local": spec}, ["local"], mode=mode, max_steps=max_steps)
    notes, asked = [], []

    def default_confirm(req):
        asked.append(req)
        return confirm(req) if callable(confirm) else (confirm or "deny")

    events = AgentEvents(on_notice=notes.append, confirm=default_confirm, **ev)
    r = FakeRouter(*script, ctx=ctx)
    a = Agent(cfg, r, events, tmp_path)
    a.notes, a.asked = notes, asked
    return a, r


def valid(messages):
    """Every tool call answered before the next non-tool message; every tool message answers a call."""
    pending = None
    for m in messages:
        if m["role"] == "tool":
            assert pending is not None and m["tool_call_id"] in pending, f"orphan tool result {m}"
            pending.discard(m["tool_call_id"])
        else:
            assert not pending, f"unanswered tool calls {pending} before {m['role']}"
            pending = {c["id"] for c in m.get("tool_calls", [])} or None
    assert not pending, f"unanswered tool calls at end: {pending}"


def run(a, text="go", cancel=None):
    return a.run(text, cancel or threading.Event())


def test_plain_reply(tmp_path):
    a, r = make(tmp_path, say("hello"))
    res = run(a, "hi")
    assert (res.text, res.stopped, res.error, res.steps) == ("hello", None, None, 1)
    assert [m["role"] for m in a.messages] == ["user", "assistant"]
    assert r.calls[0]["messages"][0]["role"] == "system" and r.calls[0]["tools"]


def test_tool_round_trip_and_events(tmp_path):
    (tmp_path / "f.txt").write_text("content")
    started, ended = [], []
    a, r = make(tmp_path, call("read_file", {"path": "f.txt"}), say("done"),
                on_tool_start=lambda n, args: started.append((n, args)), on_tool_end=lambda n, res, s: ended.append((n, res, s)))
    res = run(a)
    assert res.text == "done" and res.steps == 2
    assert started == [("read_file", {"path": "f.txt"})] and ended[0][:2] == ("read_file", "content")
    assert a.messages[-2] == {"role": "tool", "tool_call_id": "c1", "content": "content"}
    valid(a.messages)


def test_bad_json_unknown_tool_and_bad_args_go_back_to_the_model(tmp_path):
    a, r = make(tmp_path, Turn(tool_calls=[ToolCall("a", "read_file", "{nope"), ToolCall("b", "ghost", "{}"),
                                           ToolCall("c", "read_file", '{"wrong": 1}')], finish="tool_calls"), say("ok"))
    run(a)
    results = [m["content"] for m in a.messages if m["role"] == "tool"]
    assert "not a valid JSON" in results[0] and "unknown tool" in results[1] and "bad arguments" in results[2]
    valid(a.messages)


# ---- interruption and failure -----------------------------------------------------------------------

def test_interrupt_inside_a_tool_leaves_valid_history_and_next_run_works(tmp_path, monkeypatch):
    def boom(ctx, **kw):
        e = Interrupted()
        e.partial = "half the output"
        raise e

    monkeypatch.setitem(tools.TOOLS, "read_file", tools.Tool("read_file", "read", tools.TOOLS["read_file"].schema, boom, lambda a: "x"))
    two = Turn(tool_calls=[ToolCall("a", "read_file", '{"path":"x"}'), ToolCall("b", "read_file", '{"path":"y"}')], finish="tool_calls")
    a, r = make(tmp_path, two, say("second answer"))
    res = run(a, "first")
    assert res.stopped == "interrupted"
    valid(a.messages)
    results = {m["tool_call_id"]: m["content"] for m in a.messages if m["role"] == "tool"}
    assert "interrupted" in results["a"] and "half the output" in results["a"] and "not run" in results["b"]
    res = run(a, "now do this instead")
    assert res.text == "second answer" and res.stopped is None
    valid(a.messages)
    sent = r.calls[-1]["messages"]
    assert sent[-1]["content"].startswith(INTERRUPT_NOTE) and sent[-1]["content"].endswith("now do this instead")
    assert a.retry_text() == "now do this instead"


def test_interrupt_inside_the_stream_keeps_partial_text(tmp_path):
    def stream(on_text, cancel):
        on_text("partial ")
        on_text("answer")
        raise Interrupted()

    a, r = make(tmp_path, stream, say("fine"))
    res = run(a, "q")
    assert res.stopped == "interrupted" and res.text == "partial answer"
    assert a.messages[-1] == {"role": "assistant", "content": "partial answer"}
    valid(a.messages)
    assert run(a, "again").text == "fine"
    valid(a.messages)


def test_stray_keyboardinterrupt_is_treated_like_interrupted(tmp_path):
    a, _ = make(tmp_path, KeyboardInterrupt(), say("fine"))
    assert run(a).stopped == "interrupted"
    assert run(a).text == "fine"


def test_provider_error_is_returned_not_raised_and_retry_text_works(tmp_path):
    err = ProviderError("network", "down", provider="local", url="http://x/v1")
    a, _ = make(tmp_path, err)
    res = run(a, "my question")
    assert res.stopped == "error" and res.error is err
    valid(a.messages)
    assert a.retry_text() == "my question"
    assert a.undo() and a.messages == []


def test_error_after_partial_text_keeps_it(tmp_path):
    def stream(on_text, cancel):
        on_text("start")
        raise ProviderError("network", "cut")

    a, _ = make(tmp_path, stream)
    res = run(a)
    assert res.stopped == "error" and res.text == "start"
    valid(a.messages)


def test_unexpected_exception_repairs_history_then_propagates(tmp_path):
    def bad_end(name, result, secs):
        raise RuntimeError("ui bug")

    (tmp_path / "f").write_text("x")
    two = Turn(tool_calls=[ToolCall("a", "read_file", '{"path":"f"}'), ToolCall("b", "read_file", '{"path":"f"}')], finish="tool_calls")
    a, _ = make(tmp_path, two, say("ok"), on_tool_end=bad_end)
    with pytest.raises(RuntimeError):
        run(a)
    valid(a.messages)
    a.events.on_tool_end = lambda *x: None
    assert run(a).text == "ok"


def test_step_limit(tmp_path):
    (tmp_path / "f").write_text("x")
    a, r = make(tmp_path, *[call("read_file", {"path": "f"}, id=f"c{i}") for i in range(5)], max_steps=3)
    res = run(a)
    assert res.stopped == "step_limit" and res.steps == 3 and len(r.calls) == 3
    valid(a.messages)


# ---- approvals and modes ----------------------------------------------------------------------------

def shell_spy(monkeypatch):
    ran = []
    t = tools.TOOLS["run_shell"]
    monkeypatch.setitem(tools.TOOLS, "run_shell", tools.Tool("run_shell", "exec", t.schema, lambda ctx, command, **k: ran.append(command) or "exit 0", t.summarize, t.preview))
    return ran


def results(a):
    return [m["content"] for m in a.messages if m["role"] == "tool"]


def test_approve_mode_asks_and_distinguishes_declined_from_no_response(tmp_path, monkeypatch):
    ran = shell_spy(monkeypatch)
    decisions = iter(["once", "deny", None])
    a, _ = make(tmp_path, call("run_shell", {"command": "ls"}, "1"), call("run_shell", {"command": "ls"}, "2"),
                call("run_shell", {"command": "ls"}, "3"), say("end"), confirm=lambda req: next(decisions))
    run(a)
    r = results(a)
    assert ran == ["ls"] and r[0] == "exit 0"
    assert "declined" in r[1] and "No response" in r[2] and r[1] != r[2]
    assert a.asked[0].tool == "run_shell" and a.asked[0].summary == "$ ls" and a.asked[0].risk == "exec"


def test_confirm_that_raises_counts_as_no_response(tmp_path, monkeypatch):
    shell_spy(monkeypatch)

    def confirm(req):
        raise OSError("tty gone")

    a, _ = make(tmp_path, call("run_shell", {"command": "ls"}), say("end"), confirm=confirm)
    run(a)
    assert "No response" in results(a)[0]


def test_cancel_decision_aborts_the_whole_turn(tmp_path, monkeypatch):
    ran = shell_spy(monkeypatch)
    two = Turn(tool_calls=[ToolCall("a", "run_shell", '{"command":"ls"}'), ToolCall("b", "run_shell", '{"command":"pwd"}')], finish="tool_calls")
    a, r = make(tmp_path, two, say("never"), confirm="cancel")
    res = run(a)
    assert res.stopped == "cancelled" and ran == [] and len(r.calls) == 1
    valid(a.messages)
    assert run(a) is not None


def test_session_allows_the_tool_and_always_allows_one_command_prefix(tmp_path, monkeypatch):
    ran = shell_spy(monkeypatch)
    a, _ = make(tmp_path, call("run_shell", {"command": "git status"}, "1"), call("run_shell", {"command": "git status -s"}, "2"),
                call("run_shell", {"command": "git commit -m x"}, "3"), call("run_shell", {"command": "ls"}, "4"), say("end"),
                confirm=lambda req: "always")
    run(a)
    assert ran == ["git status", "git status -s", "git commit -m x", "ls"]
    assert [q.summary for q in a.asked] == ["$ git status", "$ git commit -m x", "$ ls"]  # same prefix not asked again
    a2, _ = make(tmp_path, call("run_shell", {"command": "ls"}, "1"), call("run_shell", {"command": "pwd"}, "2"), say("end"),
                 confirm="session")
    run(a2)
    assert len(a2.asked) == 1  # whole tool allowed for the session
    assert not (Path(tmp_path) / "x").exists()


def test_compound_commands_are_not_remembered_by_prefix(tmp_path, monkeypatch):
    shell_spy(monkeypatch)
    a, _ = make(tmp_path, call("run_shell", {"command": "ls && rm x"}, "1"), call("run_shell", {"command": "ls && rm y"}, "2"),
                say("end"), confirm="always")
    run(a)
    assert a.allow_prefixes == set() and a.allow_tools == set() and len(a.asked) == 2


def test_auto_mode_skips_prompts_but_dangerous_commands_still_ask_and_are_never_remembered(tmp_path, monkeypatch):
    ran = shell_spy(monkeypatch)
    a, _ = make(tmp_path, call("run_shell", {"command": "echo hi"}, "1"), call("run_shell", {"command": "git push --force"}, "2"),
                call("run_shell", {"command": "git push --force"}, "3"), say("end"), mode="auto", confirm="always")
    run(a)
    assert len(a.asked) == 2 and a.asked[0].risk == "high" and "git history" in a.asked[0].reason
    assert len(ran) == 3


def test_hard_deny_is_refused_even_in_auto_mode_without_asking(tmp_path, monkeypatch):
    ran = shell_spy(monkeypatch)
    a, _ = make(tmp_path, call("run_shell", {"command": "rm -rf /"}, "1"), call("run_shell", {"command": "rm -rf ~"}, "2"),
                call("run_shell", {"command": "dd if=/dev/zero of=/dev/sda"}, "3"), say("end"), mode="auto", confirm="once")
    run(a)
    assert ran == [] and a.asked == []
    assert all("never run" in r for r in results(a)[:3])


HARD = ["rm -rf /", "rm -fr /", "rm -r -f /", "rm -rf ~", "rm -rf ~/", "rm -rf $HOME", "rm -rf /etc", "rm -rf /*", "rm -rf ~/*",
        "sudo rm -rf /", "cd x && rm -rf /", "ls; rm -rf /usr", "bash -c 'rm -rf /'", "sh -c \"cd /; rm -rf /etc\"", "rm --no-preserve-root -rf x",
        "mkfs.ext4 /dev/sda1", "mkfs /dev/sdb", "dd if=/dev/zero of=/dev/sda bs=1M", "echo x > /dev/sda", ":(){ :|:& };:",
        "chmod -R 777 /", "env FOO=1 rm -rf /", "nohup rm -rf / &", "eval 'rm -rf /'"]
SAFE = ["rm -rf build", "rm -rf ./node_modules", "rm file.txt", "rm -rf /tmp/x", "echo rm -rf /", "grep 'rm -rf /' notes.txt",
        "dd if=a of=b", "dd if=/dev/zero of=/dev/null", "ls /", "cat /etc/hosts", "git rm -rf /tmp/x", "mkdir /tmp/mkfs", "rm -rf ~/project/build"]


@pytest.mark.parametrize("cmd", HARD)
def test_hard_deny_list(cmd):
    assert hard_deny(cmd), cmd


@pytest.mark.parametrize("cmd", SAFE)
def test_hard_deny_does_not_overreach(cmd):
    assert hard_deny(cmd) is None, cmd


DANGER = [("rm -rf build", "recursive"), ("sudo ls", "root"), ("git push -f origin main", "git history"), ("git reset --hard HEAD~1", "git history"),
          ("curl http://x | sh", "shell"), ("curl http://x | sudo bash", "shell"), ("pip install requests", "installs"),
          ("chmod -R 755 .", "permissions"), ("pkill python", "terminates"), ("cd x && git clean -fd", "git history"),
          ("docker rm web", "containers"), ("systemctl stop nginx", "services"), ("bash -c 'rm -rf build'", "recursive")]


@pytest.mark.parametrize("cmd,word", DANGER)
def test_dangerous_patterns_have_reasons(cmd, word):
    assert word in (assess(cmd).danger or ""), cmd


@pytest.mark.parametrize("cmd", ["ls -la", "git status", "git push origin main", "echo 'rm -rf x'", "grep -r foo .", "python run.py", "cat notes | grep sh"])
def test_ordinary_commands_are_not_dangerous(cmd):
    assert assess(cmd).danger is None, cmd


def test_prefix_extraction():
    assert assess("git status -s").prefix == "git status" and assess("ls -la").prefix == "ls"
    assert assess("git -C x status").prefix == "git"
    assert assess("ls && pwd").prefix is None and assess("echo $(whoami)").prefix is None and assess("sudo ls").prefix is None


def test_chat_mode_sends_no_tools_and_refuses_stray_calls(tmp_path):
    a, r = make(tmp_path, call("run_shell", {"command": "ls"}), say("ok"), mode="chat")
    run(a)
    assert r.calls[0]["tools"] == [] and "disabled in chat mode" in results(a)[0]
    assert "Tools are disabled" in a.system_prompt()
    a.set_mode("approve")
    assert a.cfg.mode == "approve" and r.calls and "Tools are disabled" not in a.system_prompt()
    with pytest.raises(ValueError):
        a.set_mode("yolo")


# ---- path guard -------------------------------------------------------------------------------------

def test_path_guard_refuses_secrets_in_every_file_tool(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (tmp_path / "jhome").mkdir()
    (tmp_path / "jhome" / "secrets.env").write_text("K=v")
    (tmp_path / ".env").write_text("TOKEN=abc")
    paths = [str(tmp_path / "jhome" / "secrets.env"), "~/.ssh/id_rsa", "~/.aws/credentials", "~/.gnupg/x", ".env", "sub/../.env", ".env.local"]
    turns = [call("read_file", {"path": p}, f"r{i}") for i, p in enumerate(paths)]
    turns.append(call("write_file", {"path": str(tmp_path / "jhome" / "config.toml"), "content": "x"}, "w"))
    turns.append(say("end"))
    a, _ = make(tmp_path, *turns, mode="auto")
    run(a)
    rs = results(a)
    assert all(r.startswith("refused:") for r in rs), rs
    assert not (tmp_path / "jhome" / "config.toml").exists()
    valid(a.messages)


def test_path_guard_allows_ordinary_and_example_files(tmp_path):
    assert guard_path(tmp_path / "a.py") is None
    assert guard_path(tmp_path / ".env.example") is None
    assert guard_path(tmp_path / ".venv" / "x") is None
    assert guard_path(tmp_path / "env") is None


def test_path_guard_for_shell_commands(tmp_path, monkeypatch):
    ran = shell_spy(monkeypatch)
    cmds = ["cat ~/.ssh/id_rsa", "cat .env", "cat secrets.env", f"cat {tmp_path}/jhome/config.toml", "tar c ~/.aws", "cat .env.local"]
    a, _ = make(tmp_path, *[call("run_shell", {"command": c}, f"c{i}") for i, c in enumerate(cmds)], say("end"), mode="auto")
    run(a)
    assert ran == [] and all(r.startswith("refused:") for r in results(a))
    ok = ["cat .env.example", "source .venv/bin/activate", "ls .envrc", "echo hi"]
    a, _ = make(tmp_path, *[call("run_shell", {"command": c}, f"c{i}") for i, c in enumerate(ok)], say("end"), mode="auto")
    run(a)
    assert ran == ok


def test_shell_runs_for_real_with_persistent_cwd_and_live_output(tmp_path):
    (tmp_path / "d").mkdir()
    live = []
    a, _ = make(tmp_path, call("run_shell", {"command": "cd d"}, "1"), call("run_shell", {"command": "pwd; echo out"}, "2"), say("end"),
                mode="auto", on_tool_output=live.append)
    run(a)
    assert results(a)[1].splitlines()[1].endswith("/d") and "out\n" in "".join(live)
    assert a.cwd == (tmp_path / "d").resolve()


# ---- compaction -------------------------------------------------------------------------------------

def history(n, size=3000):
    msgs = []
    for i in range(n):
        msgs += [{"role": "user", "content": f"request {i}"},
                 {"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"t{i}", "content": f"output {i} " + "z" * size},
                 {"role": "assistant", "content": f"answer {i}"}]
    return msgs


def test_manual_compact_shrinks_tool_outputs_then_summarises(tmp_path):
    a, r = make(tmp_path, say("THE SUMMARY"), ctx=20_000)
    a.messages[:] = history(6)
    res = a.compact("the build system")
    assert res.changed and res.summarised and res.after < res.before
    assert a.messages[0]["content"].startswith(SUMMARY_PREFIX) and "THE SUMMARY" in a.messages[0]["content"]
    assert a.messages[1]["role"] == "assistant" and a.messages[2]["content"] == "request 4"
    assert len(a.messages) == 2 + 8
    valid(a.messages)
    assert any(n.startswith("[context compacted:") and "->" in n and n.endswith("tokens]") for n in a.notes)
    ask = r.calls[0]
    assert ask["tools"] == [] and "the build system" in ask["messages"][1]["content"] and "request 0" in ask["messages"][1]["content"]
    assert a.retry_text() == "request 5"  # the summary is not a retryable request


def test_compact_step_one_alone_shrinks_old_tool_output(tmp_path):
    a, r = make(tmp_path, ctx=1_000_000)
    a.messages[:] = history(3)
    changed = a._compact(None, force=False, tail_users=2)
    assert changed.changed and not changed.summarised and r.calls == []
    assert "old output cut" in a.messages[2]["content"] and "old output cut" not in a.messages[-2]["content"]
    valid(a.messages)


def test_compact_nothing_to_do(tmp_path):
    a, _ = make(tmp_path)
    a.messages[:] = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert not a.compact().changed


def test_summary_failure_falls_back_to_a_list_of_requests(tmp_path):
    a, _ = make(tmp_path, ProviderError("network", "down"), ctx=20_000)
    a.messages[:] = history(6)
    res = a.compact()
    assert res.changed and res.error and "request 0" in a.messages[0]["content"]
    valid(a.messages)


def test_auto_compaction_uses_real_prompt_tokens(tmp_path):
    a, r = make(tmp_path, say("first", Usage(18_000, 50)), say("SUMMARY"), say("second", Usage(3000, 20)), ctx=20_000)
    a.messages[:] = history(6, size=100)
    run(a, "q")
    assert len(r.calls) == 1
    run(a, "q2")
    assert len(r.calls) == 3 and r.calls[1]["tools"] == [] and "request 0" in r.calls[1]["messages"][1]["content"]
    assert any("context compacted" in n for n in a.notes)
    assert a.messages[0]["content"].startswith(SUMMARY_PREFIX)
    valid(a.messages)


def test_no_auto_compaction_below_threshold(tmp_path):
    a, r = make(tmp_path, say("a", Usage(1000, 10)), say("b"), ctx=20_000)
    run(a)
    run(a)
    assert len(r.calls) == 2 and a.notes == []


def test_context_error_triggers_compaction_and_retry_not_failover(tmp_path):
    ctx_err = lambda: ProviderError("context", "too long", provider="local")
    a, r = make(tmp_path, ctx_err(), say("SUMMARY"), say("worked"), ctx=20_000)
    a.messages[:] = history(6)
    res = run(a, "big one")
    assert res.text == "worked" and res.stopped is None
    assert len(r.calls) == 3 and r.calls[1]["tools"] == []
    assert any("context compacted" in n for n in a.notes)
    valid(a.messages)


def test_context_error_retries_at_most_twice(tmp_path):
    ctx_err = lambda: ProviderError("context", "too long", provider="local")
    a, r = make(tmp_path, ctx_err(), say("S1"), ctx_err(), say("S2"), ctx_err(), ctx=20_000)
    a.messages[:] = history(8)
    res = run(a, "big one")
    assert res.stopped == "error" and res.error.kind == "context"
    assert sum(1 for c in r.calls if c["tools"] == [] and "Conversation to summarise" in c["messages"][-1]["content"]) <= 2
    valid(a.messages)


def test_context_error_with_nothing_to_compact_surfaces(tmp_path):
    a, _ = make(tmp_path, ProviderError("context", "too long"))
    res = run(a, "tiny")
    assert res.stopped == "error" and res.error.kind == "context"


# ---- misc -------------------------------------------------------------------------------------------

def test_undo_and_retry_text(tmp_path):
    a, _ = make(tmp_path, say("one"), say("two"))
    assert not a.undo() and a.retry_text() is None
    run(a, "first")
    run(a, "second")
    assert a.retry_text() == "second"
    assert a.undo() and [m["content"] for m in a.messages] == ["first", "one"]
    assert a.undo() and a.messages == []


def test_usage_snapshot(tmp_path):
    a, _ = make(tmp_path, say("hi", Usage(500, 40)), ctx=2000)
    s0 = a.usage_snapshot()
    assert s0.context_tokens == 2000 and s0.prompt_tokens >= 0
    run(a, "q")
    s = a.usage_snapshot()
    assert s.prompt_tokens == 540 and s.completion_tokens == 40 and s.pct == 27.0


def test_system_prompt_contents(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Always use tabs.")
    a, _ = make(tmp_path)
    p = a.system_prompt()
    assert str(tmp_path) in p and "Linux" in p and "Always use tabs." in p and "web-research" in p
    import time
    assert time.strftime("%Y-%m-%d") in p


def test_router_events_reach_the_ui(tmp_path):
    seen = []
    a, r = make(tmp_path, on_router=seen.append)

    def script(on_text, cancel):
        return say("x")

    r.script = [script]
    r.chat = lambda messages, tools_, on_text, on_reasoning=None, on_event=lambda e: None, cancel=None: (on_event(RouterEvent("failover", "a", "b")), say("x"))[1]
    run(a)
    assert [e.kind for e in seen] == ["failover"]
