import sys
from dataclasses import make_dataclass
from types import SimpleNamespace as NS

import pytest

from jarvoos import cli, config, __version__


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
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "home" / "config.toml", raising=False)
    return tmp_path


def make_cfg():
    return NS(providers={"local": NS(name="local", model="qwen3", api_key_env="")}, chain=["local"], mode="approve",
              max_steps=25)


@pytest.fixture
def setup_calls(monkeypatch):
    calls = []
    state = {"configured": False}
    def run(first_run=False, section=None, io=None):
        calls.append(("run", first_run, section))
        state["configured"] = True
        return True
    fake_module(monkeypatch, "setup", run=run, noninteractive_help=lambda: "HOW TO CONFIGURE")
    monkeypatch.setattr(config, "is_configured", lambda: state["configured"], raising=False)
    monkeypatch.setattr(config, "load", make_cfg, raising=False)
    return calls


@pytest.fixture
def repl_calls(monkeypatch):
    calls = []
    fake_module(monkeypatch, "repl", repl=lambda cfg, **kw: calls.append(kw) or 0)
    return calls


def test_first_run_tty_offers_setup_then_enters_repl_in_one_process(env, setup_calls, repl_calls, monkeypatch):
    monkeypatch.setattr(cli, "_is_tty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or "")
    assert cli.main([]) == 0
    assert "Run setup now? [Y/n]" in prompts[0]
    assert setup_calls == [("run", True, None)]
    assert len(repl_calls) == 1


def test_first_run_decline(env, setup_calls, repl_calls, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_is_tty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda p="": "n")
    assert cli.main([]) == 1
    assert "jarvoos setup" in capsys.readouterr().err
    assert setup_calls == [] and repl_calls == []


def test_first_run_no_tty_prints_help_and_exits_1(env, setup_calls, repl_calls, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_is_tty", lambda: False)
    assert cli.main([]) == 1
    assert "HOW TO CONFIGURE" in capsys.readouterr().err
    assert setup_calls == [] and repl_calls == []


def test_setup_not_saved_exits_1(env, setup_calls, repl_calls, monkeypatch):
    monkeypatch.setattr(cli, "_is_tty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda p="": "y")
    fake_module(monkeypatch, "setup", run=lambda **kw: False, noninteractive_help=lambda: "")
    assert cli.main([]) == 1 and repl_calls == []


def test_configured_goes_straight_to_repl_with_flags(env, setup_calls, repl_calls, monkeypatch):
    monkeypatch.setattr(config, "is_configured", lambda: True, raising=False)
    assert cli.main(["-c", "--auto-approve"]) == 0
    assert repl_calls[0]["continue_last"] and repl_calls[0]["auto_approve"] and repl_calls[0]["resume"] is None
    cli.main(["--resume", "2026"])
    assert repl_calls[1]["resume"] == "2026"
    assert setup_calls == []


def test_setup_subcommand_passes_section(env, setup_calls):
    assert cli.main(["setup", "fallback"]) == 0
    assert setup_calls == [("run", False, "fallback")]


def test_version_single_sourced(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out
    import tomllib
    from pathlib import Path
    proj = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert "version" in proj["project"]["dynamic"] and "version" not in proj["project"]
    assert proj["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "jarvoos.__version__"}


def test_config_path_and_errors(env, capsys, monkeypatch):
    assert cli.main(["config", "path"]) == 0
    assert str(env / "home" / "config.toml") in capsys.readouterr().out
    monkeypatch.setattr(config, "get_value", lambda k: (_ for _ in ()).throw(KeyError(k)), raising=False)
    assert cli.main(["config", "get", "nope"]) == 2


def fake_agent(monkeypatch, run):
    AE = make_dataclass("AgentEvents", ["on_text", "on_reasoning", "on_tool_start", "on_tool_end",
                                         "on_router", "on_notice", "confirm"])
    class A:
        def __init__(self, cfg, router, events, cwd, messages=None):
            self.events, self.cfg = events, cfg
        def run(self, text, cancel):
            return run(self, text)
    fake_module(monkeypatch, "agent", AgentEvents=AE, Agent=A)
    fake_module(monkeypatch, "providers", Router=lambda cfg, events=None: NS())


def test_one_shot_prints_answer_and_hides_think(env, setup_calls, monkeypatch, capsys):
    monkeypatch.setattr(config, "is_configured", lambda: True, raising=False)
    def run(a, text):
        a.events.on_text("<think>x</think>ans")
        a.events.on_text("wer")
        return NS(text="answer", stopped=None, error=None, steps=1)
    fake_agent(monkeypatch, run)
    assert cli.main(["-z", "hi"]) == 0
    assert capsys.readouterr().out == "answer\n"


def test_one_shot_approval_needed_exits_3(env, setup_calls, monkeypatch, capsys):
    monkeypatch.setattr(config, "is_configured", lambda: True, raising=False)
    decisions = []
    def run(a, text):
        decisions.append(a.events.confirm(NS(tool="shell", summary="rm x", reason="", risk="exec", preview="")))
        return NS(text="", stopped=None, error=None, steps=1)
    fake_agent(monkeypatch, run)
    assert cli.main(["-z", "do"]) == 3
    assert decisions == ["deny"] and "approval required for shell" in capsys.readouterr().err


def test_one_shot_auto_approve_sets_auto_mode(env, setup_calls, monkeypatch):
    monkeypatch.setattr(config, "is_configured", lambda: True, raising=False)
    seen = []
    fake_agent(monkeypatch, lambda a, t: seen.append(a.cfg.mode) or NS(text="", stopped=None, error=None, steps=1))
    assert cli.main(["-z", "x", "--auto-approve"]) == 0 and seen == ["auto"]


def test_one_shot_step_limit_and_interrupt(env, setup_calls, monkeypatch, capsys):
    monkeypatch.setattr(config, "is_configured", lambda: True, raising=False)
    fake_agent(monkeypatch, lambda a, t: NS(text="part", stopped="step_limit", error=None, steps=25))
    assert cli.main(["-z", "x"]) == 1
    assert "step limit" in capsys.readouterr().err
    class Interrupted(KeyboardInterrupt): ...
    def raiser(a, t): raise Interrupted
    fake_agent(monkeypatch, raiser)
    assert cli.main(["-z", "x"]) == 130


def test_one_shot_no_tty_unconfigured_never_prompts(env, setup_calls, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_is_tty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda p="": pytest.fail("must not prompt"))
    assert cli.main(["-z", "x"]) == 1
    assert "HOW TO CONFIGURE" in capsys.readouterr().err


def test_debug_log_masks_keys(env, monkeypatch):
    monkeypatch.setenv("MYKEY", "sk-supersecretvalue123")
    monkeypatch.setattr(config, "load", lambda: NS(providers={"p": NS(api_key_env="MYKEY")}), raising=False)
    monkeypatch.setattr(config, "get_secret", lambda n: "sk-supersecretvalue123", raising=False)
    monkeypatch.setattr(config, "mask", lambda s: s[:3] + "...", raising=False)
    events, path = cli.enable_debug()
    events("request", {"headers": {"Authorization": "Bearer sk-supersecretvalue123"}, "url": "http://x"})
    text = path.read_text()
    assert "supersecret" not in text and "http://x" in text and oct(path.stat().st_mode & 0o777) == "0o600"
