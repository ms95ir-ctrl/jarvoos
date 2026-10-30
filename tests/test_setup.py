import stat
import sys
import types
from dataclasses import dataclass

import pytest

from jarvoos import config, detect, setup
from test_detect import LLAMA, STRATA, fake_servers

GOOD = "sk-or-v1-goodkey123"


class ScriptIO:
    """Feeds scripted answers in order and fails loudly if the wizard asks something unexpected."""
    def __init__(self, *script):
        self.script, self.said, self.asked = list(script), [], []

    def _next(self, kind, prompt):
        self.asked.append((kind, prompt))
        assert self.script, f"unexpected {kind}: {prompt}"
        k, v = self.script.pop(0)
        if k == "interrupt":
            raise KeyboardInterrupt
        assert k == kind, f"wanted {k!r} but wizard asked {kind}: {prompt}"
        return v

    def ask(self, prompt, default=""):
        return self._next("ask", prompt) or default

    def ask_secret(self, prompt):
        return self._next("secret", prompt)

    def choose(self, prompt, options, default=0):
        v = self._next("choose", prompt)
        return default if v is None else v

    def confirm(self, prompt, default=True):
        v = self._next("confirm", prompt)
        return default if v is None else v

    def say(self, msg, style=""):
        self.said.append(msg)

    @property
    def text(self):
        return "\n".join(self.said)


@dataclass
class Probe:
    ok: bool = True
    chat_ok: bool = True
    tools_ok: bool = True
    ttft_ms: int = 310
    error: object = None
    hint: str = ""


@dataclass
class Err:
    kind: str = "network"
    raw: str = "connect failed"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path))
    for k in ("OPENROUTER_API_KEY", "JARVOOS_DETECT_HOSTS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(setup, "_sleep", lambda s: None)
    monkeypatch.setattr(setup, "LOAD_WAIT_S", 2)
    monkeypatch.setattr(detect, "_cache", {"t": 0.0, "models": None})
    calls = []
    results = {}

    def probe(spec, key, *, with_tools=True):
        calls.append((spec.name, key, with_tools))
        r = results.get(spec.name, Probe())
        return r[0] if isinstance(r, list) and len(r) == 1 else (r.pop(0) if isinstance(r, list) else r)

    providers = types.SimpleNamespace(probe=probe)
    errors = types.SimpleNamespace(friendly=lambda e, spec: (f"{spec.name} unreachable at {spec.base_url}", f"Details: {e.raw}"))
    monkeypatch.setitem(sys.modules, "jarvoos.providers", providers)
    monkeypatch.setitem(sys.modules, "jarvoos.errors", errors)
    return types.SimpleNamespace(calls=calls, results=results, home=tmp_path)


OR_MODELS = {"data": [
    {"id": "deepseek/deepseek-v4-flash", "context_length": 128000, "pricing": {"prompt": "0.0000002", "completion": "0.0000008"}, "supported_parameters": ["tools"]},
    {"id": "other/model", "context_length": 32000, "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["tools"]}]}


def or_routes(keys=(GOOD,)):
    def auth(req):
        return (200, {"data": {"label": "ok"}}) if req.headers.get("authorization", "")[7:] in keys else (401, {"error": "no"})
    return {"/api/v1/auth/key": auth, "/api/v1/models": (200, OR_MODELS)}


def mode_of(p):
    return stat.S_IMODE(p.stat().st_mode)


def test_happy_path_strata_plus_openrouter(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA}, other=or_routes())
    io = ScriptIO(("choose", 0),            # Strata [found]
                  ("ask", ""),              # URL default
                  ("confirm", True),        # use qwen3-30b
                  ("confirm", False),       # thinking off
                  ("confirm", True),        # add OpenRouter
                  ("secret", "\x1b[200~" + GOOD + "\x1b[201~\n"),
                  ("ask", ""),              # default OpenRouter model
                  ("choose", 0),            # mode approve
                  ("confirm", True))        # save
    assert setup.run(first_run=True, io=io) is True
    assert not io.script
    cfg = config.load()
    assert cfg.chain == ["local", "openrouter"] and cfg.mode == "approve"
    p, f = cfg.primary, cfg.fallback
    assert (p.kind, p.model, p.base_url, p.context_tokens, p.thinking, p.tools) == ("strata", "qwen3-30b", "http://127.0.0.1:8090/v1", 131072, "off", True)
    assert (f.kind, f.model, f.api_key_env, f.context_tokens) == ("openrouter", "deepseek/deepseek-v4-flash", "OPENROUTER_API_KEY", 128000)
    assert mode_of(config.CONFIG_PATH) == 0o600 and mode_of(config.SECRETS_PATH) == 0o600
    assert config.get_secret("OPENROUTER_API_KEY") == GOOD
    assert [c[0] for c in env.calls] == ["local", "openrouter"] and env.calls[1][1] == GOOD
    assert "OK  310ms to first token, tools OK" in io.text and "sk-..." in io.text and GOOD not in io.text
    assert "Get a key at https://openrouter.ai/keys" in io.text


def test_nothing_written_before_save_confirm(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    io = ScriptIO(("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False), ("confirm", False), ("choose", 0), ("confirm", False))
    assert setup.run(first_run=True, io=io) is False
    assert list(env.home.iterdir()) == []


def test_openrouter_only_bad_then_good_key(env, monkeypatch):
    fake_servers(monkeypatch, {}, other=or_routes())
    io = ScriptIO(("choose", 4),            # OpenRouter in the preset menu
                  ("secret", "bad-key"), ("secret", " " + GOOD + " "),
                  ("ask", "flash"),         # search -> single match
                  ("confirm", True),        # Use deepseek/deepseek-v4-flash?
                  ("choose", 1),            # mode auto
                  ("confirm", True))
    assert setup.run(io=io) is True
    assert "No local server found" in io.text and "Key rejected" in io.text
    cfg = config.load()
    assert cfg.chain == ["openrouter"] and cfg.mode == "auto" and cfg.primary.model == "deepseek/deepseek-v4-flash"
    assert config.get_secret("OPENROUTER_API_KEY") == GOOD


def test_openrouter_three_bad_keys_skips(env, monkeypatch):
    fake_servers(monkeypatch, {}, other=or_routes())
    io = ScriptIO(("choose", 4), ("secret", "a"), ("secret", "b"), ("secret", "c"), ("confirm", False))  # then "try again?" no
    assert setup.run(io=io) is False
    assert "Skipping OpenRouter" in io.text and list(env.home.iterdir()) == []


def test_env_key_is_used_and_not_copied_to_secrets(env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", GOOD)
    fake_servers(monkeypatch, {}, other=or_routes())
    io = ScriptIO(("choose", 4), ("ask", ""), ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True
    assert not config.SECRETS_PATH.exists()


def test_probe_failure_retry_then_skip_saves_nothing(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    env.results["local"] = Probe(ok=False, chat_ok=False, tools_ok=False, error=Err())
    io = ScriptIO(("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False),
                  ("confirm", False),       # no fallback
                  ("choose", 0),            # retry
                  ("choose", 2),            # skip
                  ("confirm", False))       # don't try again
    assert setup.run(first_run=True, io=io) is False
    assert len(env.calls) == 2
    assert "local unreachable at http://127.0.0.1:8090/v1" in io.text and "URL tried: http://127.0.0.1:8090/v1/chat/completions" in io.text
    assert "Details: connect failed" in io.text and "Setup cancelled" not in io.text
    assert list(env.home.iterdir()) == []


def test_skip_local_keep_openrouter(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA}, other=or_routes())
    env.results["local"] = Probe(chat_ok=False, tools_ok=False, error=Err())
    io = ScriptIO(("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False), ("confirm", True),
                  ("secret", GOOD), ("ask", ""), ("choose", 2), ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True
    cfg = config.load()
    assert cfg.chain == ["openrouter"] and cfg.mode == "approve"


def test_retry_succeeds_second_time(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    env.results["local"] = [Probe(chat_ok=False, error=Err()), Probe()]
    io = ScriptIO(("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False), ("confirm", False),
                  ("choose", 0), ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True and config.load().chain == ["local"]


def test_ctrl_c_prints_cancelled_and_saves_nothing(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    for after in range(0, 7):
        script = [("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False), ("confirm", False), ("choose", 0)][:after]
        io = ScriptIO(*script, ("interrupt", None))
        assert setup.run(first_run=True, io=io) is False
        assert "Setup cancelled, nothing saved" in io.text
        assert list(env.home.iterdir()) == []


def test_tools_fail_chat_only_llamacpp_hint(env, monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8080": LLAMA})
    env.results["local"] = Probe(tools_ok=False)
    io = ScriptIO(("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False), ("confirm", False),
                  ("confirm", True),        # continue chat-only
                  ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True
    assert "--jinja" in io.text
    p = config.load().primary
    assert p.tools is False and p.kind == "llamacpp" and p.context_tokens == 8192


def test_unauthorized_server_prompts_for_per_endpoint_key(env, monkeypatch):
    def models(req):
        return (200, {"data": [{"id": "m1"}]}) if req.headers.get("authorization") == "Bearer secret1" else (401, {"error": "no"})
    fake_servers(monkeypatch, {"127.0.0.1:8090": {"/health": (200, {"service": "strata", "max_context": 8192, "api_key": True}), "/v1/models": models}})
    io = ScriptIO(("choose", 0), ("ask", ""), ("secret", "wrong"), ("secret", " secret1\n"),
                  ("confirm", True), ("confirm", False), ("confirm", False), ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True
    p = config.load().primary
    assert p.api_key_env == "JARVOOS_KEY_127_0_0_1_8090"
    assert config.get_secret("JARVOOS_KEY_127_0_0_1_8090") == "secret1"
    assert env.calls[0][1] == "secret1"
    assert "That key was rejected" in io.text


def test_unreachable_url_distinguishes_refused_and_can_skip(env, monkeypatch):
    fake_servers(monkeypatch, {})
    io = ScriptIO(("choose", 0), ("ask", ""), ("choose", 2), ("confirm", False))
    assert setup.run(io=io) is False
    assert "Connection refused: is the server running?" in io.text


def test_loading_server_is_waited_for(env, monkeypatch):
    n = {"i": 0}
    def models(req):
        n["i"] += 1
        return (503, {"error": {"message": "Loading model"}}) if n["i"] < 3 else (200, {"data": [{"id": "m"}]})
    fake_servers(monkeypatch, {"127.0.0.1:8080": {"/v1/models": models}})
    monkeypatch.setattr(setup, "LOAD_WAIT_S", 10)
    io = ScriptIO(("choose", 1), ("ask", "http://127.0.0.1:8080/v1"), ("confirm", True), ("confirm", False), ("confirm", False), ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True
    assert "still loading" in io.text


def test_many_models_searchable_pick(env, monkeypatch):
    models = {"data": [{"id": f"model-{i}"} for i in range(20)] + [{"id": "special-one"}]}
    fake_servers(monkeypatch, {"127.0.0.1:8081": {"/v1/models": (200, models)}})
    io = ScriptIO(("choose", 0), ("ask", ""), ("ask", "spec"), ("confirm", True), ("confirm", False), ("choose", 0), ("confirm", True))
    assert setup.run(io=io) is True
    assert config.load().primary.model == "special-one" and config.load().primary.kind == "openai"


def test_section_mode_and_search(env, monkeypatch):
    config.save(config.Config({"local": config.ProviderSpec("local", "strata", "http://127.0.0.1:8090/v1", "m")}, ["local"]))
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    assert setup.run(section="mode", io=ScriptIO(("choose", 2), ("confirm", True))) is True
    assert config.load().mode == "chat" and (env.home / "config.toml.bak").exists()
    assert setup.run(section="search", io=ScriptIO(("choose", 1), ("ask", "http://sx:8888"), ("confirm", True))) is True
    s = config.load().search
    assert s["provider"] == "searxng" and s["searxng_url"] == "http://sx:8888"
    assert setup.run(section="search", io=ScriptIO(("choose", 2), ("secret", "exa-key"), ("confirm", True))) is True
    assert config.get_secret("EXA_API_KEY") == "exa-key" and config.load().search["provider"] == "exa"


def test_section_without_config_and_bad_section(env):
    assert setup.run(section="mode", io=ScriptIO()) is False
    with pytest.raises(ValueError):
        setup.run(section="bogus", io=ScriptIO())


def test_section_model_replaces_primary_keeps_fallback(env, monkeypatch):
    config.save(config.Config({"local": config.ProviderSpec("local", "strata", "http://127.0.0.1:8090/v1", "old"),
                               "openrouter": config.ProviderSpec("openrouter", "openrouter", detect.OPENROUTER_URL, "x/y", "OPENROUTER_API_KEY")},
                              ["local", "openrouter"]))
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    io = ScriptIO(("choose", 0), ("ask", ""), ("confirm", True), ("confirm", False), ("confirm", True))
    assert setup.run(section="model", io=io) is True
    cfg = config.load()
    assert cfg.chain == ["local", "openrouter"] and cfg.primary.model == "qwen3-30b" and cfg.fallback.model == "x/y"


def test_noninteractive_help_mentions_the_essentials():
    t = setup.noninteractive_help()
    assert "OPENROUTER_API_KEY" in t and "jarvoos config set" in t and "jarvoos setup" in t
