import os
import stat

import pytest

from jarvoos import config
from jarvoos.config import Config, ProviderSpec


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path))
    monkeypatch.delenv("JARVOOS_STATE", raising=False)
    for k in ("OPENROUTER_API_KEY", "EXA_API_KEY", "JARVOOS_KEY_X"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def sample() -> Config:
    return Config(
        {"local": ProviderSpec("local", "strata", "http://127.0.0.1:8090/v1", "qwen3", context_tokens=131072, thinking="off"),
         "openrouter": ProviderSpec("openrouter", "openrouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash",
                                    api_key_env="OPENROUTER_API_KEY", tools=False, first_token_timeout_s=30)},
        ["local", "openrouter"], mode="auto", max_steps=40, compact_at=0.6, verbose=True)


def mode_of(p):
    return stat.S_IMODE(p.stat().st_mode)


def test_paths_follow_jarvoos_home_and_state(home, monkeypatch):
    assert config.CONFIG_PATH == home / "config.toml" and config.SECRETS_PATH == home / "secrets.env"
    assert config.STATE_DIR == home / "state"
    monkeypatch.setenv("JARVOOS_STATE", str(home / "elsewhere"))
    assert config.STATE_DIR == home / "elsewhere"


def test_round_trip_and_perms(home):
    config.save(sample(), {"OPENROUTER_API_KEY": "sk-or-123"})
    cfg = config.load()
    assert cfg == sample()
    assert cfg.primary.name == "local" and cfg.fallback.name == "openrouter"
    assert mode_of(config.CONFIG_PATH) == 0o600 and mode_of(config.SECRETS_PATH) == 0o600
    assert config.get_secret("OPENROUTER_API_KEY") == "sk-or-123"
    assert not list(home.glob(".*.tmp"))


def test_bak_written_before_overwrite(home):
    config.save(sample())
    assert not (home / "config.toml.bak").exists()
    first = config.CONFIG_PATH.read_text()
    c = sample()
    c.mode = "chat"
    config.save(c)
    assert (home / "config.toml.bak").read_text() == first and mode_of(home / "config.toml.bak") == 0o600
    assert config.load().mode == "chat"


def test_secrets_merge_and_env_wins(home, monkeypatch):
    config.save(sample(), {"A_KEY": "1"})
    config.save(sample(), {"B_KEY": "2"})
    assert config.get_secret("A_KEY") == "1" and config.get_secret("B_KEY") == "2"
    monkeypatch.setenv("A_KEY", "from-env")
    assert config.get_secret("A_KEY") == "from-env"
    assert config.get_secret("") is None and config.get_secret("NOPE") is None
    with pytest.raises(ValueError):
        config.save(sample(), {"BAD\nKEY": "x"})
    with pytest.raises(ValueError):
        config.save(sample(), {"OK": "a\nb=c"})


def test_toml_escaping_round_trip(home):
    nasty = 'he said "hi" \\ back\\slash\nnewline\ttab \x01 ünï'
    c = sample()
    c.providers["local"].model = nasty
    c.providers["we ird.name"] = ProviderSpec("we ird.name", "openai", "http://x/v1", "m")
    c.chain.append("we ird.name")
    c.search = {"provider": "searxng", "searxng_url": nasty}
    config.save(c)
    back = config.load()
    assert back.providers["local"].model == nasty and back.search["searxng_url"] == nasty
    assert back.providers["we ird.name"].base_url == "http://x/v1"


def test_not_configured(home):
    with pytest.raises(config.NotConfigured):
        config.load()
    assert not config.is_configured()
    config.CONFIG_PATH.write_text("this is [not toml")
    with pytest.raises(config.NotConfigured):
        config.load()
    config.CONFIG_PATH.write_text('chain = ["ghost"]\n')
    with pytest.raises(config.NotConfigured, match="ghost"):
        config.load()
    config.CONFIG_PATH.write_text('mode = "yolo"\nchain = ["a"]\n[providers.a]\nbase_url = "http://x/v1"\nmodel = "m"\n')
    with pytest.raises(config.NotConfigured, match="mode"):
        config.load()
    assert config.problem()


def test_migration_from_primary_fallback(home):
    config.CONFIG_PATH.write_text('''
[primary]
name = "local"
base_url = "http://127.0.0.1:8090/v1"
model = "qwen3"
api_key_env = ""
thinking = "off"
[fallback]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-v4-flash"
api_key_env = "OPENROUTER_API_KEY"
[agent]
max_steps = 12
[search]
provider = "exa"
searxng_url = ""
''')
    cfg = config.load()
    assert cfg.chain == ["local", "openrouter"]
    assert cfg.primary.kind == "strata" and cfg.primary.thinking == "off"
    assert cfg.fallback.kind == "openrouter" and cfg.fallback.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.max_steps == 12 and cfg.search["provider"] == "ddg"  # exa without a key -> keyless default
    config.save(cfg)
    text = config.CONFIG_PATH.read_text()
    assert "[primary]" not in text and "[providers.local]" in text
    assert config.load() == cfg


def test_migration_skips_unconfigured_fallback(home):
    config.CONFIG_PATH.write_text('[primary]\nname="local"\nbase_url="http://127.0.0.1:8081/v1"\nmodel="m"\n[fallback]\nname="openrouter"\nbase_url="https://openrouter.ai/api/v1"\nmodel=""\n')
    cfg = config.load()
    assert cfg.chain == ["local"] and cfg.fallback is None and cfg.primary.kind == "llamacpp"


def test_default_search_is_keyless(home):
    c = sample()
    c.search = {}
    config.save(c)
    assert config.load().search["provider"] == "ddg"
    assert Config({}, []).search == {"provider": "ddg"}


@pytest.mark.parametrize("raw,expected", [
    ("sk-or-v1-abc123", "sk-or-v1-abc123"),
    ("\x1b[200~sk-or-v1-abc123\x1b[201~", "sk-or-v1-abc123"),
    ("[200~sk-or-v1-abc123[201~", "sk-or-v1-abc123"),
    ("  sk-or-v1-\n abc123 \r\n", "sk-or-v1-abc123"),
    ("sk-or-v1-abc​123 ", "sk-or-v1-abc123"),
    ("Bearer sk-or-v1-abc123", "sk-or-v1-abc123"),
    ("OPENROUTER_API_KEY=sk-or-v1-abc123", "sk-or-v1-abc123"),
    ("export OPENROUTER_API_KEY='sk-or-v1-abc123'", "sk-or-v1-abc123"),
    ('"sk-or-v1-abc123"', "sk-or-v1-abc123"),
    ("\x1b[31msk-or\x1b[0m-v1-abc123", "sk-or-v1-abc123"),
    ("", ""),
])
def test_clean_secret(raw, expected):
    assert config.clean_secret(raw) == expected


def test_env_name_for_and_mask():
    assert config.env_name_for("http://127.0.0.1:8081/v1") == "JARVOOS_KEY_127_0_0_1_8081"
    assert config.env_name_for("https://My-Box.local/v1") == "JARVOOS_KEY_MY_BOX_LOCAL"
    assert config.env_name_for("localhost:8090") == "JARVOOS_KEY_LOCALHOST_8090"
    assert config.mask("sk-or-v1-abcdef") == "sk-..." and config.mask("") == "" and config.mask("abc") == "***"


def test_set_get_value(home):
    config.save(sample())
    config.set_value("mode", "chat")
    config.set_value("max_steps", "7")
    config.set_value("verbose", "false")
    config.set_value("search.provider", "searxng")
    config.set_value("providers.local.context_tokens", "65536")
    config.set_value("providers.local.tools", "no")
    config.set_value("chain", "openrouter, local")
    assert config.get_value("mode") == "chat" and config.get_value("max_steps") == "7"
    assert config.get_value("chain") == "openrouter, local" and config.get_value("verbose") == "false"
    cfg = config.load()
    assert cfg.search["provider"] == "searxng" and cfg.providers["local"].context_tokens == 65536 and not cfg.providers["local"].tools
    assert cfg.chain == ["openrouter", "local"]
    with pytest.raises(ValueError):
        config.set_value("mode", "yolo")
    with pytest.raises(ValueError):
        config.set_value("nonsense", "1")
    with pytest.raises(ValueError):
        config.set_value("max_steps", "many")
    with pytest.raises(KeyError):
        config.get_value("providers.nope.model")


def test_set_value_builds_config_from_nothing(home):
    for k, v in [("providers.local.kind", "strata"), ("providers.local.base_url", "http://127.0.0.1:8090/v1"),
                 ("providers.local.model", "m"), ("chain", "local")]:
        config.set_value(k, v)
    assert config.load().primary.model == "m"
    assert mode_of(config.CONFIG_PATH) == 0o600
