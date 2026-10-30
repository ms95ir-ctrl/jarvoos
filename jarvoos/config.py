"""Config and secrets. Settings live in config.toml, keys in secrets.env (0600), both under ~/.config/jarvoos."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

HOME_DIR = Path(os.environ.get("JARVOOS_HOME") or Path.home() / ".config" / "jarvoos")
CONFIG_FILE = HOME_DIR / "config.toml"
SECRETS_FILE = HOME_DIR / "secrets.env"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "jarvoos"
EVENTS_LOG = STATE_DIR / "events.log"
SKILLS_DIR = HOME_DIR / "skills"

DEFAULTS: dict = {
    "primary": {
        "name": "local",
        "base_url": "http://127.0.0.1:8090/v1",
        "model": "",
        "api_key_env": "",
        "thinking": "off",  # off | on | default
    },
    "fallback": {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "",
        "api_key_env": "OPENROUTER_API_KEY",
        "thinking": "default",
    },
    "failover": {
        "connect_timeout": 3.0,
        "first_token_base": 15.0,  # seconds before the first token, plus the per-1k term below
        "first_token_per_1k": 10.0,  # local prefill is slow on long prompts
        "fallback_timeout": 90.0,
        "cooldown": 300.0,
    },
    "search": {"provider": "exa", "searxng_url": ""},
    "agent": {
        "max_steps": 25,
        "tool_output_chars": 12000,
        "context_budget_tokens": 24000,
    },
    "tools": {"allow_private_fetch": False},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            return _merge(DEFAULTS, tomllib.load(f))
    return copy.deepcopy(DEFAULTS)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def save_config(cfg: dict) -> None:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, values in cfg.items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in values.items()]
        lines.append("")
    CONFIG_FILE.write_text("\n".join(lines))


def _read_secrets() -> dict[str, str]:
    out: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def get_secret(name: str) -> str:
    """Environment wins over the secrets file. Keys are never copied into os.environ."""
    if not name:
        return ""
    return os.environ.get(name) or _read_secrets().get(name, "")


def set_secret(name: str, value: str) -> None:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    data = _read_secrets()
    if value:
        data[name] = value
    else:
        data.pop(name, None)
    SECRETS_FILE.write_text("".join(f"{k}={v}\n" for k, v in data.items()))
    os.chmod(SECRETS_FILE, 0o600)


def mask(value: str) -> str:
    if not value:
        return "(not set)"
    return value[:4] + "…" + value[-3:] if len(value) > 10 else "set"
