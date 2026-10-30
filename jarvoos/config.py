"""Config and secrets. Settings live in config.toml, keys in secrets.env (0600), both under ~/.config/jarvoos.

JARVOOS_HOME moves everything (tests, containers); JARVOOS_STATE moves only the state dir (sessions, history).
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

MODES = ("approve", "auto", "chat")
KINDS = ("strata", "llamacpp", "ollama", "lmstudio", "openrouter", "openai")
DEFAULT_SEARCH = {"provider": "ddg"}


class NotConfigured(Exception):
    pass


def _home() -> Path:
    return Path(os.environ.get("JARVOOS_HOME") or Path.home() / ".config" / "jarvoos")


def _state() -> Path:
    if os.environ.get("JARVOOS_STATE"):
        return Path(os.environ["JARVOOS_STATE"])
    if os.environ.get("JARVOOS_HOME"):
        return _home() / "state"
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "jarvoos"


_PATHS = {
    "HOME_DIR": _home,
    "CONFIG_PATH": lambda: _home() / "config.toml",
    "SECRETS_PATH": lambda: _home() / "secrets.env",
    "SKILLS_DIR": lambda: _home() / "skills",
    "STATE_DIR": _state,
    "EVENTS_LOG": lambda: _state() / "events.log",
}


def __getattr__(name: str):
    """Paths are resolved on access so JARVOOS_HOME can change after import (tests)."""
    if name in _PATHS:
        return _PATHS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class ProviderSpec:
    name: str
    kind: str
    base_url: str
    model: str
    api_key_env: str = ""
    context_tokens: int = 0
    tools: bool = True
    thinking: str = "default"
    first_token_timeout_s: float = 15
    idle_timeout_s: float = 60


@dataclass
class Config:
    providers: dict[str, ProviderSpec]
    chain: list[str]
    mode: str = "approve"
    max_steps: int = 25
    search: dict = field(default_factory=lambda: dict(DEFAULT_SEARCH))
    compact_at: float = 0.75
    verbose: bool = False

    @property
    def primary(self) -> ProviderSpec | None:
        return self.providers[self.chain[0]] if self.chain else None

    @property
    def fallback(self) -> ProviderSpec | None:
        return self.providers[self.chain[1]] if len(self.chain) > 1 else None


_PROVIDER_TYPES = {"name": str, "kind": str, "base_url": str, "model": str, "api_key_env": str,
                   "context_tokens": int, "tools": bool, "thinking": str,
                   "first_token_timeout_s": float, "idle_timeout_s": float}
_TOP_TYPES = {"mode": str, "chain": list, "max_steps": int, "compact_at": float, "verbose": bool}


def guess_kind(url: str) -> str:
    p = urlparse(url if "//" in url else "//" + url)
    if (p.hostname or "").endswith("openrouter.ai"):
        return "openrouter"
    return {8090: "strata", 8080: "llamacpp", 8081: "llamacpp", 11434: "ollama", 1234: "lmstudio"}.get(p.port or 0, "openai")


def _coerce(value, typ: type, where: str):
    try:
        if typ is bool:
            if isinstance(value, bool):
                return value
            if str(value).strip().lower() in ("true", "1", "yes", "on"):
                return True
            if str(value).strip().lower() in ("false", "0", "no", "off"):
                return False
            raise ValueError
        if typ is list:
            return value if isinstance(value, list) else [x.strip() for x in str(value).split(",") if x.strip()]
        return typ(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid value for {where}: {value!r} (expected {typ.__name__})") from None


# ---- TOML writer (tomllib only reads) ----

def _tstr(s: str) -> str:
    out = []
    for ch in str(s):
        if ch in '\\"':
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _tkey(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _tstr(k)


def _tval(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_tval(x) for x in v) + "]"
    return _tstr(v)


def dumps(data: dict, path: tuple[str, ...] = ()) -> str:
    scalars = [f"{_tkey(k)} = {_tval(v)}" for k, v in data.items() if not isinstance(v, dict)]
    tables = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    lines = ([f"[{'.'.join(_tkey(p) for p in path)}]"] if path and scalars else []) + scalars
    out = "\n".join(lines) + ("\n\n" if lines else "")
    return out + "".join(dumps(v, path + (k,)) for k, v in tables)


# ---- reading ----

def _migrate(raw: dict) -> dict:
    """Old layout: [primary] / [fallback] tables. Read-compatible; save() writes the new layout."""
    if "providers" in raw or not ("primary" in raw or "fallback" in raw):
        return raw
    providers: dict[str, dict] = {}
    chain: list[str] = []
    for sect in ("primary", "fallback"):
        d = raw.get(sect) or {}
        if not d.get("model"):
            continue
        name = d.get("name") or sect
        while name in providers:
            name += "2"
        providers[name] = {"kind": guess_kind(d.get("base_url", "")), "base_url": d.get("base_url", ""),
                           "model": d["model"], "api_key_env": d.get("api_key_env", ""),
                           "thinking": d.get("thinking", "default")}
        chain.append(name)
    out: dict = {"chain": chain, "providers": providers}
    if "max_steps" in (raw.get("agent") or {}):
        out["max_steps"] = raw["agent"]["max_steps"]
    s = dict(raw.get("search") or {})
    if s.get("provider") == "exa" and not get_secret("EXA_API_KEY"):
        s["provider"] = "searxng" if s.get("searxng_url") else "ddg"
    out["search"] = s
    return out


def _read_raw() -> dict:
    path = _PATHS["CONFIG_PATH"]()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return _migrate(tomllib.load(f))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as e:
        raise NotConfigured(f"{path} could not be read: {e}") from e


def _spec(name: str, d: dict) -> ProviderSpec:
    kw = {}
    for k, typ in _PROVIDER_TYPES.items():
        if k in d:
            try:
                kw[k] = _coerce(d[k], typ, f"providers.{name}.{k}")
            except ValueError as e:
                raise NotConfigured(str(e)) from e
    kw.pop("name", None)
    kw.setdefault("base_url", "")
    kw.setdefault("model", "")
    kw.setdefault("kind", guess_kind(kw["base_url"]))
    if not kw["base_url"] or not kw["model"]:
        raise NotConfigured(f"provider {name!r} needs base_url and model")
    return ProviderSpec(name=name, **kw)


def _build(raw: dict) -> Config:
    providers = {n: _spec(n, d) for n, d in (raw.get("providers") or {}).items() if isinstance(d, dict)}
    chain = [str(n) for n in (raw.get("chain") or list(providers))]
    if not chain:
        raise NotConfigured("no providers configured")
    missing = [n for n in chain if n not in providers]
    if missing:
        raise NotConfigured(f"chain names unknown provider(s): {', '.join(missing)}")
    kw = {}
    for k, typ in _TOP_TYPES.items():
        if k in raw and k != "chain":
            try:
                kw[k] = _coerce(raw[k], typ, k)
            except ValueError as e:
                raise NotConfigured(str(e)) from e
    if kw.get("mode", "approve") not in MODES:
        raise NotConfigured(f"mode must be one of {', '.join(MODES)}")
    return Config(providers, chain, search={**DEFAULT_SEARCH, **(raw.get("search") or {})}, **kw)


def load() -> Config:
    raw = _read_raw()
    if not raw:
        raise NotConfigured(f"{_PATHS['CONFIG_PATH']()} not found" if not _PATHS["CONFIG_PATH"]().exists() else "no providers configured")
    return _build(raw)


def is_configured() -> bool:
    try:
        load()
        return True
    except NotConfigured:
        return False


def problem() -> str:
    """Why load() fails, for doctor and first-run messages."""
    try:
        load()
    except NotConfigured as e:
        return str(e)
    return ""


# ---- writing ----

def _write_atomic(path: Path, data: str) -> None:
    """Temp file created 0600 (never world-readable, even briefly), then renamed over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _to_dict(cfg: Config) -> dict:
    d: dict = {"mode": cfg.mode, "chain": list(cfg.chain), "max_steps": cfg.max_steps,
               "compact_at": cfg.compact_at, "verbose": cfg.verbose}
    d["providers"] = {n: {k: v for k, v in asdict(p).items() if k != "name"} for n, p in cfg.providers.items()}
    d["search"] = dict(cfg.search)
    return d


def _write_config(data: dict) -> None:
    path = _PATHS["CONFIG_PATH"]()
    if path.exists():
        _write_atomic(path.with_name(path.name + ".bak"), path.read_text(encoding="utf-8", errors="replace"))
    _write_atomic(path, dumps(data))


def _read_secrets() -> dict[str, str]:
    out: dict[str, str] = {}
    path = _PATHS["SECRETS_PATH"]()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def save(cfg: Config, secrets: dict[str, str] | None = None) -> None:
    data = _to_dict(cfg)
    if secrets:
        for k, v in secrets.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or "\n" in v or "\r" in v:
                raise ValueError(f"invalid secret entry {k!r}")
        merged = {**_read_secrets(), **{k: v for k, v in secrets.items() if v}}
        _write_config(data)
        _write_atomic(_PATHS["SECRETS_PATH"](), "".join(f"{k}={v}\n" for k, v in merged.items()))
    else:
        _write_config(data)


def set_secrets(secrets: dict[str, str]) -> None:
    """Add or replace entries in secrets.env without touching the provider config."""
    for k, v in secrets.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not v or "\n" in v or "\r" in v:
            raise ValueError(f"invalid secret entry {k!r}")
    merged = {**_read_secrets(), **secrets}
    _write_atomic(_PATHS["SECRETS_PATH"](), "".join(f"{k}={v}\n" for k, v in merged.items()))


def get_secret(env_name: str) -> str | None:
    """Environment wins over secrets.env. Keys are never copied into os.environ."""
    if not env_name:
        return None
    return os.environ.get(env_name) or _read_secrets().get(env_name) or None


def clean_secret(raw: str) -> str:
    """Make a pasted key usable: bracketed-paste markers, ANSI codes, 'Bearer'/'KEY=' prefixes, whitespace, non-ASCII."""
    s = re.sub(r"\x1b?\[20[01]~", "", raw)
    s = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", s)
    s = re.sub(r"^\s*(?:export\s+)?(?:[A-Z][A-Z0-9_]*=|Bearer\s+)", "", s)
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[\s\x00-\x1f\x7f]+", "", s)
    return s.strip("\"'`")


def env_name_for(url: str) -> str:
    p = urlparse(url if "//" in url else "//" + url)
    host = re.sub(r"\W", "_", (p.hostname or "server")) + (f"_{p.port}" if p.port else "")
    return "JARVOOS_KEY_" + host.upper()


def mask(s: str) -> str:
    if not s:
        return ""
    return s[:3] + "..." if len(s) > 6 else "***"


# ---- `jarvoos config get|set` ----

def _walk(raw: dict, dotted: str, create: bool = False):
    parts = dotted.split(".")
    node = raw
    for p in parts[:-1]:
        if p not in node:
            if not create:
                raise KeyError(dotted)
            node[p] = {}
        node = node[p]
        if not isinstance(node, dict):
            raise KeyError(dotted)
    return node, parts[-1]


def get_value(dotted: str) -> str:
    node, key = _walk(_read_raw(), dotted)
    if key not in node:
        raise KeyError(dotted)
    v = node[key]
    if isinstance(v, bool):
        return "true" if v else "false"
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v)


def _typed(dotted: str, value: str):
    parts = dotted.split(".")
    if len(parts) == 1 and parts[0] in _TOP_TYPES:
        v = _coerce(value, _TOP_TYPES[parts[0]], dotted)
        if parts[0] == "mode" and v not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        return v
    if len(parts) == 3 and parts[0] == "providers" and parts[2] in _PROVIDER_TYPES and parts[2] != "name":
        if parts[2] == "kind" and value not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        return _coerce(value, _PROVIDER_TYPES[parts[2]], dotted)
    if len(parts) == 2 and parts[0] == "search":
        return _coerce(value, bool, dotted) if value.lower() in ("true", "false") else value
    raise ValueError(f"unknown setting {dotted!r}; try mode, chain, max_steps, compact_at, verbose, search.<key>, providers.<name>.<field>")


def set_value(dotted: str, value: str) -> None:
    typed = _typed(dotted, value)
    raw = _read_raw()
    node, key = _walk(raw, dotted, create=True)
    node[key] = typed
    _write_config(raw)
