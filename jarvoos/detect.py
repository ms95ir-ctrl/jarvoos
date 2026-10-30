"""Find and identify OpenAI-compatible servers (Strata, llama.cpp, Ollama, LM Studio) and query OpenRouter."""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from .config import guess_kind

DEFAULT_PORTS = (8090, 8080, 8081, 11434, 1234)
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_CTX = 4096
MODELS_TTL_S = 600
TRANSPORT: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport
_KIND_ORDER = {"strata": 0, "llamacpp": 1, "lmstudio": 2, "ollama": 3, "openai": 4}


@dataclass
class Found:
    kind: str
    base_url: str
    label: str
    models: list[str]
    n_ctx: int
    needs_key: bool
    loading: bool = False


@dataclass
class UrlProbe:
    ok: bool
    base_url: str
    status: str  # ok|refused|timeout|unauthorized|not_found|loading|bad_url
    kind: str = "openai"
    models: list[str] = field(default_factory=list)
    n_ctx: int = 0
    needs_key: bool = False
    hint: str = ""


@dataclass
class OrModel:
    id: str
    context: int
    price_in: float  # USD per 1M tokens
    price_out: float
    name: str = ""


def client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(timeout), transport=TRANSPORT)


def _get(c: httpx.Client, url: str, key: str | None = None):
    """-> (status, json or None, error) with error in '', 'refused', 'timeout', 'error'."""
    try:
        r = c.get(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    except httpx.ConnectError:
        return 0, None, "refused"
    except httpx.TimeoutException:
        return 0, None, "timeout"
    except httpx.HTTPError:
        return 0, None, "error"
    try:
        data = r.json()
    except ValueError:
        data = None
    return r.status_code, data, ""


def _root(url: str) -> str | None:
    """http(s)://host[:port][/path] without a trailing /v1; None if it is not a usable URL."""
    u = url.strip()
    if not u:
        return None
    if "://" not in u:
        u = "http://" + u
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname:
        return None
    path = re.sub(r"/v1$", "", p.path.rstrip("/"))
    return f"{p.scheme}://{p.netloc}{path}"


def _ids(data) -> list[str]:
    items = data.get("data") if isinstance(data, dict) else None
    return [m["id"] for m in items if isinstance(m, dict) and m.get("id")] if isinstance(items, list) else []


def _int(v) -> int:
    try:
        return max(int(v), 0)
    except (TypeError, ValueError):
        return 0


def _fingerprint(c: httpx.Client, root: str, key: str | None = None) -> Found | None:
    st, h, err = _get(c, root + "/health", key)
    if err:
        return None
    if st == 200 and isinstance(h, dict) and h.get("service") == "strata":
        _, m, _ = _get(c, root + "/v1/models", key)
        return Found("strata", root + "/v1", f"Strata at {root}", _ids(m), _int(h.get("max_context")), bool(h.get("api_key")))
    loading = st == 503 and "loading" in json.dumps(h).lower()
    if loading or (st == 200 and isinstance(h, dict) and h.get("status") in ("ok", "no slot available")):
        if loading:
            return Found("llamacpp", root + "/v1", f"llama-server at {root}", [], 0, False, loading=True)
        pst, props, _ = _get(c, root + "/props", key)
        mst, m, _ = _get(c, root + "/v1/models", key)
        n_ctx = 0
        if isinstance(props, dict):
            n_ctx = _int((props.get("default_generation_settings") or {}).get("n_ctx")) // max(_int(props.get("total_slots")), 1)
        if not n_ctx and _ids(m):
            n_ctx = _int(((m["data"][0] or {}).get("meta") or {}).get("n_ctx"))
        return Found("llamacpp", root + "/v1", f"llama-server at {root}", _ids(m), n_ctx, 401 in (pst, mst))
    st, tags, _ = _get(c, root + "/api/tags", key)
    if st == 200 and isinstance(tags, dict) and isinstance(tags.get("models"), list):
        names = [t.get("name") or t.get("model") for t in tags["models"] if isinstance(t, dict)]
        return Found("ollama", root + "/v1", f"Ollama at {root} (default context is only {OLLAMA_DEFAULT_CTX}; set OLLAMA_CONTEXT_LENGTH)",
                     [n for n in names if n], OLLAMA_DEFAULT_CTX, False)
    st, lm, _ = _get(c, root + "/api/v0/models", key)
    if st == 200 and isinstance(lm, dict) and isinstance(lm.get("data"), list):
        chat = [m for m in lm["data"] if isinstance(m, dict) and m.get("type", "llm") in ("llm", "vlm")]
        loaded = [m for m in chat if m.get("state") == "loaded"]
        n_ctx = max([_int(m.get("loaded_context_length")) for m in loaded] or [0]) or max([_int(m.get("max_context_length")) for m in chat[:1]] or [0])
        return Found("lmstudio", root + "/v1", f"LM Studio at {root}", [m["id"] for m in loaded + [m for m in chat if m not in loaded] if m.get("id")], n_ctx, False)
    st, m, _ = _get(c, root + "/v1/models", key)
    if st in (401, 403):
        return Found("openai", root + "/v1", f"OpenAI-compatible server at {root}", [], 0, True)
    if st == 200 and isinstance(m, dict) and isinstance(m.get("data"), list):
        meta = (m["data"][0].get("meta") or {}) if m["data"] and isinstance(m["data"][0], dict) else {}
        return Found("openai", root + "/v1", f"OpenAI-compatible server at {root}", _ids(m), _int(meta.get("n_ctx")), False)
    return None


def _candidates(extra: list[str]) -> list[str]:
    hosts = list(extra) + [h for h in re.split(r"[,\s;]+", os.environ.get("JARVOOS_DETECT_HOSTS", "")) if h]
    roots = [f"http://127.0.0.1:{p}" for p in DEFAULT_PORTS]
    for h in hosts:
        root = _root(h)
        if not root:
            continue
        p = urlparse(root)
        roots += [f"http://{p.hostname}:{port}" for port in DEFAULT_PORTS] if p.scheme == "http" and not p.port and not p.path else [root]
    return list(dict.fromkeys(roots))


def scan(extra_urls=(), timeout: float = 0.7) -> list[Found]:
    """Fingerprint local servers in parallel. Never raises; unreachable hosts are simply absent."""
    def one(root: str):
        try:
            with client(timeout) as c:
                return _fingerprint(c, root)
        except Exception:
            return None
    try:
        roots = _candidates(list(extra_urls))
        with ThreadPoolExecutor(max_workers=min(len(roots), 16) or 1) as pool:
            found = [f for f in pool.map(one, roots) if f]
    except Exception:
        return []
    seen: dict[str, Found] = {}
    for f in sorted(found, key=lambda f: _KIND_ORDER.get(f.kind, 9)):
        seen.setdefault(f.base_url, f)
    return list(seen.values())


_HINTS = {
    "refused": "Connection refused: is the server running? Start it, or check the host and port.",
    "timeout": "The server did not answer in time. Check the address and that nothing is blocking the port.",
    "unauthorized": "The server answered 401: it needs an API key.",
    "not_found": "The server is reachable but has no /models endpoint at that URL (it usually ends in /v1).",
    "loading": "The server is up but still loading the model.",
    "bad_url": "That is not a usable URL. Use a full address like http://127.0.0.1:8090/v1",
}


def probe_url(url: str, key: str | None = None, timeout: float = 3.0) -> UrlProbe:
    """Check that `url` answers /models, retrying with /v1 added or removed. Reports why it failed."""
    root = _root(url)
    if not root:
        return UrlProbe(False, url, "bad_url", hint=_HINTS["bad_url"])
    given = url.strip() if "://" in url else "http://" + url.strip()
    given = given.rstrip("/")
    bases = [given, root] if given.endswith("/v1") else [given, given + "/v1"]
    status, base, models = "not_found", bases[0], []
    with client(timeout) as c:
        for b in bases:
            st, data, err = _get(c, b + "/models", key)
            if err in ("refused", "timeout", "error"):
                return UrlProbe(False, bases[0], "refused" if err == "refused" else "timeout", hint=_HINTS["refused" if err == "refused" else "timeout"])
            if st == 200 and isinstance(data, dict) and isinstance(data.get("data"), list):
                status, base, models = "ok", b, _ids(data)
                break
            if st in (401, 403):
                status, base = "unauthorized", b
                break
            if st == 503:
                status, base = "loading", b
                break
        fp = _fingerprint(c, _root(base) or root, key) if status != "not_found" else None
    kind = fp.kind if fp else guess_kind(base)
    if status == "ok" and fp and fp.loading:
        status = "loading"
    hint = _HINTS.get(status, "")
    if status == "not_found":
        hint = f"{_HINTS['not_found']} Tried {bases[0]}/models and {bases[1]}/models."
    return UrlProbe(status == "ok", base, status, kind, models or (fp.models if fp else []),
                    fp.n_ctx if fp else 0, bool(fp and fp.needs_key) or status == "unauthorized", hint)


def openrouter_check_key(key: str) -> tuple[bool, str]:
    try:
        with client(8) as c:
            st, data, err = _get(c, OPENROUTER_URL + "/auth/key", key)
    except Exception as e:
        return False, f"unreachable: {e}"
    if err:
        return False, "unreachable: could not reach openrouter.ai"
    if st == 200:
        label = ((data or {}).get("data") or {}).get("label", "") if isinstance(data, dict) else ""
        return True, label or "ok"
    if st in (401, 403):
        return False, "rejected: OpenRouter says this key is not valid"
    return False, f"unexpected HTTP {st} from OpenRouter"


_cache: dict = {"t": 0.0, "models": None}


def _per_m(v) -> float:
    try:
        return max(float(v), 0.0) * 1e6
    except (TypeError, ValueError):
        return 0.0


def openrouter_models(tools_only: bool = True) -> list[OrModel]:
    """Public model list with context and $/M, cached; on network errors the stale list is served, else []."""
    now = time.monotonic()
    if _cache["models"] is None or now - _cache["t"] > MODELS_TTL_S:
        try:
            with client(10) as c:
                st, data, err = _get(c, OPENROUTER_URL + "/models")
            if st != 200 or not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError(f"HTTP {st} {err}")
            _cache["models"] = [(OrModel(m["id"], _int(m.get("context_length")), _per_m((m.get("pricing") or {}).get("prompt")),
                                         _per_m((m.get("pricing") or {}).get("completion")), m.get("name", "")),
                                 "tools" in (m.get("supported_parameters") or []))
                                for m in data["data"] if isinstance(m, dict) and m.get("id")]
            _cache["t"] = now
        except Exception:
            if _cache["models"] is None:
                return []
    return sorted((m for m, tools in _cache["models"] if tools or not tools_only), key=lambda m: m.id)
