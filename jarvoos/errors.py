"""Provider errors: one exception type with a kind, HTTP classification and the user-facing wording."""
from __future__ import annotations

import json
import re

OR_CREDITS_URL = "https://openrouter.ai/settings/credits"
OR_KEYS_URL = "https://openrouter.ai/keys"
FAILOVER_KINDS = frozenset({"network", "timeout", "server", "rate_limit", "auth", "billing", "not_found", "empty"})
CONTEXT_MARKERS = ("context length", "context_length", "context window", "maximum context", "exceeds the context",
                   "exceed_context", "too many tokens", "prompt is too long", "prompt too long", "n_ctx",
                   "input is too long", "token limit", "reduce the length", "maximum number of tokens")
RETRY_AFTER_MAX = 60.0


class Interrupted(KeyboardInterrupt):
    """The user cancelled the turn (Ctrl+C). Subclass of KeyboardInterrupt so stray handlers still see it."""


class ProviderError(Exception):
    def __init__(self, kind: str, message: str = "", *, status: int | None = None, provider: str = "",
                 url: str = "", retry_after: float | None = None, raw: str = ""):
        super().__init__(f"{kind}: {message or raw[:200]}")
        self.kind, self.status, self.provider, self.url = kind, status, provider, url
        self.retry_after, self.raw = retry_after, raw or message

    @property
    def fatal_for_failover(self) -> bool:
        return self.kind in ("context", "bad_request", "not_configured")


def should_failover(e: ProviderError) -> bool:
    return e.kind in FAILOVER_KINDS


def _clamp(v) -> float | None:
    try:
        return max(0.0, min(float(v), RETRY_AFTER_MAX))
    except (TypeError, ValueError):
        return None


def _message(body: str) -> tuple[str, dict]:
    """(human message, parsed error object) from a JSON error body; falls back to the raw text."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body.strip(), {}
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, str):
        return err, {}
    if isinstance(err, dict):
        return str(err.get("message") or err.get("msg") or body).strip(), err
    return body.strip(), {}


def _is_context(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in CONTEXT_MARKERS)


def classify_http(status: int, body: str, provider: str, url: str, headers=None) -> ProviderError:
    msg, err = _message(body or "")
    ra = None
    if headers is not None:
        ra = _clamp(headers.get("retry-after"))
    meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    if meta.get("retry_after_seconds") is not None:
        ra = _clamp(meta["retry_after_seconds"])
    if status in (401, 403):
        kind = "auth"
    elif status == 402:
        kind = "billing"
    elif status == 404:
        kind = "not_found"
    elif status == 408:
        kind = "timeout"
    elif status == 429:
        kind = "rate_limit"
    elif status >= 500:
        kind = "server"
    elif status == 413 or _is_context(f"{msg} {body}"):
        kind = "context"
    else:
        kind = "bad_request"
    return ProviderError(kind, msg, status=status, provider=provider, url=url, retry_after=ra, raw=msg or body)


def from_stream_error(err, provider: str, url: str) -> ProviderError:
    """An `error` object that arrived inside a 200 event stream."""
    body = json.dumps({"error": err}) if isinstance(err, dict) else json.dumps({"error": {"message": str(err)}})
    code = err.get("code") if isinstance(err, dict) else None
    status = code if isinstance(code, int) and 100 <= code <= 599 else None
    e = classify_http(status or 500, body, provider, url)
    if status is None and e.kind == "server" and _is_context(e.raw):
        e.kind = "context"
    e.status = status
    return e


_LOCAL_NAMES = {"strata": "Strata", "llamacpp": "llama-server", "ollama": "Ollama", "lmstudio": "LM Studio"}


def friendly(e: ProviderError, spec) -> tuple[str, str]:
    """(one sentence saying what happened and how to fix it, raw detail truncated to 300 chars)."""
    kind, name = getattr(spec, "kind", ""), getattr(spec, "name", "") or e.provider or "the provider"
    url = e.url or getattr(spec, "base_url", "")
    model, env = getattr(spec, "model", ""), getattr(spec, "api_key_env", "")
    cloud = kind == "openrouter"
    pretty = "OpenRouter" if cloud else name
    server = _LOCAL_NAMES.get(kind)
    k = e.kind
    if k == "network":
        if cloud:
            msg = f"Can't reach OpenRouter at {url}. Check your internet connection, then try: jarvoos doctor"
        else:
            ask = f"Is {server} running?" if server else "Is the server running?"
            msg = f"Local server at {url} isn't answering. {ask} Try: jarvoos doctor"
    elif k == "timeout":
        msg = (f"{pretty} stopped responding ({url}). A local model may still be loading; wait a moment and retry, "
               "or switch with /model.")
    elif k == "auth":
        if cloud:
            msg = f"OpenRouter rejected the API key. Get a new one at {OR_KEYS_URL}, then run: jarvoos setup fallback"
        else:
            msg = (f"{pretty} rejected the API key. Check ${env or 'the key variable'} "
                   f"or run: jarvoos setup model")
    elif k == "billing":
        msg = (f"OpenRouter says the account is out of credits. Top up at {OR_CREDITS_URL}" if cloud
               else f"{pretty} says payment is required. Check your account balance.")
    elif k == "rate_limit":
        wait = f" Try again in about {e.retry_after:.0f}s," if e.retry_after else " Wait a moment,"
        msg = f"{pretty} is rate limiting requests.{wait} or switch provider with /model."
    elif k == "server":
        msg = f"{pretty} had an internal error{f' (HTTP {e.status})' if e.status else ''}. Usually temporary: retry, or switch with /model."
    elif k == "context":
        msg = f"The conversation no longer fits {model or 'the model'}'s context window. Run /compress or start fresh with /new."
    elif k == "not_found":
        msg = (f"{pretty} has no model '{model}' (or the URL is wrong; it should end in /v1). "
               "Check it with: jarvoos setup model" if model else f"{pretty} was not found at {url}. Check the URL.")
    elif k == "empty":
        m = re.search(r"(\d+) chars of reasoning", e.raw)
        if m and int(m.group(1)) > 0:
            msg = (f"{model or pretty} only thought and never answered. Turn thinking off: "
                   f"jarvoos config set providers.{name}.thinking off")
        else:
            msg = f"{pretty} returned an empty reply. Retry, or switch with /model."
    elif k == "not_configured":
        msg = "No provider is set up yet. Run: jarvoos setup"
    else:
        msg = f"{pretty} rejected the request. Details below; try /new if the conversation is the problem."
    raw = " ".join((e.raw or str(e)).split())
    return msg, raw[:300] + ("..." if len(raw) > 300 else "")
