"""OpenAI-compatible streaming providers and the failover router.

Strata, llama.cpp, Ollama, LM Studio and OpenRouter all speak the chat-completions dialect, so a conversation
moves between them without translation. The Router walks cfg.chain: on a down/limited/unauthorised provider it
switches at once (visibly, via RouterEvent); retries with backoff only happen when nothing else is left to try.
"""
from __future__ import annotations

import json
import queue
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import httpx

from .config import Config, ProviderSpec, get_secret
from .errors import Interrupted, ProviderError, classify_http, friendly, from_stream_error, should_failover

DEFAULT_CONTEXT = 32000
COOLDOWN_S = 300.0
RETRIES, BACKOFF_S, BACKOFF_CAP_S, JITTER = 3, 1.0, 30.0, 0.2
RETRY_AFTER_MAX = 60.0
RETRY_KINDS = ("network", "timeout", "server", "rate_limit")
APP_URL, APP_TITLE = "https://github.com/jarvoos/jarvoos", "Jarvoos"
MANDATORY_REASONING = re.compile(r"(/o\d|^o\d|deepseek-r1|gemini-2\.5-pro|-thinking|qwq|gpt-5(?!-chat))", re.I)
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
DONE = object()


def noop(*a, **k) -> None:
    return None


@dataclass
class ToolCall:
    id: str
    name: str
    args: str = ""  # JSON text exactly as streamed


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    finish: str = ""
    provider: str = ""
    reasoning_chars: int = 0


@dataclass
class RouterEvent:
    kind: str  # 'failover' | 'recovered' | 'restart' | 'retry'
    from_: str = ""
    to: str = ""
    error: ProviderError | None = None


@dataclass
class ProbeResult:
    ok: bool
    chat_ok: bool = False
    tools_ok: bool = False
    ttft_ms: int = 0
    error: ProviderError | None = None
    hint: str = ""


def estimate_tokens(messages: list[dict]) -> int:
    return int(len(json.dumps(messages, ensure_ascii=False)) / 3.5)


def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before retry number `attempt` (0-based): 1s doubling, 30s cap, +-20% jitter."""
    if retry_after:
        return min(float(retry_after), RETRY_AFTER_MAX)
    return min(BACKOFF_S * 2 ** attempt, BACKOFF_CAP_S) * (1 + random.uniform(-JITTER, JITTER))


def thinking_params(spec: ProviderSpec, model: str | None = None, thinking: str | None = None) -> dict:
    mode = thinking or spec.thinking
    if mode not in ("on", "off"):
        return {}
    on = mode == "on"
    if spec.kind in ("strata", "llamacpp"):
        return {"chat_template_kwargs": {"enable_thinking": on}}
    if spec.kind == "openrouter" and (on or not MANDATORY_REASONING.search(model or spec.model)):
        return {"reasoning": {"enabled": on}}
    return {}


def build_body(spec: ProviderSpec, model: str, messages: list[dict], tools: list[dict], *,
               max_tokens: int | None = None, thinking: str | None = None) -> dict:
    body: dict = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    if tools and spec.tools:
        body["tools"] = tools
    if max_tokens:
        body["max_tokens"] = max_tokens
    body.update(thinking_params(spec, model, thinking))
    return body


def request_headers(spec: ProviderSpec, key: str | None) -> dict:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if spec.kind == "openrouter":
        h["HTTP-Referer"], h["X-Title"] = APP_URL, APP_TITLE
    return h


def _exc_to_error(e: Exception, spec: ProviderSpec, url: str) -> ProviderError:
    detail = str(e) or type(e).__name__
    if isinstance(e, httpx.ConnectTimeout):
        kind = "timeout"
    elif isinstance(e, httpx.ConnectError):
        kind = "network"
    elif isinstance(e, httpx.TimeoutException):
        kind = "timeout"
    elif isinstance(e, (httpx.InvalidURL, httpx.UnsupportedProtocol)):
        kind = "bad_request"
    else:
        kind = "network"
    return ProviderError(kind, detail, provider=spec.name, url=url, raw=detail)


def _has_token(chunk: dict) -> bool:
    for c in chunk.get("choices") or []:
        d = c.get("delta") or {}
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
            return True
    return False


def _close_later(client: httpx.Client) -> None:
    threading.Thread(target=lambda: _quiet(client.close), daemon=True).start()


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:
        pass


def _sse(spec: ProviderSpec, key: str | None, body: dict, *, cancel: threading.Event | None, first_s: float,
         idle_s: float, transport=None) -> Iterator:
    """Yield parsed stream chunks, then DONE. The socket is read on a helper thread so the caller can enforce a
    first-token timeout, a per-chunk idle timeout and cancellation (within ~50ms) without waiting on a blocked read."""
    url = spec.base_url.rstrip("/") + "/chat/completions"
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    client = httpx.Client(transport=transport, timeout=httpx.Timeout(max(first_s, idle_s) + 2, connect=min(first_s, 10)))

    def work() -> None:
        try:
            with client.stream("POST", url, json=body, headers=request_headers(spec, key)) as r:
                if r.status_code != 200:
                    r.read()
                    q.put(("http", r.status_code, r.text, r.headers))
                    return
                for line in r.iter_lines():
                    if stop.is_set():
                        return
                    q.put(("line", line))
            q.put(("eof",))
        except Exception as e:
            q.put(("exc", e))

    threading.Thread(target=work, daemon=True).start()
    deadline, seen_token = time.monotonic() + first_s, False
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise Interrupted()
            wait = deadline - time.monotonic()
            if wait <= 0:
                what = f"no data for {idle_s:.0f}s mid-reply" if seen_token else f"no first token after {first_s:.0f}s"
                raise ProviderError("timeout", what, provider=spec.name, url=url)
            try:
                item = q.get(timeout=min(wait, 0.05))
            except queue.Empty:
                continue
            tag = item[0]
            if tag == "http":
                raise classify_http(item[1], item[2], spec.name, url, item[3])
            if tag == "exc":
                raise _exc_to_error(item[1], spec, url)
            if tag == "eof":
                return
            if seen_token:
                deadline = time.monotonic() + idle_s
            line = item[1]
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                yield DONE
                return
            if not payload:
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as e:
                raise ProviderError("server", f"bad stream data: {e}", provider=spec.name, url=url, raw=payload[:300]) from e
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk and not chunk.get("choices"):
                raise from_stream_error(chunk["error"], spec.name, url)
            if not seen_token and _has_token(chunk):
                seen_token = True
                deadline = time.monotonic() + idle_s
            yield chunk
    finally:
        stop.set()
        _close_later(client)


def _collect(spec: ProviderSpec, stream: Iterator, on_text, on_reasoning, emitted: list[int],
             on_first: Callable[[], None] = noop) -> Turn:
    turn = Turn(provider=spec.name)
    text: list[str] = []
    calls: list[ToolCall] = []
    slots: dict[int, ToolCall] = {}
    done, first = False, False
    url = spec.base_url
    try:
        for chunk in stream:
            if chunk is DONE:
                done = True
                break
            u = chunk.get("usage")
            if isinstance(u, dict):
                turn.usage = Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))
            for choice in chunk.get("choices") or []:
                d = choice.get("delta") or {}
                piece = d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if not first:
                        first = True
                        on_first()
                    turn.reasoning_chars += len(piece)
                    if on_reasoning:
                        emitted[0] += len(piece)
                        on_reasoning(piece)
                if d.get("content"):
                    if not first:
                        first = True
                        on_first()
                    text.append(d["content"])
                    emitted[0] += len(d["content"])
                    on_text(d["content"])
                for tc in d.get("tool_calls") or []:
                    if not first:
                        first = True
                        on_first()
                    idx, fn = tc.get("index", 0), tc.get("function") or {}
                    cur = slots.get(idx)
                    if cur is None or (tc.get("id") and cur.id and tc["id"] != cur.id):
                        cur = slots[idx] = ToolCall(tc.get("id") or "", "", "")
                        calls.append(cur)
                    cur.id = cur.id or tc.get("id") or ""
                    cur.name += fn.get("name") or ""
                    a = fn.get("arguments")
                    cur.args += a if isinstance(a, str) else (json.dumps(a) if a else "")
                if choice.get("finish_reason"):
                    turn.finish = choice["finish_reason"]
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()
    if not done and not turn.finish:
        raise ProviderError("network", "the stream ended before the reply was complete", provider=spec.name, url=url,
                            raw=f"stream cut after {sum(map(len, text))} chars")
    turn.text = "".join(text)
    turn.tool_calls = [ToolCall(c.id or f"call_{i}", c.name, c.args or "{}") for i, c in enumerate(calls) if c.name]
    if turn.finish == "tool_calls" and not turn.tool_calls and not turn.text:
        raise ProviderError("empty", "tool call finish with no tool call", provider=spec.name, url=url)
    if not turn.text and not turn.tool_calls:
        raise ProviderError("empty", f"no answer (finish={turn.finish or '?'}), {turn.reasoning_chars} chars of reasoning",
                            provider=spec.name, url=url,
                            raw=f"no answer (finish={turn.finish or '?'}), {turn.reasoning_chars} chars of reasoning")
    return turn


class Router:
    def __init__(self, cfg: Config, events: Callable[[str, dict], None] = noop, *, transport=None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float, threading.Event | None], None] | None = None):
        self.cfg, self.events, self.transport, self.clock = cfg, events, transport, clock
        self.sleep = sleep or _sleep
        self.active = cfg.chain[0] if cfg.chain else ""
        self.failed_over = False
        self.pinned: str | None = None
        self.down: dict[str, float] = {}
        self.models: dict[str, str] = {}
        self.last_usage: Usage | None = None

    def pin(self, name: str | None) -> None:
        if name is not None and name not in self.cfg.providers:
            raise KeyError(name)
        self.pinned = name
        if name:
            self.down.pop(name, None)
            self.active = name
        else:
            self.active = self.cfg.chain[0] if self.cfg.chain else ""
        self.failed_over = bool(self.cfg.chain) and self.active != self.cfg.chain[0] and not name

    def reset_cooldowns(self) -> None:
        self.down.clear()

    def context_tokens(self) -> int:
        spec = self.cfg.providers.get(self.active) or self.cfg.primary
        return (spec.context_tokens if spec else 0) or DEFAULT_CONTEXT

    def _order(self) -> list[str]:
        if self.pinned:
            return [self.pinned]
        names = [n for n in self.cfg.chain if n in self.cfg.providers]
        now = self.clock()
        up = [n for n in names if self.down.get(n, 0) <= now]
        return up + [n for n in names if n not in up]

    def _emit(self, ev: RouterEvent, on_event) -> None:
        self.events(ev.kind, {"from": ev.from_, "to": ev.to, "error": str(ev.error) if ev.error else ""})
        on_event(ev)

    def _key(self, spec: ProviderSpec) -> str | None:
        return get_secret(spec.api_key_env) if spec.api_key_env else None

    def _first_timeout(self, spec: ProviderSpec, est_tokens: int) -> float:
        extra = 0.0 if spec.kind == "openrouter" else est_tokens / 1000  # local prefill grows with the prompt
        return spec.first_token_timeout_s + extra

    def _sole_model(self, spec: ProviderSpec, key: str | None) -> str | None:
        try:
            r = httpx.Client(transport=self.transport, timeout=5).get(
                spec.base_url.rstrip("/") + "/models", headers=request_headers(spec, key))
            ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            return None
        return ids[0] if len(ids) == 1 and ids[0] != spec.model else None

    def _once(self, spec, key, messages, tools, on_text, on_reasoning, cancel, emitted) -> Turn:
        if spec.kind == "openrouter" and not key:
            raise ProviderError("auth", "no API key", provider=spec.name, url=spec.base_url,
                                raw=f"no API key found in ${spec.api_key_env or 'OPENROUTER_API_KEY'}")
        model = self.models.get(spec.name, spec.model)
        body = build_body(spec, model, messages, tools)
        stream = _sse(spec, key, body, cancel=cancel, first_s=self._first_timeout(spec, estimate_tokens(messages)),
                      idle_s=spec.idle_timeout_s, transport=self.transport)
        return _collect(spec, stream, on_text, on_reasoning, emitted)

    def chat(self, messages: list[dict], tools: list[dict], on_text: Callable[[str], None],
             on_reasoning: Callable[[str], None] | None = None, on_event: Callable[[RouterEvent], None] = noop,
             cancel: threading.Event | None = None) -> Turn:
        names = self._order()
        if not names:
            raise ProviderError("not_configured", "no providers configured")
        last_err: ProviderError | None = None
        for i, name in enumerate(names):
            spec, key = self.cfg.providers[name], self._key(self.cfg.providers[name])
            is_last = i == len(names) - 1
            attempt, tried_sole = 0, False
            while True:
                if cancel is not None and cancel.is_set():
                    raise Interrupted()
                emitted = [0]
                try:
                    turn = self._once(spec, key, messages, tools, on_text, on_reasoning, cancel, emitted)
                except ProviderError as e:
                    e.provider = e.provider or name
                    last_err = e
                    if e.kind == "not_found" and spec.kind != "openrouter" and not tried_sole:
                        tried_sole = True
                        if (m := self._sole_model(spec, key)):
                            self.models[name] = m
                            continue
                    if not should_failover(e):
                        raise
                    if is_last and not emitted[0] and e.kind in RETRY_KINDS and attempt < RETRIES:
                        self._emit(RouterEvent("retry", name, name, e), on_event)
                        self.sleep(retry_delay(attempt, e.retry_after), cancel)
                        attempt += 1
                        continue
                    if e.kind != "empty":
                        self.down[name] = self.clock() + COOLDOWN_S
                    if is_last:
                        raise
                    self._emit(RouterEvent("failover", name, names[i + 1], e), on_event)
                    if emitted[0]:
                        self._emit(RouterEvent("restart", name, names[i + 1], e), on_event)
                    break
                if turn.usage:
                    self.last_usage = turn.usage
                was_over, self.active = self.failed_over, name
                self.failed_over = not self.pinned and name != self.cfg.chain[0]
                self.down.pop(name, None)
                if was_over and not self.failed_over:
                    self._emit(RouterEvent("recovered", "", name), on_event)
                return turn
        raise last_err or ProviderError("not_configured", "no providers configured")


def _sleep(seconds: float, cancel: threading.Event | None) -> None:
    if cancel is None:
        time.sleep(seconds)
    elif cancel.wait(seconds):
        raise Interrupted()


def probe(spec: ProviderSpec, key: str | None, *, with_tools: bool = True, transport=None) -> ProbeResult:
    """One tiny streaming chat (time to first token), then a get_weather tool-call check."""
    first_s, idle_s = max(spec.first_token_timeout_s, 30.0), 30.0
    ttft: list[float] = []
    t0 = time.monotonic()
    sink = [0]

    def run(messages, tools):
        body = build_body(spec, spec.model, messages, tools, max_tokens=80, thinking="off")
        stream = _sse(spec, key, body, cancel=None, first_s=first_s, idle_s=idle_s, transport=transport)
        return _collect(spec, stream, noop, None, sink, on_first=lambda: ttft.append(time.monotonic() - t0))

    def fail(e: ProviderError, chat_ok: bool) -> ProbeResult:
        e.provider = e.provider or spec.name
        return ProbeResult(False, chat_ok, False, int(ttft[0] * 1000) if ttft else 0, e, friendly(e, spec)[0])

    try:
        run([{"role": "user", "content": "Reply with the single word OK."}], [])
    except ProviderError as e:
        if not (e.kind == "empty" and ttft):  # a thinking model may burn its 80 tokens; the stream still worked
            return fail(e, False)
    ms = int(ttft[0] * 1000) if ttft else int((time.monotonic() - t0) * 1000)
    if not with_tools:
        return ProbeResult(True, True, False, ms, None, "")
    try:
        turn = run([{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}], [WEATHER_TOOL])
    except ProviderError as e:
        if e.kind in ("bad_request", "not_found", "empty") or "tool" in e.raw.lower():
            return ProbeResult(False, True, False, ms, None, _no_tools_hint(spec, e.raw))
        return fail(e, True)
    if any(c.name == "get_weather" for c in turn.tool_calls):
        return ProbeResult(True, True, True, ms, None, "")
    return ProbeResult(False, True, False, ms, None, _no_tools_hint(spec, ""))


def _no_tools_hint(spec: ProviderSpec, raw: str) -> str:
    if spec.kind == "llamacpp" or "jinja" in raw.lower():
        return "Tool calls need llama-server started with --jinja. Restart it with --jinja, or run with tools off: jarvoos config set providers.%s.tools false" % spec.name
    return "This model did not make a tool call, so it probably has no tool support. Pick another model, or run chat-only: jarvoos config set providers.%s.tools false" % spec.name
