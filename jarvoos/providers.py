"""OpenAI-compatible streaming providers and the failover router.

Both the local server (Strata, llama.cpp, ...) and OpenRouter speak the same chat-completions
dialect, so there is no message translation when a conversation moves between them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .config import EVENTS_LOG, STATE_DIR


class ProviderError(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(f"{kind}: {msg}")
        self.kind, self.msg = kind, msg


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[tuple[str, ProviderError]]):
        super().__init__("; ".join(f"{n}: {e}" for n, e in errors))
        self.errors = errors


@dataclass
class Turn:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {id, name, args(str)}
    finish: str = ""
    usage: dict = field(default_factory=dict)
    provider: str = ""


class Provider:
    def __init__(self, name: str, base_url: str, model: str, api_key: str = "",
                 thinking: str = "default", connect_timeout: float = 3.0, headers: dict | None = None):
        self.name, self.model, self.api_key, self.thinking = name, model, api_key, thinking
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.connect_timeout = connect_timeout
        self.headers = headers or {}

    def stream(self, messages: list[dict], tools: list[dict], *, first_token_timeout: float,
               on_text: Callable[[str], None] | None = None,
               on_reasoning: Callable[[str], None] | None = None) -> Turn:
        body: dict = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
        if self.thinking == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.thinking == "on":
            body["chat_template_kwargs"] = {"enable_thinking": True}
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(first_token_timeout, connect=self.connect_timeout)
        turn = Turn(provider=self.name)
        calls: dict[int, dict] = {}
        try:
            with httpx.stream("POST", self.url, json=body, headers=headers, timeout=timeout) as r:
                if r.status_code != 200:
                    r.read()
                    raise ProviderError("http", f"{r.status_code} {r.text[:200]}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue  # SSE comments / keep-alives
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise ProviderError("http", str(chunk["error"])[:200])
                    if chunk.get("usage"):
                        turn.usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        d = choice.get("delta") or {}
                        if d.get("reasoning_content") or d.get("reasoning"):
                            piece = d.get("reasoning_content") or d.get("reasoning")
                            turn.reasoning += piece
                            if on_reasoning:
                                on_reasoning(piece)
                        if d.get("content"):
                            turn.text += d["content"]
                            if on_text:
                                on_text(d["content"])
                        for tc in d.get("tool_calls") or []:
                            c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                            c["id"] = tc.get("id") or c["id"]
                            fn = tc.get("function") or {}
                            c["name"] += fn.get("name") or ""
                            c["args"] += fn.get("arguments") or ""
                        if choice.get("finish_reason"):
                            turn.finish = choice["finish_reason"]
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ProviderError("connect", str(e) or type(e).__name__) from e
        except httpx.TimeoutException as e:
            raise ProviderError("timeout", f"no data for {first_token_timeout:.0f}s") from e
        except httpx.HTTPError as e:
            raise ProviderError("network", str(e) or type(e).__name__) from e
        except json.JSONDecodeError as e:
            raise ProviderError("http", f"bad stream data: {e}") from e
        turn.tool_calls = [
            {**c, "id": c["id"] or f"call_{i}"} for i, c in sorted(calls.items())
        ]
        if not turn.text and not turn.tool_calls:
            raise ProviderError("empty", f"no answer (finish={turn.finish or '?'}, "
                                         f"{len(turn.reasoning)} chars of reasoning)")
        return turn


def estimate_tokens(messages: list[dict]) -> int:
    return int(len(json.dumps(messages, ensure_ascii=False)) / 3.5)


class Router:
    """Primary first; on any failure switch to the fallback at once. After a primary failure the
    fallback is used for `cooldown` seconds, then the primary is probed again."""

    def __init__(self, primary: Provider, fallback: Provider | None, failover_cfg: dict,
                 notifiers: list[Callable[[dict], None]] | None = None):
        self.primary, self.fallback, self.cfg = primary, fallback, failover_cfg
        self.notifiers = notifiers or []
        self.down_until = 0.0
        self.primary_bad = False
        self.forced: Provider | None = None
        self.active = primary.name

    def _timeout(self, p: Provider, est_tokens: int) -> float:
        if p is self.primary:
            return self.cfg["first_token_base"] + self.cfg["first_token_per_1k"] * est_tokens / 1000
        return self.cfg["fallback_timeout"]

    def _chain(self) -> list[Provider]:
        if self.forced:
            return [self.forced]
        if not self.fallback:
            return [self.primary]
        if time.time() < self.down_until:
            return [self.fallback, self.primary]
        return [self.primary, self.fallback]

    def emit(self, kind: str, **info) -> None:
        event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **info}
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(EVENTS_LOG, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass
        for n in self.notifiers:
            try:
                n(event)
            except Exception:
                pass

    def complete(self, messages, tools, *, on_text=None, on_reasoning=None, on_reset=None) -> Turn:
        est = estimate_tokens(messages)
        chain = self._chain()
        errors: list[tuple[str, ProviderError]] = []
        for i, p in enumerate(chain):
            try:
                turn = p.stream(messages, tools, first_token_timeout=self._timeout(p, est),
                                on_text=on_text, on_reasoning=on_reasoning)
            except ProviderError as e:
                errors.append((p.name, e))
                if p is self.primary and not self.forced:
                    self.down_until = time.time() + self.cfg["cooldown"]
                    if not self.primary_bad and self.fallback:
                        self.emit("failover", frm=p.name, to=self.fallback.name, reason=str(e))
                    self.primary_bad = True
                if i + 1 < len(chain) and on_reset:
                    on_reset(p.name, chain[i + 1].name, str(e))
                continue
            if p is self.primary and self.primary_bad:
                self.down_until, self.primary_bad = 0.0, False
                self.emit("recovered", provider=p.name)
            self.active = p.name
            return turn
        raise AllProvidersFailed(errors)
