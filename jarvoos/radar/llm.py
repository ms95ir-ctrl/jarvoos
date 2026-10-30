"""One-call model access through the same provider chain the chat uses (local first, cloud fallback)."""
from __future__ import annotations

import dataclasses
import json
import re

from .. import config
from ..providers import ProviderError, Router
from ..ui import ThinkFilter


class LLMUnavailable(Exception):
    pass


def _router() -> Router:
    """Radar prompts are structured one-shot jobs: reasoning only slows them down (minutes on a local 35B), so it is switched off."""
    cfg = config.load()
    return Router(dataclasses.replace(cfg, providers={n: dataclasses.replace(p, thinking="off") for n, p in cfg.providers.items()}))


def complete(prompt: str, system: str = "", router: Router | None = None) -> str:
    """Plain text answer, reasoning blocks removed. Raises LLMUnavailable when no provider answers."""
    try:
        r = router or _router()
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        turn = r.chat(msgs, [], lambda s: None)
    except (config.NotConfigured, ProviderError) as e:
        raise LLMUnavailable(str(e)) from e
    flt = ThinkFilter()
    return (flt.feed(turn.text) + flt.flush()).strip()


def parse_json(text: str, want: type = list):
    """First JSON value of the wanted type in a model answer (fenced or bare); None when there is none."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    o, c = ("[", "]") if want is list else ("{", "}")
    a, b = s.find(o), s.rfind(c)
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(s[a:b + 1])
    except ValueError:
        return None
    return v if isinstance(v, want) else None
