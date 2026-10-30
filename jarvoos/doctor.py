"""`jarvoos doctor`: check the install, the config and every provider, and say exactly how to fix what is wrong."""
from __future__ import annotations

import importlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from . import config, detect
from .config import Config, ProviderSpec

_TAGS = {"ok": ("OK", "green"), "warn": ("WARN", "yellow"), "fail": ("FAIL", "red")}


@dataclass
class Check:
    name: str
    level: str  # ok|warn|fail
    detail: str
    fix: str = ""


def _python() -> Check:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return Check("python", "ok", ver)
    return Check("python", "fail", f"{ver} is too old, Jarvoos needs 3.11+", "install Python 3.11 or newer and rerun the installer")


def _secrets_perms() -> Check:
    path = config.SECRETS_PATH
    if not path.exists():
        return Check("secrets file", "ok", "none (keys come from the environment)")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        return Check("secrets file", "fail", f"{path} is mode {mode:o}, other users can read your keys", f"chmod 600 {path}")
    return Check("secrets file", "ok", f"{path} (mode 600)")


def _tools() -> list[Check]:
    out = []
    for exe, why in (("rg", "fast code search; grep is used instead"), ("git", "diffs and repo awareness")):
        if shutil.which(exe):
            out.append(Check(exe, "ok", shutil.which(exe) or ""))
        else:
            out.append(Check(exe, "warn", f"not installed ({why})", f"sudo apt install {'ripgrep' if exe == 'rg' else exe}   (or your package manager)"))
    try:
        n = len(importlib.import_module("jarvoos.skills").load_all())
        out.append(Check("skills", "ok", f"{n} loaded"))
    except Exception as e:
        out.append(Check("skills", "warn", f"could not load skills: {e}", "check ~/.config/jarvoos/skills/*.md frontmatter"))
    return out


def _search(cfg: Config) -> Check:
    s = cfg.search
    prov = s.get("provider", "ddg")
    try:
        if prov == "exa":
            if config.get_secret("EXA_API_KEY"):
                return Check("search", "ok", "exa (key set)")
            return Check("search", "warn", "exa selected but EXA_API_KEY is not set", "jarvoos setup search")
        url = (s.get("searxng_url", "").rstrip("/") + "/search?q=test&format=json") if prov == "searxng" else "https://duckduckgo.com/html/?q=jarvoos"
        if prov == "searxng" and not s.get("searxng_url"):
            return Check("search", "warn", "searxng selected but searxng_url is empty", "jarvoos setup search")
        with detect.client(6) as c:
            r = c.get(url, headers={"User-Agent": "jarvoos-doctor"})
        if r.status_code < 400:
            return Check("search", "ok", f"{prov} reachable")
        return Check("search", "warn", f"{prov} answered HTTP {r.status_code}", "jarvoos setup search")
    except Exception as e:
        return Check("search", "warn", f"{prov} not reachable ({type(e).__name__}); is this machine online?", "jarvoos setup search")


def _provider(name: str, spec: ProviderSpec, lite: bool) -> list[Check]:
    rows: list[Check] = []
    key = config.get_secret(spec.api_key_env) if spec.api_key_env else None
    label = f"{name} ({spec.model})"
    if spec.api_key_env and not key:
        return [Check(f"{name} key", "fail", f"{spec.api_key_env} is not set", f"export {spec.api_key_env}=...   or   jarvoos setup model")]
    models: list[str] = []
    try:
        with detect.client(6) as c:
            st, data, err = detect._get(c, spec.base_url.rstrip("/") + "/models", key)
    except Exception as e:
        st, data, err = 0, None, type(e).__name__
    if err == "refused":
        return [Check(f"{name} reachable", "fail", f"connection refused at {spec.base_url}", "start the server, or run: jarvoos setup model")]
    if err:
        return [Check(f"{name} reachable", "fail", f"{err} talking to {spec.base_url}", "check the URL and your network, or run: jarvoos setup model")]
    if st in (401, 403):
        return [Check(f"{name} key", "fail", f"{spec.base_url} rejected the key (HTTP {st})", "jarvoos setup model" if name == "local" else "jarvoos setup fallback")]
    if st == 503:
        rows.append(Check(f"{name} reachable", "warn", "model is still loading", "wait a few seconds and rerun: jarvoos doctor"))
    elif st != 200:
        return [Check(f"{name} reachable", "fail", f"HTTP {st} from {spec.base_url}/models", "check the URL (it usually ends in /v1): jarvoos config set providers.%s.base_url ..." % name)]
    else:
        models = detect._ids(data)
        rows.append(Check(f"{name} reachable", "ok", f"{spec.base_url}, {len(models)} model(s)"))
    if spec.kind == "openrouter" and key:
        ok, msg = detect.openrouter_check_key(key)
        rows.append(Check(f"{name} key", "ok" if ok else "fail", msg, "" if ok else "get a new key at https://openrouter.ai/keys, then: jarvoos setup fallback"))
    if models and spec.model not in models and spec.kind != "strata":
        rows.append(Check(f"{name} model", "warn", f"{spec.model!r} is not in the server's model list", "jarvoos setup model" if name == "local" else "jarvoos setup fallback"))
    elif models:
        rows.append(Check(f"{name} model", "ok", spec.model))
    if spec.kind == "ollama" and (spec.context_tokens == 0 or spec.context_tokens > detect.OLLAMA_DEFAULT_CTX):
        rows.append(Check(f"{name} context", "warn", f"Ollama runs with {detect.OLLAMA_DEFAULT_CTX} tokens unless told otherwise, config says {spec.context_tokens or 'unknown'}",
                          "restart Ollama with OLLAMA_CONTEXT_LENGTH=32768 and set providers.%s.context_tokens to match" % name))
    if lite:
        return rows
    try:
        res = importlib.import_module("jarvoos.providers").probe(spec, key, with_tools=spec.tools)
    except Exception as e:
        return rows + [Check(f"{name} chat", "warn", f"probe unavailable: {type(e).__name__}: {e}")]
    if res.chat_ok:
        rows.append(Check(f"{name} chat", "ok", f"{res.ttft_ms}ms to first token"))
        if spec.tools and not res.tools_ok:
            fix = "restart llama-server with --jinja" if spec.kind == "llamacpp" else f"use a model with tool support, or: jarvoos config set providers.{name}.tools false"
            rows.append(Check(f"{name} tools", "warn", f"{label} did not make a tool call", fix))
        elif spec.tools:
            rows.append(Check(f"{name} tools", "ok", "tool call works"))
    else:
        detail, fix = res.hint or "no response", "jarvoos setup model" if name == "local" else "jarvoos setup fallback"
        if res.error is not None:
            try:
                detail = importlib.import_module("jarvoos.errors").friendly(res.error, spec)[0]
            except Exception:
                detail = f"{res.error.kind}: {res.error.raw[:200]}"
        rows.append(Check(f"{name} chat", "fail", detail, fix))
    return rows


def run_checks(cfg: Config | None, live: bool = True, lite: bool = False) -> list[Check]:
    """lite = config, secrets and provider reachability only (used at the end of setup, which already ran the chat test)."""
    jobs: list[Callable[[], list[Check]]] = [lambda: [_python()]]
    if cfg is None:
        why = config.problem() or "no providers configured"
        jobs.append(lambda: [Check("config", "fail", why, "jarvoos setup")])
    else:
        jobs.append(lambda: [Check("config", "ok", f"{config.CONFIG_PATH}: {' -> '.join(cfg.chain)}, mode {cfg.mode}")])
    jobs.append(lambda: [_secrets_perms()])
    if cfg is not None:
        for name in cfg.chain:
            spec = cfg.providers[name]
            if live:
                jobs.append(lambda n=name, s=spec: _provider(n, s, lite))
            else:
                jobs.append(lambda n=name, s=spec: [Check(n, "ok", f"{s.kind} {s.model} at {s.base_url} (not tested)")])
        if live and not lite:
            jobs.append(lambda: [_search(cfg)])
    if not lite:
        jobs.append(_tools)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(j) for j in jobs]
    return [c for f in futures for c in f.result()]


def rows(checks: list[Check]) -> list[str]:
    out = [f"{_TAGS[c.level][0]:<5} {c.name:<22} {c.detail}" for c in checks]
    issues = [c for c in checks if c.level != "ok"]
    if issues:
        out += ["", "To fix:"]
        for n, c in enumerate(issues, 1):
            out.append(f"{n}. {c.name}: {c.detail}")
            if c.fix:
                out.append(f"   {c.fix}")
    return out


def render(checks: list[Check], console=None) -> int:
    from rich.console import Console
    console = console or Console(highlight=False)
    lines = rows(checks)
    for c, line in zip(checks, lines):
        console.print(line[:5], style=_TAGS[c.level][1], end="", markup=False)
        console.print(line[5:], markup=False)
    for line in lines[len(checks):]:
        console.print(line, markup=False)
    fails = sum(c.level == "fail" for c in checks)
    warns = sum(c.level == "warn" for c in checks)
    console.print(f"\n{fails} problem(s), {warns} warning(s)" if fails or warns else "\nAll good.", markup=False)
    return 1 if fails else 0


def main(fix: bool = False) -> int:
    if fix:
        done = []
        for d in (config.HOME_DIR, config.STATE_DIR):
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                done.append(f"created {d}")
        sp = config.SECRETS_PATH
        if sp.exists() and sp.stat().st_mode & 0o077:
            sp.chmod(0o600)
            done.append(f"chmod 600 {sp}")
        for line in done or ["nothing to fix automatically"]:
            print(f"fix: {line}")
    try:
        cfg = config.load()
    except config.NotConfigured:
        cfg = None
    return render(run_checks(cfg))
