"""Setup wizard: find a server, pick a model, test it for real, then save once.

Nothing is written until the summary is confirmed. All I/O goes through SetupIO so tests can script it.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import config, detect, doctor
from .config import Config, ProviderSpec

OR_ENV = "OPENROUTER_API_KEY"
OR_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
OR_KEYS_URL = "https://openrouter.ai/keys"
SECTIONS = ("model", "fallback", "search", "mode")
LOAD_WAIT_S = 30
KEY_TRIES = 3
_sleep = time.sleep
_MODE_LABELS = ["approve (ask before write/exec) [default]", "auto (run everything without asking)", "chat (no tools)"]
_PRESETS = [("Strata", "strata", "http://127.0.0.1:8090/v1"), ("llama-server (llama.cpp)", "llamacpp", "http://127.0.0.1:8080/v1"),
            ("Ollama", "ollama", "http://127.0.0.1:11434/v1"), ("LM Studio", "lmstudio", "http://127.0.0.1:1234/v1"),
            ("Other OpenAI-compatible URL", "openai", "")]


class SetupIO(Protocol):
    def ask(self, prompt: str, default: str = "") -> str: ...
    def ask_secret(self, prompt: str) -> str: ...
    def choose(self, prompt: str, options: list[str], default: int = 0) -> int: ...
    def confirm(self, prompt: str, default: bool = True) -> bool: ...
    def say(self, msg: str, style: str = "") -> None: ...


class RichIO:
    def __init__(self):
        from rich.console import Console
        self.console = Console(highlight=False)

    def say(self, msg: str, style: str = "") -> None:
        self.console.print(msg, style=style or None, markup=False)

    def ask(self, prompt: str, default: str = "") -> str:
        from rich.markup import escape
        from rich.prompt import Prompt
        return Prompt.ask(escape(prompt), default=default, show_default=bool(default), console=self.console)

    def ask_secret(self, prompt: str) -> str:
        from rich.markup import escape
        from rich.prompt import Prompt
        return Prompt.ask(escape(prompt), password=True, default="", show_default=False, console=self.console)

    def choose(self, prompt: str, options: list[str], default: int = 0) -> int:
        self.say(prompt)
        for n, o in enumerate(options, 1):
            self.say(f"  {n}) {o}")
        while True:
            a = self.ask(f"Choose 1-{len(options)}", str(default + 1)).strip()
            if a.isdigit() and 1 <= int(a) <= len(options):
                return int(a) - 1
            self.say("Enter one of the numbers above.", "yellow")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        from rich.markup import escape
        from rich.prompt import Confirm
        return Confirm.ask(escape(prompt), default=default, console=self.console)


@dataclass
class Entry:
    spec: ProviderSpec
    secrets: dict[str, str] = field(default_factory=dict)  # typed keys to persist
    redo: Callable[[], "Entry | None"] | None = None

    @property
    def key(self) -> str | None:
        return self.secrets.get(self.spec.api_key_env) or config.get_secret(self.spec.api_key_env)


def noninteractive_help() -> str:
    return """Jarvoos is not set up and there is no terminal to ask questions.
Either run `jarvoos setup` in a terminal, or configure it by hand:

  export OPENROUTER_API_KEY=sk-or-...        # a key from https://openrouter.ai/keys
  jarvoos config set chain openrouter
  jarvoos config set providers.openrouter.kind openrouter
  jarvoos config set providers.openrouter.base_url https://openrouter.ai/api/v1
  jarvoos config set providers.openrouter.model deepseek/deepseek-v4-flash
  jarvoos config set providers.openrouter.api_key_env OPENROUTER_API_KEY

For a local server (Strata, llama-server --jinja, Ollama, LM Studio) add a provider the same way:
  jarvoos config set providers.local.kind strata
  jarvoos config set providers.local.base_url http://127.0.0.1:8090/v1
  jarvoos config set providers.local.model <model id from /v1/models>
  jarvoos config set chain local,openrouter     # first = primary, the rest are fallbacks

Config file: ~/.config/jarvoos/config.toml (see config.example.toml). Keys: env vars or ~/.config/jarvoos/secrets.env.
Check the result with: jarvoos doctor"""


def run(first_run: bool = False, section: str | None = None, io: SetupIO | None = None) -> bool:
    io = io or RichIO()
    if section is not None and section not in SECTIONS:
        raise ValueError(f"section must be one of {', '.join(SECTIONS)}")
    try:
        return _run(io, first_run, section)
    except (KeyboardInterrupt, EOFError):
        io.say("")
        io.say("Setup cancelled, nothing saved")
        return False


# ---- providers ----

def _mod(name: str):
    return importlib.import_module(name)


def _unique(taken: list[str], base: str) -> str:
    name, n = base, 2
    while name in taken:
        name, n = f"{base}{n}", n + 1
    taken.append(name)
    return name


def _short(models: list[str]) -> str:
    return models[0] if len(models) == 1 else f"{len(models)} models" if models else "no model listed"


def _ctx(n: int) -> str:
    return f"{n // 1000}k ctx" if n >= 1000 else (f"{n} ctx" if n else "ctx unknown")


def _search_pick(io: SetupIO, items: list[tuple[str, str]], default: str) -> str:
    """Type to filter, number to pick, Enter for the default, '!id' to use an id as typed."""
    ids = [i for i, _ in items]
    query = ""
    while True:
        words = query.lower().split()
        matches = [it for it in items if all(w in it[0].lower() for w in words)]
        if not query and default in ids:
            matches = [it for it in items if it[0] == default] + [it for it in matches if it[0] != default]
        shown = matches[:10]
        io.say(f"{len(matches)} model(s)" + (f" matching '{query}'" if query else "") + (f", showing {len(shown)}" if len(shown) < len(matches) else ""))
        for n, (i, desc) in enumerate(shown, 1):
            io.say(f"  {n:>2}) {i}  {desc}")
        ans = io.ask("Number to pick, text to search, !id to type one" + (f", Enter = {default}" if default else ""), "").strip()
        if not ans:
            if default:
                return default
        elif ans.startswith("!") and len(ans) > 1:
            return ans[1:]
        elif ans.isdigit() and 1 <= int(ans) <= len(shown):
            return shown[int(ans) - 1][0]
        elif ans in ids:
            return ans
        else:
            query = ans
            hit = [it for it in items if all(w in it[0].lower() for w in ans.lower().split())]
            if len(hit) == 1 and io.confirm(f"Use {hit[0][0]}?", True):
                return hit[0][0]


def _local_model(io: SetupIO, models: list[str]) -> str:
    if len(models) == 1:
        if io.confirm(f"Use {models[0]}?", True):
            return models[0]
    elif 2 <= len(models) <= 8:
        i = io.choose("Which model?", models + ["Type a name"])
        if i < len(models):
            return models[i]
    elif models:
        return _search_pick(io, [(m, "") for m in models], models[0])
    while True:
        m = io.ask("Model name").strip()
        if m:
            return m


def _connect(io: SetupIO, url: str):
    """Probe until the server answers; returns (UrlProbe, key) or None to skip."""
    key: str | None = None
    tries = 0
    while True:
        key = key or config.get_secret(config.env_name_for(url))
        up = detect.probe_url(url, key)
        if up.status == "loading":
            io.say(f"The model is still loading, waiting up to {LOAD_WAIT_S}s...")
            for _ in range(LOAD_WAIT_S):
                _sleep(1)
                up = detect.probe_url(url, key)
                if up.status != "loading":
                    break
        if up.ok and not (up.needs_key and not key):
            return up, key
        if up.status == "unauthorized" or (up.ok and up.needs_key):
            if tries >= KEY_TRIES:
                io.say("Too many rejected keys.", "yellow")
            else:
                env = config.env_name_for(up.base_url or url)
                io.say(("That key was rejected. " if key else "") + f"This server needs an API key; it will be stored as {env} in secrets.env.")
                key = config.clean_secret(io.ask_secret("API key (input hidden)")) or None
                tries += 1
                continue
        else:
            io.say(up.hint or f"Could not use {url} ({up.status}).", "yellow")
        c = io.choose("What now?", ["Try again", "Change the URL", "Skip this server"])
        if c == 0:
            continue
        if c == 1:
            url, key, tries = io.ask("Server URL", url).strip(), None, 0
            continue
        return None


def _local(io: SetupIO, kind: str, url: str, taken: list[str], found: detect.Found | None = None) -> Entry | None:
    url = io.ask("Server URL", url).strip()
    res = _connect(io, url)
    if res is None:
        return None
    up, key = res
    model = _local_model(io, up.models)
    k = up.kind if up.kind != "openai" or kind in ("openai", "openrouter") else kind
    if k == "ollama":
        io.say(f"Warning: Ollama uses a {detect.OLLAMA_DEFAULT_CTX}-token context by default, too small for an agent. "
               "Start it with OLLAMA_CONTEXT_LENGTH=32768 and set context_tokens in config.toml.", "yellow")
    thinking = "default"
    if k in ("strata", "llamacpp"):
        thinking = "on" if io.confirm("Enable thinking mode? (slower, may reason better; default off)", False) else "off"
    env = config.env_name_for(up.base_url) if key else ""
    spec = ProviderSpec(_unique(taken, "local" if k in ("strata", "llamacpp") else k if k != "openai" else "custom"),
                        k, up.base_url, model, env, up.n_ctx, True, thinking)
    return Entry(spec, {env: key} if key and key != config.get_secret(env) else {}, lambda: taken.remove(spec.name) or _local(io, kind, spec.base_url, taken))


def _openrouter(io: SetupIO, taken: list[str]) -> Entry | None:
    key = config.get_secret(OR_ENV)
    typed = ""
    if key:
        ok, msg = detect.openrouter_check_key(key)
        if ok:
            io.say(f"Using {OR_ENV} ({config.mask(key)}).")
        else:
            io.say(f"The key in {OR_ENV} does not work ({msg}).", "yellow")
            key = None
    if not key:
        io.say(f"Get a key at {OR_KEYS_URL}")
        for _ in range(KEY_TRIES):
            key = config.clean_secret(io.ask_secret("Paste your OpenRouter API key (input hidden)"))
            if not key:
                io.say("Nothing received. Paste the key and press Enter.", "yellow")
                continue
            ok, msg = detect.openrouter_check_key(key)
            if ok:
                typed = key
                break
            if msg.startswith("unreachable") and io.confirm("Could not reach OpenRouter to check the key. Use it anyway?", False):
                typed = key
                break
            io.say(f"Key rejected ({msg.split(': ', 1)[-1]}). Paste it again.", "yellow")
            key = None
        else:
            io.say("Skipping OpenRouter after 3 rejected keys.", "yellow")
        if not key:
            return None
    models = detect.openrouter_models()
    ctx = 0
    if models:
        model = _search_pick(io, [(m.id, f"{_ctx(m.context)}  ${m.price_in:g}/${m.price_out:g} per M in/out") for m in models], OR_DEFAULT_MODEL)
        ctx = next((m.context for m in models if m.id == model), 0)
    else:
        io.say(f"Could not load the model list; the default is {OR_DEFAULT_MODEL}.", "yellow")
        model = io.ask("OpenRouter model id", OR_DEFAULT_MODEL).strip() or OR_DEFAULT_MODEL
    spec = ProviderSpec(_unique(taken, "openrouter"), "openrouter", detect.OPENROUTER_URL, model, OR_ENV, ctx, True, "default", 30)
    return Entry(spec, {OR_ENV: typed} if typed else {}, lambda: taken.remove(spec.name) or _openrouter(io, taken))


def _backend(io: SetupIO, found: list[detect.Found], taken: list[str]) -> Entry | None:
    def local(kind: str, url: str, f: detect.Found | None = None):
        return lambda: _local(io, kind, url, taken, f)
    labels: list[str] = []
    acts: list[Callable[[], Entry | None]] = []
    for f in found:
        labels.append(f"{f.label}  ({'loading model' if f.loading else _short(f.models) + ', ' + _ctx(f.n_ctx)})  [found]")
        acts.append(local(f.kind, f.base_url, f))
    if found:
        io.say(f"Found {len(found)} local server(s).")
        labels.append("OpenRouter (cloud, recommended fallback)")
        acts.append(lambda: _openrouter(io, taken))
        labels.append("Ollama / LM Studio / other OpenAI-compatible URL")
        acts.append(lambda: _backend_preset(io, taken, with_openrouter=False))
    else:
        io.say("No local server found.")
        return _backend_preset(io, taken, with_openrouter=True)
    return acts[io.choose("Which backend do you want to use?", labels, 0)]()


def _backend_preset(io: SetupIO, taken: list[str], with_openrouter: bool) -> Entry | None:
    labels = [p[0] for p in _PRESETS]
    if with_openrouter:
        labels.insert(len(_PRESETS) - 1, "OpenRouter (cloud, needs an API key)")
    i = io.choose("Which backend do you want to use?", labels, 0)
    if with_openrouter and i == len(_PRESETS) - 1:
        return _openrouter(io, taken)
    _, kind, url = _PRESETS[i if not with_openrouter or i < len(_PRESETS) - 1 else i - 1]
    return _local(io, kind, url, taken)


# ---- testing ----

def _test(io: SetupIO, e: Entry) -> bool:
    providers = _mod("jarvoos.providers")
    while True:
        s = e.spec
        io.say(f"Testing {s.name} ({s.model})...")
        try:
            res = providers.probe(s, e.key, with_tools=s.tools)
        except Exception as ex:
            res = None
            io.say(f"The test itself failed: {type(ex).__name__}: {ex}", "red")
        if res is not None and res.chat_ok:
            if res.tools_ok or not s.tools:
                io.say(f"OK  {res.ttft_ms}ms to first token, " + ("tools OK" if s.tools else "chat only"), "green")
                return True
            io.say(f"Chat works ({res.ttft_ms}ms to first token) but the tool-call test failed. "
                   + ("Restart llama-server with --jinja." if s.kind == "llamacpp" else "This model has no tool support."), "yellow")
            if io.confirm("Continue chat-only?", False):
                s.tools = False
                return True
        elif res is not None:
            if res.error is not None:
                msg, details = _mod("jarvoos.errors").friendly(res.error, s)
                io.say(msg, "red")
                io.say(details)
            else:
                io.say(res.hint or "No answer from the model.", "red")
            io.say(f"URL tried: {s.base_url.rstrip('/')}/chat/completions")
        c = io.choose(f"{s.name} is not working yet", ["Retry", "Edit settings", "Skip this provider"])
        if c == 1 and e.redo:
            new = e.redo()
            if new is None:
                return False
            e.spec, e.secrets, e.redo = new.spec, new.secrets, new.redo
        elif c == 2:
            return False


# ---- sections ----

def _ask_mode(io: SetupIO, current: str = "approve") -> str:
    return config.MODES[io.choose("What may Jarvoos do without asking?", _MODE_LABELS, config.MODES.index(current))]


def _setup_search(io: SetupIO, cfg: Config, secrets: dict[str, str]) -> None:
    cur = cfg.search.get("provider", "ddg")
    opts = ["DuckDuckGo (no key needed) [default]", "SearXNG (your own instance, no key)", "Exa (API key, better results)"]
    c = io.choose("Web search backend", opts, {"ddg": 0, "searxng": 1, "exa": 2}.get(cur, 0))
    if c == 0:
        cfg.search["provider"] = "ddg"
    elif c == 1:
        cfg.search.update(provider="searxng", searxng_url=io.ask("SearXNG URL", cfg.search.get("searxng_url", "")).strip())
    else:
        io.say("Get a key at https://dashboard.exa.ai/api-keys")
        key = config.clean_secret(io.ask_secret("Exa API key (input hidden, Enter keeps the current one)"))
        if key:
            secrets["EXA_API_KEY"] = key
        cfg.search["provider"] = "exa"


def _wizard(io: SetupIO, taken: list[str], ask_fallback: bool) -> list[Entry] | None:
    while True:
        io.say("Looking for local servers...")
        entries: list[Entry] = []
        e = _backend(io, detect.scan(), taken)
        if e:
            entries.append(e)
        if e and ask_fallback and e.spec.kind != "openrouter" and io.confirm("Also add OpenRouter as fallback?", bool(config.get_secret(OR_ENV))):
            o = _openrouter(io, taken)
            if o:
                entries.append(o)
        ok = [x for x in entries if _test(io, x)]
        if ok:
            return ok
        io.say("Nothing passed the test, so nothing was saved.", "yellow")
        taken.clear()
        if not io.confirm("Try again?", True):
            return None


def _summary(io: SetupIO, cfg: Config, secrets: dict[str, str]) -> None:
    io.say("")
    io.say("Summary")
    io.say(f"  {'role':<9}{'provider':<12}{'model':<36}{'context':<12}{'tools':<7}key")
    for i, n in enumerate(cfg.chain):
        p = cfg.providers[n]
        key = secrets.get(p.api_key_env) or config.get_secret(p.api_key_env) or ""
        io.say(f"  {'primary' if i == 0 else 'fallback':<9}{n:<12}{p.model[:34]:<36}{_ctx(p.context_tokens):<12}{'yes' if p.tools else 'no':<7}"
               + (config.mask(key) if key else "none"))
    io.say(f"  mode: {cfg.mode}    search: {cfg.search.get('provider', 'ddg')}")


def _existing() -> Config | None:
    try:
        return config.load()
    except config.NotConfigured:
        return None


def _run(io: SetupIO, first_run: bool, section: str | None) -> bool:
    cfg = _existing()
    if first_run:
        io.say("Welcome to Jarvoos. Nothing is saved until you confirm at the end (Ctrl+C cancels).")
    if cfg is None and section in ("model", "fallback"):
        section = None
    if cfg is None and section in ("search", "mode"):
        io.say("Add a model first: run `jarvoos setup`.")
        return False
    secrets: dict[str, str] = {}
    if section is None:
        cfg = cfg or Config({}, [])
        entries = _wizard(io, [], True)
        if entries is None:
            return False
        cfg.providers = {e.spec.name: e.spec for e in entries}
        cfg.chain = list(cfg.providers)
        for e in entries:
            secrets.update(e.secrets)
        cfg.mode = _ask_mode(io, cfg.mode)
    elif section == "model":
        others = cfg.chain[1:]
        entries = _wizard(io, list(others), False)
        if entries is None:
            return False
        new = entries[0]
        cfg.providers = {new.spec.name: new.spec, **{n: cfg.providers[n] for n in others}}
        cfg.chain = [new.spec.name, *others]
        secrets.update(new.secrets)
    elif section == "fallback":
        primary = cfg.chain[0]
        if io.choose("Fallback", ["Add or replace the fallback", "Remove the fallback"]) == 1:
            cfg.providers, cfg.chain = {primary: cfg.providers[primary]}, [primary]
        else:
            taken = [primary]
            io.say("Looking for local servers...")
            e = _backend(io, detect.scan(), taken)
            if e is None or not _test(io, e):
                io.say("No fallback added, nothing saved.")
                return False
            cfg.providers, cfg.chain = {primary: cfg.providers[primary], e.spec.name: e.spec}, [primary, e.spec.name]
            secrets.update(e.secrets)
    elif section == "search":
        _setup_search(io, cfg, secrets)
    else:
        cfg.mode = _ask_mode(io, cfg.mode)
    _summary(io, cfg, secrets)
    if not io.confirm("Save this configuration?", True):
        io.say("Nothing saved.")
        return False
    config.save(cfg, secrets)
    io.say(f"Saved {config.CONFIG_PATH}" + (f" and keys in {config.SECRETS_PATH} (mode 600)" if secrets else "") + ".")
    try:
        for line in doctor.rows(doctor.run_checks(config.load(), lite=True)):
            io.say(line)
    except Exception as e:
        io.say(f"Saved, but the final check failed: {e}", "yellow")
    io.say("Change anything later with `jarvoos setup model|fallback|search|mode`, check health with `jarvoos doctor`.")
    return True
