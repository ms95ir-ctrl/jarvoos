"""Interactive chat loop: prompt_toolkit input, slash commands, Ctrl+C handling, persistence."""
from __future__ import annotations

import difflib
import json
import os
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config
from .sessions import SessionMeta, SessionStore
from .ui import UI, fmt_tokens, sanitize

TIPS = [
    "Alt+Enter (or Ctrl+J) adds a new line; Enter sends.",
    "/model openrouter switches provider for this session; /model auto goes back to the normal order.",
    "/undo drops the last exchange, /retry runs it again.",
    "/compress [focus] shrinks the conversation when the context bar turns yellow.",
    "Ctrl+C once stops the current answer and keeps what was said; twice quits.",
    "jarvoos -c continues your last session; /sessions lists older ones.",
    "/verbose shows full tool output instead of a short preview.",
    "/mode auto stops asking before write and exec tools; /mode approve restores it.",
    "Skills are slash commands too: type / and press Tab to see them.",
    "/doctor checks every provider with a real request.",
]
HINTS = {
    "alt_enter": "Alt+Enter inserts a new line when you want to write more than one line.",
    "model": "Use /model <name> to pin a provider, /model auto to let failover decide.",
    "approvals": "Approvals: o = once, s = for this session, a = always, d = deny, c = cancel the whole turn.",
}


@dataclass
class Command:
    name: str
    help: str
    handler: Callable[["ReplState", str], None]
    aliases: tuple = ()


@dataclass
class ReplState:
    cfg: Any
    agent: Any
    router: Any
    store: SessionStore
    ui: UI
    session: SessionMeta
    cancel: threading.Event = field(default_factory=threading.Event)
    last_ctrl_c: float = 0.0
    done: bool = False
    skills: list = field(default_factory=list)
    saved: int = 0
    first_saved: Any = None
    last_saved: Any = None
    pinned: str = ""
    turns_run: int = 0


def _seen_path() -> Path:
    return config.STATE_DIR / "seen.json"


def _load_seen() -> dict:
    try:
        data = json.loads(_seen_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_seen(seen: dict) -> None:
    try:
        _seen_path().parent.mkdir(parents=True, exist_ok=True)
        _seen_path().write_text(json.dumps(seen))
    except OSError:
        pass


def hint(st: ReplState, key: str) -> None:
    seen = _load_seen()
    if key not in seen.get("hints", []):
        st.ui.notice(f"tip: {HINTS[key]}", "dim")
        seen["hints"] = [*seen.get("hints", []), key]
        _save_seen(seen)


def next_tip() -> str:
    seen = _load_seen()
    i = seen.get("tip", random.randrange(len(TIPS))) % len(TIPS)
    _save_seen({**seen, "tip": i + 1})
    return TIPS[i]


def load_skills() -> list:
    try:
        from . import skills
        return list(skills.load_all())
    except Exception:
        return []


def _valid_len(messages: list[dict]) -> int:
    """Length of the longest prefix that doesn't end inside an unanswered tool call."""
    j = len(messages)
    while j and messages[j - 1].get("role") == "tool":
        j -= 1
    if j and messages[j - 1].get("role") == "assistant" and messages[j - 1].get("tool_calls"):
        answered = {m.get("tool_call_id") for m in messages[j:]}
        if not all(c.get("id") in answered for c in messages[j - 1]["tool_calls"]):
            return j - 1
    return len(messages)


def persist(st: ReplState) -> None:
    msgs = st.agent.messages
    n = _valid_len(msgs)
    intact = st.saved == 0 or (0 < st.saved <= n and msgs[0] is st.first_saved and msgs[st.saved - 1] is st.last_saved)
    try:
        if not intact:
            st.store.rewrite(st.session.id, msgs[:n])
        else:
            for m in msgs[st.saved:n]:
                st.store.append(st.session.id, m)
    except OSError as e:
        st.ui.notice(f"could not save session: {e}", "warn")
        return
    st.saved = n
    st.first_saved, st.last_saved = (msgs[0], msgs[n - 1]) if n else (None, None)


def resync(st: ReplState) -> None:
    """After undo/compact/clear the whole history is rewritten."""
    st.saved = -1
    st.first_saved = object()
    persist(st)


def provider_name(st: ReplState) -> str:
    return getattr(st.router, "active", "") or (st.cfg.chain[0] if st.cfg.chain else "")


# ---- turns ----------------------------------------------------------------------------------

def _force_exit(st: ReplState) -> None:
    try:
        persist(st)
    except Exception:
        pass
    _resume_hint(st)
    sys.stdout.flush()
    os._exit(130)


def _sigint_handler(st: ReplState):
    def handler(signum, frame):
        now = time.monotonic()
        if st.cancel.is_set() and now - st.last_ctrl_c < 2:
            _force_exit(st)
            return
        st.last_ctrl_c = now
        st.cancel.set()
        st.ui.notice("stopping... (Ctrl+C again to force quit)", "warn")
    return handler


def run_turn(st: ReplState, text: str) -> None:
    st.cancel.clear()
    st.ui.cancel = st.cancel
    st.ui.new_turn()
    try:
        old = signal.signal(signal.SIGINT, _sigint_handler(st))
    except ValueError:
        old = None
    res = None
    try:
        with st.ui.spinner("thinking"):
            res = st.agent.run(text, st.cancel)
    except KeyboardInterrupt:
        st.ui.notice("interrupted", "warn")
    except Exception as e:
        st.ui.error(f"Something went wrong: {type(e).__name__}", sanitize(str(e))[:300])
    finally:
        if old is not None:
            signal.signal(signal.SIGINT, old)
        st.ui.end_turn()
        persist(st)
        st.last_ctrl_c = 0.0
        st.ui.cancel = None
    st.turns_run += 1
    if res is not None:
        _report(st, res)
    if st.turns_run == 1:
        hint(st, "alt_enter")
    elif st.turns_run == 3:
        hint(st, "model")


def _report(st: ReplState, res) -> None:
    stopped = getattr(res, "stopped", None)
    if stopped in ("interrupted", "cancelled"):
        st.ui.notice("stopped; what was said so far is kept" if getattr(res, "text", "") else "stopped", "warn")
    elif stopped == "step_limit":
        st.ui.notice(f"stopped after {getattr(res, 'steps', st.cfg.max_steps)} steps (max_steps); "
                     "say 'continue' to go on", "warn")
    elif stopped == "error" and getattr(res, "error", None) is not None:
        e = res.error
        try:
            from .errors import friendly
            msg, details = friendly(e, st.cfg.providers.get(e.provider) or st.cfg.providers[st.cfg.chain[0]])
        except Exception:
            msg, details = str(e), ""
        st.ui.error(msg, details)
    elif stopped is None:
        st.ui.render_answer(getattr(res, "text", ""))


# ---- commands -------------------------------------------------------------------------------

def say(st: ReplState, text: str, style: str = "") -> None:
    st.ui._p(text, style)


def cmd_help(st, arg):
    width = max(len(c.name) for c in COMMANDS) + 3
    for c in COMMANDS:
        say(st, f"/{c.name:<{width}}{c.help}" + (f"  (also {', '.join('/' + a for a in c.aliases)})" if c.aliases else ""))
    if st.skills:
        say(st, "skills: " + "  ".join(f"/{s.name}" for s in st.skills), "dim")
    say(st, "Enter sends, Alt+Enter or Ctrl+J adds a line, Ctrl+C stops, Ctrl+D quits.", "dim")


def cmd_exit(st, arg):
    st.done = True


def new_session(st: ReplState) -> None:
    drop_if_empty(st)
    st.agent.messages.clear()
    st.session = st.store.new(Path.cwd(), st.pinned or st.cfg.chain[0], _model_of(st, st.pinned or st.cfg.chain[0]))
    st.saved, st.first_saved, st.last_saved = 0, None, None


def cmd_new(st, arg):
    new_session(st)
    say(st, f"new session {st.session.id}", "dim")


def cmd_clear(st, arg):
    st.ui.console.clear()


def cmd_history(st, arg):
    if not st.agent.messages:
        return say(st, "empty", "dim")
    for i, m in enumerate(st.agent.messages, 1):
        body = m.get("content") or ""
        if m.get("tool_calls"):
            body += " [calls: " + ", ".join(c["function"]["name"] for c in m["tool_calls"]) + "]"
        say(st, f"{i:>3} {m.get('role', '?'):<9}{' '.join(str(body).split())[:100]}", "dim" if m.get("role") == "tool" else "")


def cmd_retry(st, arg):
    text = st.agent.retry_text()
    if not text:
        return say(st, "nothing to retry", "dim")
    st.agent.undo()
    resync(st)
    run_turn(st, text)


def cmd_undo(st, arg):
    say(st, "undone" if st.agent.undo() else "nothing to undo", "dim")
    resync(st)


def cmd_title(st, arg):
    if not arg.strip():
        return say(st, st.session.title or "(untitled)")
    st.store.set_title(st.session.id, arg)
    st.session = st.store.load(st.session.id)[0]
    say(st, f"title: {st.session.title}", "dim")


def cmd_sessions(st, arg):
    rows = st.store.list(10)
    for i, m in enumerate(rows, 1):
        mark = "*" if m.id == st.session.id else " "
        say(st, f"{mark}{i:>2} {m.id}  {m.turns:>3} turns  {m.title or '(untitled)'}")
    say(st, "/resume <number|id> to continue one" if rows else "no saved sessions", "dim")


def restore(st: ReplState, sid: str) -> None:
    meta, msgs = st.store.load(sid)
    msgs = msgs[:_valid_len(msgs)]
    drop_if_empty(st)
    st.agent.messages[:] = msgs
    st.session = meta
    st.saved, st.first_saved, st.last_saved = len(msgs), (msgs[0] if msgs else None), (msgs[-1] if msgs else None)
    if meta.provider and meta.provider in st.cfg.providers and meta.provider != st.cfg.chain[0]:
        st.router.pin(meta.provider)
        st.pinned = meta.provider
    else:
        st.router.pin(None)
        st.pinned = ""
    say(st, f"resumed {meta.id}: {meta.title or '(untitled)'} ({len(msgs)} messages)", "dim")
    last = next((m for m in reversed(msgs) if m.get("role") == "assistant" and m.get("content")), None)
    if last:
        say(st, "last answer: " + " ".join(str(last["content"]).split())[:200], "dim")


def cmd_resume(st, arg):
    arg = arg.strip()
    rows = st.store.list(10)
    if not arg:
        return cmd_sessions(st, "")
    sid = rows[int(arg) - 1].id if arg.isdigit() and 0 < int(arg) <= len(rows) else st.store.resolve(arg)
    if not sid:
        return say(st, f"no session matches {arg!r}; /sessions lists them", "red")
    restore(st, sid)


def cmd_compress(st, arg):
    with st.ui.spinner("compacting"):
        st.agent.compact(arg.strip() or None)
    resync(st)
    s = st.agent.usage_snapshot()
    say(st, f"context now {ctx_pct(s):.0f}% ({fmt_tokens(s.prompt_tokens)} tokens)", "dim")


def _model_of(st, name: str) -> str:
    spec = st.cfg.providers.get(name)
    return spec.model if spec else ""


def cmd_model(st, arg):
    arg = arg.strip()
    if not arg:
        for n in st.cfg.chain:
            s = st.cfg.providers[n]
            flag = "pinned" if n == st.pinned else ("active" if n == provider_name(st) else "")
            say(st, f"{'*' if flag else ' '} {n}: {s.model}  {flag}")
        return say(st, "/model <name> pins one, /model auto restores failover order", "dim")
    if arg == "auto":
        st.router.pin(None)
        st.router.reset_cooldowns()
        st.pinned = ""
        name = st.cfg.chain[0]
    elif arg in st.cfg.providers:
        st.router.pin(arg)
        st.pinned = name = arg
    else:
        close = difflib.get_close_matches(arg, st.cfg.providers, n=1)
        return say(st, f"unknown provider {arg!r}" + (f"; did you mean {close[0]}?" if close else f"; have: {', '.join(st.cfg.providers)}"), "red")
    st.store.update(st.session.id, provider=name, model=_model_of(st, name))
    say(st, f"provider: {name if arg != 'auto' else 'auto (' + name + ' first)'}", "dim")


def cmd_mode(st, arg):
    arg = arg.strip()
    if not arg:
        return say(st, f"mode: {st.cfg.mode}  (approve | auto | chat)")
    if arg not in ("approve", "auto", "chat"):
        return say(st, "mode must be approve, auto or chat", "red")
    st.agent.set_mode(arg)
    st.cfg.mode = arg
    say(st, f"mode: {arg}", "dim")


def ctx_pct(s) -> float:
    ctx = getattr(s, "context_tokens", 0)
    return 100 * s.prompt_tokens / ctx if ctx else float(getattr(s, "pct", 0))


def cmd_status(st, arg):
    s = st.agent.usage_snapshot()
    name = provider_name(st)
    say(st, f"provider {name}:{_model_of(st, name)}{' (pinned)' if st.pinned else ''}"
            f"{'  FALLBACK in use' if getattr(st.router, 'failed_over', False) else ''}")
    say(st, f"context {ctx_pct(s):.0f}% of {fmt_tokens(getattr(s, 'context_tokens', 0))}; "
            f"tokens in {s.prompt_tokens} out {s.completion_tokens}")
    say(st, f"session {st.session.id}  mode {st.cfg.mode}  cwd {st.session.cwd}")


def cmd_tools(st, arg):
    from . import tools
    enabled = {t["function"]["name"] for t in tools.tool_schemas(st.cfg.mode, st.cfg.providers[st.cfg.chain[0]].tools)}
    for n, t in tools.TOOLS.items():
        say(st, f"{n:<16}{t.risk:<7}{'' if n in enabled else 'off in this mode'}")


def cmd_skills(st, arg):
    for s in st.skills:
        say(st, f"/{s.name:<20}{s.status:<14}{s.description}")
    if not st.skills:
        say(st, "no skills installed", "dim")


def cmd_reload(st, arg):
    st.skills = load_skills()
    say(st, f"reloaded {len(st.skills)} skills (provider changes need a restart)", "dim")


def cmd_config(st, arg):
    say(st, f"config: {config.CONFIG_PATH}")
    for n in st.cfg.chain:
        s = st.cfg.providers[n]
        key = config.get_secret(s.api_key_env) if s.api_key_env else None
        say(st, f"  {n}: {s.kind} {s.model} @ {s.base_url}  ctx {s.context_tokens or 'default'}  "
                f"tools {'on' if s.tools else 'off'}" + (f"  key {config.mask(key)}" if key else ""))
    say(st, f"  mode {st.cfg.mode}, max_steps {st.cfg.max_steps}, compact_at {st.cfg.compact_at}", "dim")
    say(st, "change with `jarvoos setup` or `jarvoos config set KEY VALUE`", "dim")


def cmd_doctor(st, arg):
    from . import doctor
    with st.ui.spinner("checking providers"):
        checks = doctor.run_checks(st.cfg)
    doctor.render(checks)


def cmd_verbose(st, arg):
    st.ui.verbose = not st.ui.verbose
    say(st, f"verbose {'on' if st.ui.verbose else 'off'}", "dim")


COMMANDS: list[Command] = [
    Command("help", "list commands", cmd_help, ("?",)),
    Command("exit", "quit", cmd_exit, ("quit", "q")),
    Command("new", "start a fresh session", cmd_new),
    Command("clear", "clear the screen", cmd_clear),
    Command("history", "show the conversation so far", cmd_history),
    Command("retry", "undo the last exchange and run it again", cmd_retry),
    Command("undo", "drop the last exchange", cmd_undo),
    Command("title", "show or set the session title", cmd_title),
    Command("sessions", "list recent sessions", cmd_sessions),
    Command("resume", "continue a saved session: /resume <number|id>", cmd_resume),
    Command("compress", "shrink the conversation: /compress [focus]", cmd_compress, ("compact",)),
    Command("model", "show providers, /model <name> pins one, /model auto", cmd_model),
    Command("mode", "show or set permission mode (approve|auto|chat)", cmd_mode),
    Command("status", "provider, context and token usage", cmd_status),
    Command("tools", "list tools and their risk level", cmd_tools),
    Command("skills", "list skills", cmd_skills),
    Command("reload", "reload skills", cmd_reload),
    Command("config", "show the active configuration", cmd_config),
    Command("doctor", "check every provider with a real request", cmd_doctor),
    Command("verbose", "toggle full tool output", cmd_verbose),
]


def find_command(name: str) -> Command | None:
    return next((c for c in COMMANDS if name == c.name or name in c.aliases), None)


def all_names(st: ReplState | None = None) -> list[str]:
    names = [n for c in COMMANDS for n in (c.name, *c.aliases)]
    return names + [s.name for s in (st.skills if st else [])]


def suggest(name: str, st: ReplState | None = None) -> str | None:
    hit = difflib.get_close_matches(name, all_names(st), n=1, cutoff=0.5)
    return hit[0] if hit else None


def handle_line(st: ReplState, line: str) -> None:
    line = line.strip()
    if not line:
        return
    name, _, arg = line[1:].partition(" ") if line.startswith("/") else ("", "", "")
    if not line.startswith("/") or "/" in name or not name:  # plain text, or a path like /etc/hosts
        return run_turn(st, line)
    cmd = find_command(name)
    if cmd:
        try:
            cmd.handler(st, arg.strip())
        except KeyboardInterrupt:
            say(st, "interrupted", "yellow")
        except Exception as e:
            st.ui.error(f"/{name} failed: {type(e).__name__}", sanitize(str(e))[:300])
        return
    skill = next((s for s in st.skills if s.name == name), None)
    if skill:
        task = arg.strip() or "Follow it for the current task."
        return run_turn(st, f"Use the '{skill.name}' skill.\n\n{skill.body}\n\n{task}")
    close = suggest(name, st)
    say(st, f"unknown command /{name}" + (f"; did you mean /{close}?" if close else "; /help lists commands"), "red")


# ---- prompt ---------------------------------------------------------------------------------

def toolbar(st: ReplState):
    try:
        s = st.agent.usage_snapshot()
        name = provider_name(st)
        pct = ctx_pct(s)
        color = "ansired" if pct > 80 else "ansiyellow" if pct >= 50 else "ansigreen"
        parts = [("", f" {name}:{_model_of(st, name)} "), (f"fg:{color}", f" ctx {pct:.0f}% "),
                 ("", f" {fmt_tokens(s.prompt_tokens + s.completion_tokens)} tokens ")]
        if getattr(st.router, "failed_over", False):
            parts.insert(0, ("bold bg:ansired fg:ansiwhite", f" FALLBACK: {name} "))
        return parts
    except Exception:
        return ""


def make_reader(st: ReplState) -> Callable[[], str]:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    class Slash(Completer):
        def get_completions(self, document, complete_event):
            t = document.text_before_cursor
            if not t.startswith("/") or " " in t or "\n" in t:
                return
            helps = {n: c.help for c in COMMANDS for n in (c.name, *c.aliases)}
            helps.update({s.name: s.description for s in st.skills})
            for n in sorted(helps):
                if ("/" + n).startswith(t):
                    yield Completion("/" + n, start_position=-len(t), display_meta=helps[n])

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(config.STATE_DIR / "history")), multiline=True,
                            key_bindings=kb, completer=Slash(), complete_while_typing=False,
                            bottom_toolbar=lambda: toolbar(st), prompt_continuation=lambda w, n, s: "  ")
    return lambda: session.prompt("> ")


def loop(st: ReplState, read: Callable[[], str]) -> None:
    while not st.done:
        try:
            line = read()
        except EOFError:
            return
        except KeyboardInterrupt:
            now = time.monotonic()
            if st.last_ctrl_c and now - st.last_ctrl_c < 2:
                return
            st.last_ctrl_c = now
            say(st, "press Ctrl+C again within 2s (or Ctrl+D) to exit", "dim")
            continue
        st.last_ctrl_c = 0.0
        handle_line(st, line)


def drop_if_empty(st: ReplState) -> None:
    if not st.agent.messages:
        st.store.delete(st.session.id)


def _resume_hint(st: ReplState) -> None:
    if st.agent.messages:
        sys.stdout.write("\nResume with: jarvoos -c\n")


def repl(cfg, resume: str | None = None, continue_last: bool = False, auto_approve: bool = False,
         debug: Callable[[str, dict], None] | None = None) -> int:
    from .agent import Agent
    from .providers import Router

    ui = UI(cfg)
    store = SessionStore()
    store.prune(50)
    sid = None
    if resume:
        sid = store.resolve(resume)
        if not sid:
            ui.error(f"No session matches {resume!r}.", "jarvoos sessions lists saved ones")
            return 2
    elif continue_last:
        sid = store.latest(Path.cwd()) or store.latest()
        if not sid:
            ui.notice("no earlier session; starting a new one", "dim")
    if auto_approve and cfg.mode != "chat":
        cfg.mode = "auto"
    router = Router(cfg, debug) if debug else Router(cfg)
    primary = cfg.chain[0]
    meta = store.new(Path.cwd(), primary, cfg.providers[primary].model)
    st = ReplState(cfg, None, router, store, ui, meta)
    events = ui.events()
    inner_confirm, inner_end, inner_text = events.confirm, events.on_tool_end, events.on_text

    def confirm(req):
        hint(st, "approvals")
        return inner_confirm(req)

    def tool_end(name, result, secs):
        inner_end(name, result, secs)
        persist(st)

    def on_text(chunk):
        inner_text(chunk)
        if len(st.agent.messages) != st.saved:
            persist(st)

    events.confirm, events.on_tool_end, events.on_text = confirm, tool_end, on_text
    st.agent = Agent(cfg, router, events, Path.cwd())
    st.skills = load_skills()
    if sid:
        store.delete(meta.id)
        restore(st, sid)
    ui.banner(st.session, next_tip(), len(st.skills))
    try:
        loop(st, make_reader(st))
    finally:
        drop_if_empty(st)
        _resume_hint(st)
    return 0
