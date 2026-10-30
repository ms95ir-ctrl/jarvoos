"""Command line entry point: argument parsing and wiring only."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import __version__, config

EXIT_APPROVAL = 3


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="jarvoos", description="Terminal agent: local model first, cloud fallback.")
    ap.add_argument("--version", action="version", version=f"jarvoos {__version__}")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true", help="continue the last session")
    ap.add_argument("--resume", metavar="ID", help="resume a session by id (or unique prefix)")
    ap.add_argument("-z", "--prompt", metavar="PROMPT", help="run one prompt non-interactively and print the answer")
    ap.add_argument("--auto-approve", action="store_true", help="run write/exec tools without asking (dangerous)")
    ap.add_argument("--debug", action="store_true", help="log requests and responses (keys masked) to the state dir")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("setup", help="guided setup")
    s.add_argument("section", nargs="?", choices=["model", "fallback", "search", "mode"])
    d = sub.add_parser("doctor", help="check configuration and providers")
    d.add_argument("--fix", action="store_true")
    c = sub.add_parser("config", help="get | set | path | edit")
    c.add_argument("action", nargs="?", choices=["get", "set", "path", "edit"], default="path")
    c.add_argument("args", nargs="*")
    sub.add_parser("sessions", help="list saved sessions")
    sub.add_parser("update", help="update jarvoos")
    sub.add_parser("skills", help="list bundled and user skills with their test status")
    from .radar import cli as radar_cli
    radar_cli.add_parser(sub)
    return ap


# ---- debug log ------------------------------------------------------------------------------

def _masker():
    secrets = []
    try:
        cfg = config.load()
        secrets = [v for p in cfg.providers.values() if p.api_key_env and (v := config.get_secret(p.api_key_env)) and len(v) >= 6]
    except Exception:
        pass
    patterns = [re.compile(r"(?i)(bearer\s+)[^\s'\"]+"), re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")]

    def mask(text: str) -> str:
        for sec in secrets:
            text = text.replace(sec, config.mask(sec))
        text = patterns[0].sub(lambda m: m.group(1) + "***", text)
        return patterns[1].sub(lambda m: config.mask(m.group(0)), text)
    return mask


def enable_debug():
    """Returns (events callback for Router, log path). Also routes httpx/httpcore debug logging to the same file."""
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.STATE_DIR / f"debug-{time.strftime('%Y%m%d-%H%M%S')}.log"
    mask = _masker()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    f = os.fdopen(fd, "a", encoding="utf-8")
    lock = threading.Lock()

    def write(line: str) -> None:
        with lock:
            f.write(mask(line) + "\n")
            f.flush()

    def events(kind: str, data: dict) -> None:
        write(json.dumps({"t": round(time.time(), 3), "kind": kind, "data": data}, default=str, ensure_ascii=False))

    class H(logging.Handler):
        def emit(self, record):
            write(f"{record.name}: {record.getMessage()}")

    for name in ("httpx", "httpcore"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(H())
    os.environ["JARVOOS_DEBUG"] = "1"
    return events, path


# ---- one shot -------------------------------------------------------------------------------

def one_shot(cfg, prompt: str, auto_approve: bool, debug=None) -> int:
    from .agent import Agent, AgentEvents
    from .providers import Router
    from .ui import ThinkFilter, sanitize

    if auto_approve and cfg.mode != "chat":
        cfg.mode = "auto"
    flt = ThinkFilter()
    needed: list = []
    err = lambda s: print(sanitize(s), file=sys.stderr, flush=True)

    def on_text(chunk: str) -> None:
        sys.stdout.write(sanitize(flt.feed(chunk)))
        sys.stdout.flush()

    def on_tool_start(name: str, args: dict) -> None:
        try:
            from .tools import summarize_call
            err(f"> {summarize_call(name, args)}")
        except Exception:
            err(f"> {name}")

    def on_router(ev) -> None:
        err(f"[{ev.kind}] {getattr(ev, 'from_', '')} -> {getattr(ev, 'to', '')}")

    def confirm(req) -> str:
        needed.append(req)
        return "deny"

    events = AgentEvents(on_text=on_text, on_reasoning=lambda s: None, on_tool_start=on_tool_start,
                         on_tool_end=lambda n, r, s: None, on_router=on_router, on_notice=err, confirm=confirm)
    router = Router(cfg, debug) if debug else Router(cfg)
    agent = Agent(cfg, router, events, Path.cwd())
    cancel = threading.Event()
    old = signal.signal(signal.SIGINT, lambda *_: cancel.set())
    try:
        res = agent.run(prompt, cancel)
    except KeyboardInterrupt:
        err("\ninterrupted")
        return 130
    finally:
        signal.signal(signal.SIGINT, old)
    sys.stdout.write(sanitize(flt.flush()) + "\n")
    if needed:
        req = needed[0]
        err(f"approval required for {req.tool}: {req.summary}\n"
            "Not allowed non-interactively. Re-run with --auto-approve, or set mode to auto: jarvoos config set mode auto")
        return EXIT_APPROVAL
    stopped = getattr(res, "stopped", None)
    if stopped == "step_limit":
        err(f"stopped: step limit reached after {res.steps} steps (raise it: jarvoos config set max_steps N)")
        return 1
    if stopped in ("interrupted", "cancelled"):
        err("stopped: interrupted")
        return 130
    if stopped == "error":
        try:
            from .errors import friendly
            msg, details = friendly(res.error, cfg.providers.get(res.error.provider) or cfg.providers[cfg.chain[0]])
        except Exception:
            msg, details = str(res.error), ""
        err(f"error: {msg}" + (f"\nDetails: {details}" if details else ""))
        return 1
    return 0


# ---- subcommands ----------------------------------------------------------------------------

def cmd_config(action: str, args: list[str]) -> int:
    try:
        if action == "path":
            print(config.CONFIG_PATH)
        elif action == "get" and len(args) == 1:
            print(config.get_value(args[0]))
        elif action == "set" and len(args) >= 2:
            config.set_value(args[0], " ".join(args[1:]))
        elif action == "edit":
            if not config.CONFIG_PATH.exists():
                print("no config yet; run `jarvoos setup`", file=sys.stderr)
                return 1
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
            return subprocess.call([*shlex.split(editor), str(config.CONFIG_PATH)])
        else:
            print("usage: jarvoos config get KEY | set KEY VALUE | path | edit", file=sys.stderr)
            return 2
    except (KeyError, ValueError, config.NotConfigured) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_sessions() -> int:
    from .sessions import SessionStore
    rows = SessionStore().list(20)
    for m in rows:
        print(f"{m.id}  {m.updated[:16]}  {m.turns:>3} turns  {m.title or '(untitled)'}")
    if not rows:
        print("no saved sessions")
    return 0


def cmd_skills() -> int:
    from . import skills
    for s in skills.load_all():
        print(f"{s.name:<18}{s.status:<14}{s.description}")
    return 0


def cmd_update() -> int:
    src = Path(__file__).resolve().parents[1]
    if not (src / ".git").exists():
        print("not a git install; reinstall with the install script", file=sys.stderr)
        return 1
    try:
        subprocess.run(["git", "-C", str(src), "pull", "--ff-only"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(src)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"update failed: {e}", file=sys.stderr)
        return 1
    print("updated")
    return 0


def first_run_guard(non_interactive: bool) -> bool:
    """True when a usable config exists afterwards."""
    from . import setup
    if config.is_configured():
        return True
    if non_interactive or not _is_tty():
        print(setup.noninteractive_help(), file=sys.stderr)
        return False
    try:
        answer = input("Jarvoos isn't set up yet. Run setup now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer not in ("", "y", "yes"):
        print("Run `jarvoos setup` any time.", file=sys.stderr)
        return False
    try:
        return bool(setup.run(first_run=True)) and config.is_configured()
    except KeyboardInterrupt:
        print("\nSetup cancelled, nothing saved.", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "setup":
        from . import setup
        try:
            setup.run(section=args.section)
        except KeyboardInterrupt:
            print("\nSetup cancelled, nothing saved.", file=sys.stderr)
            return 130
        return 0
    if args.cmd == "doctor":
        from . import doctor
        return doctor.main(fix=args.fix)
    if args.cmd == "config":
        return cmd_config(args.action, args.args)
    if args.cmd == "sessions":
        return cmd_sessions()
    if args.cmd == "update":
        return cmd_update()
    if args.cmd == "skills":
        return cmd_skills()
    if args.cmd == "radar":
        from .radar import cli as radar_cli
        return radar_cli.main(args)

    if not first_run_guard(args.prompt is not None):
        return 1
    try:
        cfg = config.load()
    except config.NotConfigured as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    debug = None
    if args.debug:
        debug, path = enable_debug()
        print(f"debug log: {path}", file=sys.stderr)
    if args.prompt is not None:
        return one_shot(cfg, args.prompt, args.auto_approve, debug)
    from .repl import repl
    return repl(cfg, resume=args.resume, continue_last=args.cont, auto_approve=args.auto_approve, debug=debug)


if __name__ == "__main__":
    sys.exit(main())
