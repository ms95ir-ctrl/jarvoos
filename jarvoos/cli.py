"""Command line entry point: `jarvoos` (chat), `jarvoos -z "..."` (one shot), setup, doctor, update."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import CONFIG_FILE, STATE_DIR, get_secret, load_config, mask
from .providers import AllProvidersFailed, Provider, Router
from .skills import SkillStore
from .tools import build_registry
from .ui import Terminal

HELP = """/help            this list
/clear           forget the conversation
/think           show or hide the model's reasoning
/model           show the active provider; /model primary|fallback|auto forces one
/events          recent failover events
/exit            quit (Ctrl+D also works)"""


def make_provider(section: dict, failover: dict) -> Provider | None:
    if not section.get("model"):
        return None
    key = get_secret(section.get("api_key_env", ""))
    headers = {"X-Title": "jarvoos"} if "openrouter" in section["base_url"] else {}
    return Provider(section["name"], section["base_url"], section["model"], key,
                    section.get("thinking", "default"), failover["connect_timeout"], headers)


def build(args, ui: Terminal):
    cfg = load_config()
    primary = make_provider(cfg["primary"], cfg["failover"])
    fallback = make_provider(cfg["fallback"], cfg["failover"])
    if not primary and not fallback:
        return None, cfg
    # a missing primary (or fallback) simply leaves the other one as the only provider
    router = Router(primary or fallback, fallback if primary else None, cfg["failover"])
    skills = SkillStore()
    agent = Agent(router, build_registry(cfg, skills), skills.index(), cfg, ui, auto_approve=args.auto_approve)
    return agent, cfg


def run_once(agent: Agent, text: str) -> int:
    try:
        agent.run(text)
        print()
        return 0
    except AllProvidersFailed as e:
        print(f"\nerror: every provider failed: {e}", file=sys.stderr)
        return 1


def repl(agent: Agent, ui: Terminal) -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(STATE_DIR / "history")))
    r = agent.router
    ui.console.print(f"[bold]jarvoos[/bold] {__version__}  primary: {r.primary.name} ({r.primary.model})"
                     + (f"  fallback: {r.fallback.name} ({r.fallback.model})" if r.fallback else "  [dim]no fallback[/dim]")
                     + "   [dim]/help for commands[/dim]")
    while True:
        try:
            line = session.prompt("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            if cmd in ("exit", "quit"):
                return 0
            elif cmd == "help":
                print(HELP)
            elif cmd == "clear":
                agent.reset()
                print("cleared")
            elif cmd == "think":
                ui.show_reasoning = not ui.show_reasoning
                print(f"reasoning display {'on' if ui.show_reasoning else 'off'}")
            elif cmd == "model":
                if rest in ("primary", "fallback") and (r.primary if rest == "primary" else r.fallback):
                    r.forced = r.primary if rest == "primary" else r.fallback
                elif rest == "auto":
                    r.forced = None
                print(f"active: {r.active}; forced: {r.forced.name if r.forced else 'no (auto failover)'}")
            elif cmd == "events":
                from .config import EVENTS_LOG
                print("".join(EVENTS_LOG.read_text().splitlines(True)[-10:]) if EVENTS_LOG.exists() else "none")
            else:
                print("unknown command; /help")
            continue
        try:
            agent.run(line)
            print("\n")
        except AllProvidersFailed as e:
            print(f"\nerror: every provider failed: {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(interrupted)\n")


def doctor() -> int:
    import httpx
    from rich.console import Console
    c = Console()
    cfg = load_config()
    c.print(f"config: {CONFIG_FILE} ({'found' if CONFIG_FILE.exists() else '[yellow]missing — run jarvoos setup[/yellow]'})")
    bad = 0
    for label in ("primary", "fallback"):
        s = cfg[label]
        if not s["model"]:
            c.print(f"{label}: [dim]not configured[/dim]")
            continue
        key = get_secret(s["api_key_env"])
        try:
            h = {"Authorization": f"Bearer {key}"} if key else {}
            r = httpx.get(s["base_url"].rstrip("/") + "/models", headers=h, timeout=8)
            ok = r.status_code == 200
            if label == "fallback" and ok and key:  # /models is public on OpenRouter; check the key itself
                ok = httpx.get(s["base_url"].rstrip("/") + "/auth/key", headers=h, timeout=8).status_code == 200
            c.print(f"{label}: {s['name']} {s['model']} @ {s['base_url']} — {'[green]ok[/green]' if ok else f'[red]HTTP {r.status_code}[/red]'}"
                    + (f"  key {mask(key)}" if s["api_key_env"] else ""))
            bad += not ok
        except Exception as e:
            c.print(f"{label}: {s['name']} @ {s['base_url']} — [red]{type(e).__name__}[/red]")
            bad += 1
    c.print(f"search: exa key {mask(get_secret('EXA_API_KEY'))}; searxng {cfg['search']['searxng_url'] or '(none)'}")
    return 1 if bad else 0


def update() -> int:
    src = Path(__file__).resolve().parents[1]
    if not (src / ".git").exists():
        print("not a git install; reinstall with the install script", file=sys.stderr)
        return 1
    subprocess.run(["git", "-C", str(src), "pull", "--ff-only"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(src)], check=True)
    print("updated")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="jarvoos", description="Terminal agent: local model first, cloud fallback.")
    ap.add_argument("command", nargs="?", choices=["chat", "setup", "doctor", "update", "version"], default="chat")
    ap.add_argument("section", nargs="?", choices=["primary", "fallback", "search"], help="with `setup`: only this part")
    ap.add_argument("-z", "--prompt", help="run one prompt non-interactively and print the answer")
    ap.add_argument("--auto-approve", action="store_true", help="run write/exec tools without asking (dangerous)")
    ap.add_argument("--show-reasoning", action="store_true")
    args = ap.parse_args()

    if args.command == "version":
        print(__version__)
        return
    if args.command == "setup":
        from . import setup
        try:
            setup.run(args.section)
        except KeyboardInterrupt:
            print("\naborted, nothing saved")
        return
    if args.command == "doctor":
        sys.exit(doctor())
    if args.command == "update":
        sys.exit(update())

    ui = Terminal(show_reasoning=args.show_reasoning, interactive=args.prompt is None and sys.stdin.isatty())
    agent, _ = build(args, ui)
    if agent is None:
        print("No model configured yet. Run: jarvoos setup", file=sys.stderr)
        sys.exit(2)
    if args.prompt is not None:
        sys.exit(run_once(agent, args.prompt))
    sys.exit(repl(agent, ui))
