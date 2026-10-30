"""Terminal front end (rich output, prompt_toolkit input)."""
from __future__ import annotations

import json
import sys

from rich.console import Console


class Terminal:
    def __init__(self, show_reasoning: bool = False, interactive: bool = True):
        self.console = Console(highlight=False)
        self.err = Console(stderr=True, highlight=False)
        self.show_reasoning = show_reasoning
        self.interactive = interactive
        self._in_reasoning = False

    def text(self, chunk: str) -> None:
        if self._in_reasoning:
            self.console.print()
            self._in_reasoning = False
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def reasoning(self, chunk: str) -> None:
        if self.show_reasoning:
            self._in_reasoning = True
            self.console.print(chunk, style="dim", end="")

    def notice(self, msg: str) -> None:
        self.err.print(f"\n[yellow]↻ {msg}[/yellow]")

    def tool(self, name: str, args: dict, risk: str) -> None:
        summary = json.dumps(args, ensure_ascii=False)
        self.err.print(f"\n[cyan]▸ {name}[/cyan] [dim]{summary[:160]}[/dim]")

    def confirm(self, name: str, args: dict, risk: str) -> str:
        if not self.interactive:
            return "no"
        detail = args.get("command") or args.get("path") or json.dumps(args)[:200]
        self.err.print(f"[bold yellow]Allow {name} ({risk})?[/bold yellow] {detail}")
        try:
            a = input("  [y]es / [n]o / [a]lways this session: ").strip().lower()
        except EOFError:
            return "no"
        return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(a, "no")
