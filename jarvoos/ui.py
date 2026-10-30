"""Terminal rendering: streaming text, tool lines, notices, errors and the approval prompt."""
from __future__ import annotations

import contextlib
import json
import os
import re
import select
import shutil
import sys
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from . import __version__

_CTRL = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b.?|[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_DECISIONS = {"": "once", "o": "once", "y": "once", "once": "once", "yes": "once",
              "s": "session", "session": "session", "a": "always", "always": "always",
              "d": "deny", "n": "deny", "deny": "deny", "no": "deny", "c": "cancel", "cancel": "cancel"}


def sanitize(s: str) -> str:
    """Drop ANSI escapes and control characters (keeps newline and tab) so model or tool text can't drive the terminal."""
    return _CTRL.sub("", s.replace("\r\n", "\n").replace("\r", "\n"))


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


class ThinkFilter:
    """Removes <think>..</think> blocks from a stream; tags split across chunks are held back until resolved."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.inside, self.buf, self._strip = False, "", False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out: list[str] = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if not self.inside:
                    out.append(self.buf[:i])
                else:
                    self._strip = True
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            keep = next((k for k in range(min(len(tag) - 1, len(self.buf)), 0, -1) if tag.startswith(self.buf[-k:])), 0)
            emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            if not self.inside:
                out.append(emit)
            break
        text = "".join(out)
        if self._strip and text:
            text, self._strip = text.lstrip("\n"), False
        return text

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.inside, self.buf, self._strip = False, "", False
        return rest

    def reset(self) -> None:
        self.flush()


class UI:
    def __init__(self, cfg, console: Console | None = None, tty: bool | None = None):
        self.cfg = cfg
        self.tty = sys.stdout.isatty() if tty is None else tty
        self.console = console or Console(highlight=False, no_color=bool(os.environ.get("NO_COLOR")),
                                          force_terminal=None if tty is None else tty)
        self.verbose = bool(getattr(cfg, "verbose", False))
        opts = getattr(cfg, "ui", None)
        opts = opts if isinstance(opts, dict) else {}
        self.bell = bool(opts.get("bell") or os.environ.get("JARVOOS_BELL"))
        self.markdown = bool(opts.get("markdown") or os.environ.get("JARVOOS_MARKDOWN"))
        self.approval_timeout = float(os.environ.get("JARVOOS_APPROVAL_TIMEOUT", 120))
        self.cancel: threading.Event | None = None
        self.preview_lines = 4
        self._filter = ThinkFilter()
        self._answer: list[str] = []
        self._midline = False
        self._spin = None

    def _p(self, text: str, style: str = "", end: str = "\n") -> None:
        self.console.print(Text(sanitize(text), style=style), end=end, soft_wrap=True)

    def _write(self, text: str) -> None:
        self.console.file.write(text)
        self.console.file.flush()
        self._midline = not text.endswith("\n")

    def _newline(self) -> None:
        if self._midline:
            self._write("\n")

    @contextlib.contextmanager
    def spinner(self, label: str):
        self._spin = self.console.status(label, spinner="dots") if self.tty else None
        if self._spin:
            self._spin.start()
        try:
            yield self
        finally:
            self._stop_spin()

    def _stop_spin(self) -> None:
        if self._spin:
            self._spin.stop()
            self._spin = None

    def new_turn(self) -> None:
        self._filter = ThinkFilter()
        self._answer = []

    def end_turn(self) -> None:
        self._stop_spin()
        tail = self._filter.flush()
        if tail:
            self.text(tail, filtered=True)
        self._newline()
        if self.bell and self.tty:
            self._write("\a")

    def text(self, chunk: str, filtered: bool = False) -> None:
        self._stop_spin()
        visible = chunk if filtered else self._filter.feed(chunk)
        visible = sanitize(visible)
        if not visible:
            return
        if not self._answer and not self._midline:
            visible = visible.lstrip("\n")
        if visible:
            self._answer.append(visible)
            self._write(visible)

    def reasoning(self, chunk: str) -> None:
        self._stop_spin()

    def tool_start(self, name: str, args: dict) -> None:
        self._stop_spin()
        self._newline()
        try:
            from .tools import summarize_call
            line = summarize_call(name, args)
        except Exception:
            line = f"{name} {json.dumps(args, ensure_ascii=False)[:120]}"
        self._p(f"> {line}", "cyan")

    def tool_end(self, name: str, result: str, seconds: float) -> None:
        lines = sanitize(str(result)).splitlines() or [""]
        shown = lines if self.verbose else lines[:self.preview_lines]
        self._p(f"  {len(lines)} line{'s' * (len(lines) != 1)} in {seconds:.1f}s", "dim")
        for ln in shown:
            self._p("  | " + (ln if self.verbose or len(ln) <= 160 else ln[:159] + "…"), "dim")
        if len(lines) > len(shown):
            self._p(f"  | … {len(lines) - len(shown)} more (/verbose for full output)", "dim")
        self._midline = False

    def router_event(self, ev) -> None:
        self._stop_spin()
        self._newline()
        kind, to = getattr(ev, "kind", ""), getattr(ev, "to", "")
        spec = getattr(self.cfg, "providers", {}).get(to)
        target = f"{to} ({spec.model})" if spec else to
        if kind == "failover":
            self.notice(f"{getattr(ev, 'from_', '')} unreachable -> using {target} for this turn", "warn")
        elif kind == "restart":
            self.notice(f"--- restarting on {to} ---", "warn")
            self._filter = ThinkFilter()
            self._answer = []
        elif kind == "recovered":
            self.notice(f"{to} is back", "info")
        elif kind == "retry":
            self.notice(f"retrying {getattr(ev, 'from_', '') or to}...", "dim")

    def notice(self, msg: str, level: str = "info") -> None:
        self._stop_spin()
        self._newline()
        self._p(msg, {"warn": "yellow", "dim": "dim", "error": "red"}.get(level, "dim cyan"))

    def error(self, friendly: str, details: str = "") -> None:
        self._stop_spin()
        self._newline()
        self._p(friendly, "bold red")
        if details:
            self._p(f"Details: {details}", "dim")

    def banner(self, meta, tip: str, nskills: int) -> None:
        spec = getattr(self.cfg, "providers", {}).get(meta.provider) or next(iter(self.cfg.providers.values()), None)
        ctx = (spec.context_tokens if spec else 0) or 32000
        name = f"{spec.name}: {spec.model}" if spec else meta.provider
        self._p(" | ".join([f"jarvoos {__version__}", name, f"ctx {fmt_tokens(ctx)}", meta.cwd,
                            f"session {meta.id}", f"{nskills} skills", f"mode {getattr(self.cfg, 'mode', 'approve')}"]), "bold")
        self._p(f"tip: {tip}  (Alt+Enter = newline, /help)", "dim")

    def render_answer(self, text: str) -> None:
        if not (self.tty and self.markdown and text.strip()):
            return
        width, height = shutil.get_terminal_size()
        rows = sum(max(1, -(-len(ln) // width)) for ln in "".join(self._answer).split("\n"))
        if rows >= height - 2:
            return
        self._newline()
        self._write(f"\x1b[{rows}F\x1b[J")  # erase the raw stream, then draw it formatted
        self.console.print(Markdown(sanitize(text)))

    def _read_line(self, timeout: float) -> str | None:
        """One line from stdin; None on timeout or cancel."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel is not None and self.cancel.is_set():
                return None
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                line = sys.stdin.readline()
                if not line:
                    raise EOFError
                return line.strip().lower()
        return None

    def approval_prompt(self, req) -> str:
        if not (self.tty and sys.stdin.isatty()):
            return "deny"
        self._stop_spin()
        self._newline()
        self._p(f"Approval needed [{req.risk}]: {req.tool} - {req.summary}", "bold yellow")
        if req.reason:
            self._p(f"  why: {req.reason}", "yellow")
        for ln in sanitize(req.preview or "").splitlines()[:20]:
            self._p("  | " + ln[:200], "dim")
        for _ in range(3):
            self._p("  [o]nce  [s]ession  [a]lways  [d]eny  [c]ancel turn  (Enter = once): ", "yellow", end="")
            self.console.file.flush()
            try:
                answer = self._read_line(self.approval_timeout)
            except EOFError:
                return "deny"
            if answer is None:
                self._p("\n  no answer: denied", "red")
                return "cancel" if self.cancel is not None and self.cancel.is_set() else "deny"
            if answer in _DECISIONS:
                return _DECISIONS[answer]
            self._p("  please answer o, s, a, d or c", "red")
        return "deny"

    def events(self):
        from .agent import AgentEvents
        return AgentEvents(on_text=self.text, on_reasoning=self.reasoning, on_tool_start=self.tool_start,
                           on_tool_end=self.tool_end, on_router=self.router_event,
                           on_notice=lambda m: self.notice(m, "dim"), confirm=self.approval_prompt)
