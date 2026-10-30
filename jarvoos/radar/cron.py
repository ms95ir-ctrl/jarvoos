"""Keep the radar fresh with a crontab block of our own; nothing else in the crontab is touched."""
from __future__ import annotations

import shutil
import subprocess
import sys

from . import store

BEGIN, END = "# >>> jarvoos radar >>>", "# <<< jarvoos radar <<<"


def block(hours: int = 6, python: str | None = None) -> str:
    hours = max(1, min(24, int(hours)))
    py = python or sys.executable
    log = store.log_path()
    every = "0 * * * *" if hours == 1 else f"0 */{hours} * * *"
    return f"{BEGIN}\n{every} flock -n /tmp/jarvoos-radar.lock {py} -m jarvoos radar run >> {log} 2>&1\n{END}\n"


def strip(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines():
        if line.strip() == BEGIN:
            skip = True
        elif line.strip() == END:
            skip = False
        elif not skip:
            out.append(line)
    return "\n".join(out).strip("\n") + ("\n" if out else "")


def _read() -> str:
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _write(text: str) -> None:
    r = subprocess.run(["crontab", "-"], input=text, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "crontab failed")


def available() -> bool:
    return bool(shutil.which("crontab") and shutil.which("flock"))


def status() -> str | None:
    cur = _read()
    return block_of(cur)


def block_of(text: str) -> str | None:
    if BEGIN not in text:
        return None
    return text[text.index(BEGIN):text.index(END) + len(END)] if END in text else None


def enable(hours: int = 6) -> None:
    cur = strip(_read())
    _write((cur + "\n" if cur and not cur.endswith("\n\n") else cur) + block(hours))


def disable() -> None:
    _write(strip(_read()))
