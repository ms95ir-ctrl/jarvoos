"""Conversation store: one JSONL file per session. Line 1 is the header, every other line is one message."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config


@dataclass
class SessionMeta:
    id: str
    title: str = ""
    created: str = ""
    updated: str = ""
    cwd: str = ""
    provider: str = ""
    model: str = ""
    turns: int = 0


_HEADER_KEYS = ("id", "title", "created", "cwd", "provider", "model")


def _dump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_atomic(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else config.STATE_DIR / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}.jsonl"

    def _read(self, path: Path) -> tuple[dict, list[dict]]:
        header: dict = {}
        messages: list[dict] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a crash can leave a truncated last line
                if not isinstance(obj, dict):
                    continue
                if i == 0 and "_session" in obj:
                    header = obj["_session"]
                else:
                    messages.append(obj)
        return header, messages

    def _meta(self, path: Path, header: dict, messages: list[dict]) -> SessionMeta:
        updated = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        turns = sum(1 for m in messages if m.get("role") == "user")
        return SessionMeta(**{k: header.get(k, "") for k in _HEADER_KEYS}, updated=updated, turns=turns)

    def _save(self, sid: str, header: dict, messages: list[dict]) -> None:
        _write_atomic(self._path(sid), [_dump({"_session": header}), *map(_dump, messages)])

    def new(self, cwd: Path, provider: str, model: str) -> SessionMeta:
        now = datetime.now()
        sid = now.strftime("%Y%m%d-%H%M%S")
        n = 1
        while self._path(sid).exists():
            n += 1
            sid = f"{now:%Y%m%d-%H%M%S}-{n}"
        header = {"id": sid, "title": "", "created": now.isoformat(timespec="seconds"),
                  "cwd": str(cwd), "provider": provider, "model": model}
        self._save(sid, header, [])
        return self.load(sid)[0]

    def append(self, sid: str, message: dict) -> None:
        path = self._path(sid)
        header, _ = self._read(path)
        with open(path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            if f.tell():
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")  # don't glue onto a truncated last line
            f.write(_dump(message).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        if message.get("role") == "user" and not header.get("title"):
            text = message.get("content")
            if isinstance(text, str) and text.strip():
                self.set_title(sid, text.strip().splitlines()[0])

    def rewrite(self, sid: str, messages: list[dict]) -> None:
        header, _ = self._read(self._path(sid))
        self._save(sid, header, messages)

    def load(self, sid: str) -> tuple[SessionMeta, list[dict]]:
        path = self._path(sid)
        if not path.exists():
            raise FileNotFoundError(sid)
        header, messages = self._read(path)
        header.setdefault("id", sid)
        return self._meta(path, header, messages), messages

    def update(self, sid: str, **fields: str) -> None:
        path = self._path(sid)
        header, messages = self._read(path)
        header.update({k: v for k, v in fields.items() if k in _HEADER_KEYS and k != "id"})
        self._save(sid, header, messages)

    def set_title(self, sid: str, title: str) -> None:
        title = " ".join(title.split())
        self.update(sid, title=title if len(title) <= 60 else title[:59].rstrip() + "…")

    def delete(self, sid: str) -> None:
        self._path(sid).unlink(missing_ok=True)

    def _files(self) -> list[Path]:
        return sorted(self.root.glob("*.jsonl"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)

    def list(self, limit: int = 10) -> list[SessionMeta]:
        out = []
        for p in self._files():
            if len(out) >= limit:
                break
            header, messages = self._read(p)
            header.setdefault("id", p.stem)
            if messages:
                out.append(self._meta(p, header, messages))
        return out

    def latest(self, cwd: Path | None = None) -> str | None:
        for p in self._files():
            header, messages = self._read(p)
            if messages and (cwd is None or header.get("cwd") == str(cwd)):
                return header.get("id", p.stem)
        return None

    def resolve(self, ref: str) -> str | None:
        """Exact id, or a unique id prefix."""
        if self._path(ref).exists():
            return ref
        hits = [p.stem for p in self._files() if p.stem.startswith(ref)]
        return hits[0] if len(hits) == 1 else None

    def prune(self, keep: int = 50) -> None:
        for p in self._files()[keep:]:
            p.unlink(missing_ok=True)
