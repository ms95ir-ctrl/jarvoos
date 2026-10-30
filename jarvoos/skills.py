"""Skill files: markdown with `name` and `description` frontmatter. Only a one-line index goes in the
system prompt; the full text is loaded on demand so a small local model keeps a short context."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config

BUNDLED = Path(__file__).parent / "skills"


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source: str  # 'bundled' | 'user'
    path: Path | None = None


def _kv(lines) -> dict:
    meta = {}
    for line in lines:
        if ":" in line and not line.startswith((" ", "\t", "#")):
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta


def parse(text: str) -> tuple[dict, str]:
    """(frontmatter dict, body). Tolerates BOM, CRLF and a frontmatter block that is never closed."""
    lines = text.lstrip("﻿").replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, "\n".join(lines).strip()
    end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
    if end is None:  # no closing marker: header runs to the first blank line
        end = next((i for i, ln in enumerate(lines[1:], 1) if not ln.strip()), len(lines))
        return _kv(lines[1:end]), "\n".join(lines[end:]).strip()
    return _kv(lines[1:end]), "\n".join(lines[end + 1:]).strip()


def _load_dir(d: Path, source: str) -> list[Skill]:
    out = []
    for f in sorted(d.glob("*.md")) if d.is_dir() else []:
        try:
            meta, body = parse(f.read_text(errors="replace"))
        except OSError:
            continue
        desc = meta.get("description") or next((ln.strip() for ln in body.splitlines() if ln.strip()), "")[:120]
        out.append(Skill(meta.get("name") or f.stem, desc, body, source, f))
    return out


def load_all() -> list[Skill]:
    """Bundled skills, then the user's (same name overrides bundled)."""
    found: dict[str, Skill] = {}
    for s in _load_dir(BUNDLED, "bundled") + _load_dir(config.SKILLS_DIR, "user"):
        found[s.name] = s
    return list(found.values())


def get(name: str) -> Skill | None:
    return next((s for s in load_all() if s.name == name), None)


def index(items: list[Skill] | None = None) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in (load_all() if items is None else items))
