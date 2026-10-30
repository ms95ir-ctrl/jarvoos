"""Skill files: markdown with `name` and `description` frontmatter. Only a one-line index goes in the
system prompt; the full text is loaded on demand so a small local model keeps a short context."""
from __future__ import annotations

from pathlib import Path

from .config import SKILLS_DIR

BUNDLED = Path(__file__).parent / "skills"


def _parse(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    meta: dict = {}
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = body.strip()
    meta.setdefault("name", path.stem)
    return meta, text


class SkillStore:
    def __init__(self, dirs: list[Path] | None = None):
        self.skills: dict[str, tuple[str, str]] = {}  # name -> (description, body)
        for d in dirs if dirs is not None else [BUNDLED, SKILLS_DIR]:  # user skills override bundled ones
            for f in sorted(d.glob("*.md")) if d.is_dir() else []:
                meta, body = _parse(f)
                self.skills[meta["name"]] = (meta.get("description", ""), body)

    def index(self) -> str:
        return "\n".join(f"- {n}: {d}" for n, (d, _) in self.skills.items())

    def load(self, name: str) -> str | None:
        item = self.skills.get(name)
        return item[1] if item else None
