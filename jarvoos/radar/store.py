"""Radar files and database. Everything lives in one folder (JARVOOS_RADAR or ~/.local/share/jarvoos/radar)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit

PROFILE_KEYS = ("summary", "target_roles", "keywords", "avoid_words", "locations", "remote", "min_salary", "currency",
                "languages", "notes", "notify_score")
STATUSES = ("new", "interested", "applied", "interview", "offer", "rejected", "dismissed")


def root() -> Path:
    p = Path(os.environ.get("JARVOOS_RADAR") or Path.home() / ".local" / "share" / "jarvoos" / "radar")
    p.mkdir(parents=True, exist_ok=True)
    return p


def profile_path() -> Path:
    return root() / "profile.json"


def cv_path() -> Path:
    return root() / "cv.json"


def factbank_path() -> Path:
    return root() / "factbank.md"


def page_path() -> Path:
    return root() / "index.html"


def log_path() -> Path:
    return root() / "radar.log"


def _list(v) -> list[str]:
    if isinstance(v, str):
        v = v.replace(";", ",").split(",")
    return [str(x).strip() for x in (v or []) if str(x).strip()]


def clean_profile(raw: dict) -> dict:
    """Coerce whatever the model or the user produced into the profile shape."""
    raw = raw if isinstance(raw, dict) else {}
    try:
        salary = int(float(str(raw.get("min_salary") or 0).replace(",", "").replace(".", "")[:12] or 0))
    except ValueError:
        salary = 0
    try:
        notify = max(0, min(100, int(raw.get("notify_score") or 70)))
    except (TypeError, ValueError):
        notify = 70
    return {"summary": str(raw.get("summary") or "").strip(), "target_roles": _list(raw.get("target_roles")),
            "keywords": _list(raw.get("keywords")), "avoid_words": _list(raw.get("avoid_words")),
            "locations": _list(raw.get("locations")), "remote": bool(raw.get("remote", False)), "min_salary": salary,
            "currency": str(raw.get("currency") or "EUR").strip()[:5].upper() or "EUR",
            "languages": _list(raw.get("languages")), "notes": str(raw.get("notes") or "").strip(),
            "notify_score": notify}


def load_profile() -> dict | None:
    try:
        return clean_profile(json.loads(profile_path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def save_profile(p: dict) -> None:
    profile_path().write_text(json.dumps(clean_profile(p), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_factbank() -> str:
    try:
        return factbank_path().read_text(encoding="utf-8")
    except OSError:
        return ""


def db() -> sqlite3.Connection:
    c = sqlite3.connect(root() / "jobs.db", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("""create table if not exists jobs(id text primary key, url text, title text, company text, location text,
        source text, found_at text, posted text, description text, salary text, score integer, reasons text, flags text,
        scored_by text, status text default 'new', notified integer default 0, notes text)""")
    c.execute("create table if not exists runs(ts text, source text, found integer, new integer, error text)")
    return c


def job_id(url: str, title: str = "", company: str = "") -> str:
    u = urlsplit(url or "")
    key = (u.netloc + u.path).lower().rstrip("/") if url else (title + company).lower()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def safe_url(url: str) -> str:
    """Only http(s) links ever reach the page; job posts are untrusted input."""
    return url if isinstance(url, str) and url.lower().startswith(("http://", "https://")) else ""


def add_job(c: sqlite3.Connection, j: dict) -> bool:
    url = safe_url(j.get("url", ""))
    title = (j.get("title") or "").strip()
    if not title or not url:
        return False
    i = job_id(url, title, j.get("company", ""))
    if c.execute("select 1 from jobs where id=?", (i,)).fetchone():
        return False
    dupe = (title + "|" + (j.get("company") or "") + "|" + (j.get("location") or "")).lower()
    for r in c.execute("select title, company, location from jobs where company=?", (j.get("company") or "",)):
        if (r["title"] + "|" + r["company"] + "|" + (r["location"] or "")).lower() == dupe:
            return False
    c.execute("insert into jobs(id,url,title,company,location,source,found_at,posted,description,salary) values(?,?,?,?,?,?,?,?,?,?)",
              (i, url, title[:200], (j.get("company") or "")[:120], (j.get("location") or "")[:120], j.get("source", ""),
               time.strftime("%Y-%m-%dT%H:%M:%S"), str(j.get("posted") or "")[:40], (j.get("description") or "")[:4000],
               (j.get("salary") or "")[:80]))
    c.commit()
    return True


def snapshot(c: sqlite3.Connection | None = None) -> dict:
    """Everything the page needs, as plain JSON-able data."""
    own = c is None
    c = c or db()
    jobs = []
    for r in c.execute("select * from jobs where score is not null order by score desc, found_at desc limit 1000"):
        d = dict(r)
        d["reasons"], d["flags"] = json.loads(d["reasons"] or "[]"), json.loads(d["flags"] or "[]")
        d["description"] = (d["description"] or "")[:600]
        jobs.append(d)
    runs = [dict(x) for x in c.execute("select * from runs order by ts desc limit 8")]
    pending = c.execute("select count(*) from jobs where score is null").fetchone()[0]
    if own:
        c.close()
    p = load_profile() or {}
    return {"jobs": jobs, "runs": runs, "pending": pending, "now": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "roles": p.get("target_roles", []), "notify_score": p.get("notify_score", 70)}
