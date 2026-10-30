"""Free, keyless job sources. Each returns dicts: url, title, company, location, posted, description, salary, source."""
from __future__ import annotations

import re
import urllib.parse
from typing import Callable

import httpx

from .store import safe_url

UA = {"User-Agent": "Mozilla/5.0 (jarvoos job radar)"}


def _get(url: str, params: dict | None = None, headers: dict | None = None, client: httpx.Client | None = None):
    own = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        r = client.get(url, params=params, headers={**UA, **(headers or {})})
        r.raise_for_status()
        return r.json()
    finally:
        if own:
            client.close()


def _text(html: str, limit: int = 2500) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()[:limit]


def arbeitnow(profile: dict, get=_get) -> list[dict]:
    out = []
    for page in (1, 2, 3):
        for j in get("https://www.arbeitnow.com/api/job-board-api", {"page": page}).get("data", []):
            out.append({"url": j.get("url", ""), "title": j.get("title", ""), "company": j.get("company_name", ""),
                        "location": j.get("location", ""), "posted": "", "salary": "",
                        "description": _text(j.get("description", "")) + " " + " ".join(j.get("tags", [])), "source": "arbeitnow"})
    return out


def remotive(profile: dict, get=_get) -> list[dict]:
    out = []
    for term in (profile.get("target_roles") or [])[:6]:
        for j in get("https://remotive.com/api/remote-jobs", {"search": term, "limit": 30}).get("jobs", []):
            out.append({"url": j.get("url", ""), "title": j.get("title", ""), "company": j.get("company_name", ""),
                        "location": j.get("candidate_required_location", "") or "Remote", "posted": j.get("publication_date", ""),
                        "salary": j.get("salary", ""), "description": _text(j.get("description", "")), "source": "remotive"})
    return out


def remoteok(profile: dict, get=_get) -> list[dict]:
    out = []
    for j in get("https://remoteok.com/api")[1:]:
        if not isinstance(j, dict):
            continue
        sal = f"{j['salary_min']}-{j['salary_max']} USD" if j.get("salary_min") else ""
        out.append({"url": j.get("url", ""), "title": j.get("position", ""), "company": j.get("company", ""),
                    "location": j.get("location") or "Remote", "posted": j.get("date", ""), "salary": sal,
                    "description": _text(j.get("description", "")) + " " + " ".join(j.get("tags", [])), "source": "remoteok"})
    return out


def arbeitsagentur(profile: dict, get=_get) -> list[dict]:
    out = []
    for term in (profile.get("target_roles") or [])[:6]:
        d = get("https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs",
                {"was": term, "size": 50, "veroeffentlichtseit": 14}, {"X-API-Key": "jobboerse-jobsuche"})
        for j in d.get("stellenangebote", []):
            out.append({"url": "https://www.arbeitsagentur.de/jobsuche/jobdetail/" + j.get("refnr", ""), "title": j.get("titel", ""),
                        "company": j.get("arbeitgeber", ""), "location": (j.get("arbeitsort") or {}).get("ort", ""),
                        "posted": j.get("aktuelleVeroeffentlichungsdatum", ""), "salary": "", "description": j.get("beruf", ""),
                        "source": "arbeitsagentur"})
    return out


def _germany(profile: dict) -> bool:
    return bool(re.search(r"\b(germany|deutschland|german|berlin|munich|münchen|hamburg|köln|cologne|frankfurt|stuttgart|düsseldorf|leipzig|dresden|hannover|nuremberg|nürnberg|bremen|dortmund|essen)\b",
                          " ".join(profile.get("locations", [])), re.I))


def active_sources(profile: dict) -> dict[str, Callable]:
    """arbeitnow is a German board, so it only runs for German searches; the remote boards run when remote work is wanted.
    arbeitsagentur (Bundesagentur) is left out: its public API answered 403 to every request when this was tested (Oct 2026)."""
    s: dict[str, Callable] = {}
    if _germany(profile):
        s["arbeitnow"] = arbeitnow
    if profile.get("remote"):
        s["remotive"], s["remoteok"] = remotive, remoteok
    if not s:  # nothing matched: still give the user something to look at
        s["remotive"], s["remoteok"] = remotive, remoteok
    return s


_GENERIC = {"manager", "engineer", "senior", "junior", "lead", "specialist", "consultant", "officer", "assistant", "coordinator", "project", "projekt", "head", "director"}


def _words(terms: list[str]) -> list[str]:
    return [w.lower() for t in terms for w in re.findall(r"[\wäöüß+#.-]{3,}", t, re.I)]


def relevant(job: dict, profile: dict) -> bool:
    """Cheap pre-filter before spending model time. No avoided word in the title, and either a wanted role in the title
    (whole phrase or any of its words) or at least two distinct keywords in the text."""
    title = job.get("title", "").lower()
    if any(w.lower() in title for w in profile.get("avoid_words", [])):
        return False
    if any(r.lower() in title for r in profile.get("target_roles", [])):
        return True
    role_words = {w for w in _words(profile.get("target_roles", [])) if w not in _GENERIC}
    if any(w in title for w in role_words):
        return True
    hay = title + " " + (job.get("description") or "").lower()
    return len({k.lower() for k in profile.get("keywords", []) if k and k.lower() in hay}) >= 2


def host_ok(url: str) -> bool:
    return bool(safe_url(url)) and bool(urllib.parse.urlsplit(url).netloc)
