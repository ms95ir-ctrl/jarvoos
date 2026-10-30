"""Optional Telegram notifications. Token and chat id live in secrets.env, like the provider keys."""
from __future__ import annotations

import time
from typing import Callable

import httpx

from .. import config

API = "https://api.telegram.org"


def creds() -> tuple[str | None, str | None]:
    return config.get_secret("JARVOOS_TELEGRAM_TOKEN"), config.get_secret("JARVOOS_TELEGRAM_CHAT")


def send(text: str, token: str | None = None, chat: str | None = None, client: httpx.Client | None = None) -> bool:
    t, c = (token, chat) if token else creds()
    if not (t and c):
        return False
    try:
        r = (client or httpx).post(f"{API}/bot{t}/sendMessage", data={"chat_id": c, "text": text[:3900], "disable_web_page_preview": "true"}, timeout=20)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def find_chat(token: str, wait_s: int = 90, client: httpx.Client | None = None, sleep=time.sleep) -> tuple[str | None, str]:
    """(chat id, bot username). The user has to send the bot any message first; Telegram only exposes the chat then."""
    cl = client or httpx.Client(timeout=20)
    me = cl.get(f"{API}/bot{token}/getMe")
    if me.status_code != 200:
        return None, ""
    name = me.json().get("result", {}).get("username", "")
    end = time.monotonic() + wait_s
    while True:
        r = cl.get(f"{API}/bot{token}/getUpdates")
        msgs = [u for u in r.json().get("result", []) if "message" in u] if r.status_code == 200 else []
        if msgs:
            return str(msgs[-1]["message"]["chat"]["id"]), name
        if time.monotonic() >= end:
            return None, name
        sleep(3)


def notify_hot(c, profile: dict, say: Callable[[str], None]) -> int:
    t, chat = creds()
    if not (t and chat):
        return 0
    n = 0
    for r in c.execute("select * from jobs where score>=? and notified=0 order by score desc limit 10", (profile.get("notify_score", 70),)):
        import json
        why = "; ".join(json.loads(r["reasons"] or "[]"))
        if send(f"{r['score']}  {r['title']}\n{r['company']} · {r['location']}\n{r['salary'] or 'salary ?'}\n{why}\n{r['url']}", t, chat):
            c.execute("update jobs set notified=1 where id=?", (r["id"],))
            n += 1
    c.commit()
    if n:
        say(f"telegram: {n} notification(s) sent")
    return n
