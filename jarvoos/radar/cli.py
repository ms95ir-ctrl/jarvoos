"""`jarvoos radar ...` subcommands."""
from __future__ import annotations

import argparse
import sys

from . import cron, cv as cvmod, dashboard, engine, llm, store, telegram


def add_parser(sub) -> None:
    r = sub.add_parser("radar", help="job radar: search, score, web page, notifications, CV")
    rs = r.add_subparsers(dest="radar_cmd")
    rs.add_parser("setup", help="define what you search for, then schedule it")
    rs.add_parser("run", help="search, score and refresh the page now")
    s = rs.add_parser("serve", help="show the radar page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8800)
    rs.add_parser("status", help="profile, counts, schedule, Telegram")
    c = rs.add_parser("cron", help="automatic updates")
    c.add_argument("action", choices=["on", "off", "status"])
    c.add_argument("--every", type=int, default=6, metavar="HOURS")
    rs.add_parser("telegram", help="connect or test Telegram notifications")
    cv = rs.add_parser("cv", help="your CV: facts, review, build, tailor")
    cv.add_argument("action", nargs="?", choices=["facts", "show", "review", "build", "tailor"], default="show")
    cv.add_argument("job", nargs="?", help="job id (or unique prefix) for tailor")


def _io():
    from ..setup import RichIO
    return RichIO()


def status() -> int:
    p = store.load_profile()
    print(f"folder:   {store.root()}")
    print(f"profile:  {', '.join(p['target_roles']) if p and p['target_roles'] else 'none (run: jarvoos radar setup)'}")
    c = store.db()
    n, scored = (c.execute("select count(*), count(score) from jobs").fetchone())
    hot = c.execute("select count(*) from jobs where score>=?", ((p or {}).get("notify_score", 70),)).fetchone()[0]
    print(f"jobs:     {n} stored, {scored} scored, {hot} strong matches")
    print(f"cv facts: {'yes' if store.load_factbank() else 'no (run: jarvoos radar cv facts)'}")
    print(f"telegram: {'connected' if all(telegram.creds()) else 'not connected (run: jarvoos radar telegram)'}")
    print(f"cron:     {(cron.status() or 'off').splitlines()[1] if cron.available() and cron.status() else 'off'}")
    print(f"page:     {store.page_path()}")
    return 0


def _find_job(prefix: str):
    c = store.db()
    rows = c.execute("select * from jobs where id like ?", (prefix + "%",)).fetchall()
    return rows[0] if len(rows) == 1 else None


def cmd_cv(action: str, job: str | None) -> int:
    io = _io()
    cv = cvmod.load()
    if action == "facts":
        cvmod.save(cvmod.interview(io, cv))
        print(f"Saved {store.cv_path()}")
        return 0
    if cv is None:
        print("No CV facts yet. Run: jarvoos radar cv facts", file=sys.stderr)
        return 1
    if action == "show":
        print(store.load_factbank())
        return 0
    try:
        if action == "review":
            print(cvmod.review(cv))
            return 0
        if action == "build":
            html, pdf = cvmod.build(cv)
            print(f"HTML: {html}")
            print(f"PDF:  {pdf}" if pdf else "No PDF tool found (chromium, weasyprint or wkhtmltopdf). Open the HTML and print to PDF.")
            return 0
        if not job:
            print("Give a job id from the page or `jarvoos radar status`: jarvoos radar cv tailor <id>", file=sys.stderr)
            return 2
        row = _find_job(job)
        if row is None:
            print(f"No single job matches id {job!r}.", file=sys.stderr)
            return 1
        html, pdf = cvmod.build(cvmod.tailor(cv, dict(row)), f"cv-{row['id']}")
        print(f"Tailored for: {row['title']} at {row['company']}\nHTML: {html}" + (f"\nPDF:  {pdf}" if pdf else ""))
        return 0
    except (llm.LLMUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def main(args: argparse.Namespace) -> int:
    cmd = args.radar_cmd
    try:
        if cmd == "setup":
            from . import setup
            return setup.run()
        if cmd == "run":
            s = engine.run()
            return 1 if s["errors"] and not s["found"] else 0
        if cmd == "serve":
            dashboard.serve(args.host, args.port)
            return 0
        if cmd == "status":
            return status()
        if cmd == "cron":
            if not cron.available():
                print("cron/flock not available on this machine.", file=sys.stderr)
                return 1
            if args.action == "on":
                cron.enable(args.every)
                print(f"Radar will update every {args.every} h. Log: {store.log_path()}")
            elif args.action == "off":
                cron.disable()
                print("Automatic updates removed.")
            else:
                print(cron.status() or "off")
            return 0
        if cmd == "telegram":
            from . import setup
            return 0 if setup.telegram_setup(_io()) else 1
        if cmd == "cv":
            return cmd_cv(args.action, args.job)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130
    print("usage: jarvoos radar {setup,run,serve,status,cron,telegram,cv}   (start with: jarvoos radar setup)")
    return 0
