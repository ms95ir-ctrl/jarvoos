"""The radar web page: a live server (status buttons work) and a static snapshot rewritten after every run."""
from __future__ import annotations

import http.server
import json
from pathlib import Path

from . import store

PAGE = Path(__file__).with_name("page.html")


def _json_for_script(data: dict) -> str:
    """JSON that is safe inside a <script> block even if a posting contains '</script>' or '<!--'."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def static_html(data: dict | None = None) -> str:
    html = PAGE.read_text(encoding="utf-8")
    return html.replace("<script>\nvar DATA=window.__DATA__", f"<script>window.__DATA__={_json_for_script(data or store.snapshot())};</script>\n<script>\nvar DATA=window.__DATA__", 1)


def write_static() -> Path:
    p = store.page_path()
    p.write_text(static_html(), encoding="utf-8")
    return p


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: str, ct: str = "application/json") -> None:
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ct + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/jobs"):
            self._send(200, json.dumps(store.snapshot(), ensure_ascii=False))
        else:
            self._send(200, PAGE.read_text(encoding="utf-8"), "text/html")

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._send(400, "{}")
        if self.path != "/api/status" or d.get("status") not in store.STATUSES or not isinstance(d.get("id"), str):
            return self._send(400, "{}")
        c = store.db()
        c.execute("update jobs set status=? where id=?", (d["status"], d["id"]))
        c.commit()
        c.close()
        self._send(200, "{}")


def serve(host: str = "127.0.0.1", port: int = 8800) -> None:
    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"Job radar page: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}   (Ctrl+C stops it)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
