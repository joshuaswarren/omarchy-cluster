"""Heartbeat hub: agents POST 1 Hz heartbeats; hub answers node liveness."""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HUB_PORT = 8030
UP_AFTER_S = 3.0  # one heartbeat interval with jitter grace


class Registry:
    def __init__(self):
        self.last = {}  # name -> (monotonic, seq)
        self._lock = threading.Lock()

    def beat(self, name, seq):
        with self._lock:
            self.last[name] = (time.monotonic(), seq)

    def nodes(self):
        now = time.monotonic()
        with self._lock:
            return {name: {"age_s": round(now - t, 3), "seq": seq, "up": now - t < UP_AFTER_S}
                    for name, (t, seq) in self.last.items()}

    def up_names(self):
        return {n for n, v in self.nodes().items() if v["up"]}


class Handler(BaseHTTPRequestHandler):
    registry = None

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/nodes":
            return self._json(200, self.registry.nodes())
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/heartbeat":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            name = req.get("name") or self.headers.get("X-Node-Name")
            if not name:
                return self._json(400, {"error": "missing name"})
            self.registry.beat(name, int(req.get("seq", 0)))
            self._json(200, {"ok": True})
        except (ValueError, json.JSONDecodeError) as e:
            self._json(400, {"error": str(e)})


def serve(port=DEFAULT_HUB_PORT):
    Handler.registry = Registry()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print("omarchy-cluster hub listening on :%d" % port, flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster hub")
    ap.add_argument("--port", type=int, default=DEFAULT_HUB_PORT)
    args = ap.parse_args(argv)
    serve(port=args.port)


if __name__ == "__main__":
    main()
