"""Heartbeat hub: agents POST 1 Hz heartbeats; hub answers node liveness and draws the cluster."""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import probe

DEFAULT_HUB_PORT = 8030
UP_AFTER_S = 3.0  # one heartbeat interval with jitter grace

STR_FACTS = ("chip", "os", "gpu_backend")
NUM_FACTS = ("memory_total_bytes", "memory_free_bytes", "load1", "cpus")

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>omarchy-cluster</title>
<style>
  :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
  body { margin: 1rem; }
  svg { width: 100%; max-width: 960px; height: auto; }
  .card { fill: Canvas; stroke: currentColor; stroke-width: 1.5; }
  .out { stroke-dasharray: 5 4; opacity: .55; }
  .host { stroke-width: 3; }
  .edge { stroke: currentColor; stroke-width: 2; opacity: .6; }
  .bar { fill: currentColor; opacity: .55; }
  text { fill: currentColor; font-size: 13px; }
  .sub { opacity: .7; font-size: 11px; }
</style>
<h1>omarchy-cluster</h1>
<p id="msg">loading</p>
<svg id="g" viewBox="0 0 960 560"></svg>
<script>
const NS = "http://www.w3.org/2000/svg", W = 960, H = 560, CW = 230, CH = 104;
function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
const gb = (b) => (b / 2 ** 30).toFixed(1);
function place(names) {
  if (names.length === 1) return {[names[0]]: [W / 2, H / 2]};
  const pos = {}, rx = (W - CW) / 2, ry = (H - CH) / 2;
  names.forEach((n, i) => {
    const a = (2 * Math.PI * i) / names.length + Math.PI;
    pos[n] = [W / 2 + rx * Math.cos(a), H / 2 + ry * Math.sin(a)];
  });
  return pos;
}
function card(g, name, n, x, y, host) {
  const f = n.facts || {};
  const cls = "card" + (n.up ? "" : " out") + (host ? " host" : "");
  g.appendChild(el("rect", {class: cls, x: x - CW / 2, y: y - CH / 2, width: CW, height: CH, rx: 8}));
  g.appendChild(el("text", {x: x - CW / 2 + 10, y: y - CH / 2 + 20},
                   name + (host ? " (hub host)" : "") + (n.up ? "" : " - out")));
  g.appendChild(el("text", {class: "sub", x: x - CW / 2 + 10, y: y - CH / 2 + 38}, f.chip || "no facts yet"));
  if (f.memory_total_bytes) {
    const used = Math.max(0, f.memory_total_bytes - (f.memory_free_bytes || 0));
    const w = CW - 20, bx = x - CW / 2 + 10, by = y - CH / 2 + 48;
    g.appendChild(el("rect", {class: "card", x: bx, y: by, width: w, height: 10}));
    g.appendChild(el("rect", {class: "bar", x: bx, y: by, width: w * used / f.memory_total_bytes, height: 10}));
    g.appendChild(el("text", {class: "sub", x: bx, y: by + 26}, "memory " + gb(used) + " / " + gb(f.memory_total_bytes) + " GB"));
  }
  if (f.load1 !== undefined) {
    g.appendChild(el("text", {class: "sub", x: x - CW / 2 + 10, y: y + CH / 2 - 8},
                     "load " + f.load1.toFixed(2) + (f.cpus ? " on " + f.cpus + " cores" : "")));
  }
}
function draw(t) {
  const g = document.getElementById("g"), names = Object.keys(t.nodes).sort(), pos = place(names);
  g.replaceChildren();
  for (const e of t.links) {
    if (!pos[e.a] || !pos[e.b]) continue;
    const [x1, y1] = pos[e.a], [x2, y2] = pos[e.b];
    g.appendChild(el("line", {class: "edge", x1, y1, x2, y2}));
    const rate = [e.gbps_a2b, e.gbps_b2a].filter((v) => v).map((v) => v.toFixed(2)).join(" / ");
    g.appendChild(el("text", {x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 6, "text-anchor": "middle"},
                     e.media + (rate ? " " + rate + " Gb/s" : "")));
  }
  for (const n of names) card(g, n, t.nodes[n], pos[n][0], pos[n][1], n === t.host);
  document.getElementById("msg").textContent = names.length
    ? names.length + " node(s), " + t.links.length + " pinned link(s)"
    : "no heartbeats yet: start agents with --hub " + location.origin;
}
async function tick() {
  try {
    draw(await (await fetch("/v1/topology")).json());
  } catch (e) {
    document.getElementById("msg").textContent = "hub unreachable";
  }
}
tick();
setInterval(tick, 1000);
</script>
"""


def _count(v):
    """A number the page can draw and /v1/topology can serialize: not NaN, not infinite, not negative."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < 2 ** 63


def clean_facts(raw):
    """Keep only the facts the page draws, with bounded types and values."""
    if not isinstance(raw, dict):
        raise TypeError("heartbeat facts must be a JSON object")
    out = {k: raw[k][:120] for k in STR_FACTS if isinstance(raw.get(k), str)}
    out.update({k: raw[k] for k in NUM_FACTS if _count(raw.get(k))})
    return out


def topology_links(links):
    """Pinned route of every probed pair, as edges for the page."""
    edges = []
    for pair in (links or {}).get("pairs", []):
        pin = pair.get("pinned")
        if pair.get("error") or not pin:
            continue
        route = next((r for r in pair.get("routes", [])
                      if (r.get("a_ip"), r.get("b_ip")) == (pin.get("a_ip"), pin.get("b_ip"))), None)
        if route:
            edges.append({"a": pair["a"], "b": pair["b"], "media": route.get("media", "?"),
                          "gbps_a2b": route.get("gbps_a2b"), "gbps_b2a": route.get("gbps_b2a")})
    return edges


class Registry:
    def __init__(self):
        self.last = {}  # name -> (monotonic, seq)
        self.facts = {}  # name -> latest facts from its heartbeat
        self._lock = threading.Lock()

    def beat(self, name, seq, facts=None):
        with self._lock:
            self.last[name] = (time.monotonic(), seq)
            if facts:
                self.facts[name] = facts

    def nodes(self):
        now = time.monotonic()
        with self._lock:
            return {name: {"age_s": round(now - t, 3), "seq": seq, "up": now - t < UP_AFTER_S,
                           **({"facts": self.facts[name]} if name in self.facts else {})}
                    for name, (t, seq) in self.last.items()}

    def up_names(self):
        return {n for n, v in self.nodes().items() if v["up"]}


class Handler(BaseHTTPRequestHandler):
    registry = None

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        if self.path == "/v1/nodes":
            return self._json(200, self.registry.nodes())
        if self.path == "/v1/topology":
            return self._json(200, {"host": socket.gethostname().split(".")[0],
                                    "nodes": self.registry.nodes(),
                                    "links": topology_links(probe.load_links())})
        if self.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/heartbeat":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 <= n <= 64 * 1024:
                raise ValueError("heartbeat body length must be between 0 and 65536 bytes")
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise TypeError("heartbeat body must be a JSON object")
            name = req.get("name") or self.headers.get("X-Node-Name")
            if not isinstance(name, str) or not name:
                return self._json(400, {"error": "name must be a nonempty string"})
            facts = clean_facts(req["facts"]) if "facts" in req else None
            self.registry.beat(name, int(req.get("seq", 0)), facts)
            self._json(200, {"ok": True})
        except (ValueError, TypeError, OverflowError) as e:
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
