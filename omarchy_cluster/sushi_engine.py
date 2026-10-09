"""Sushi engine for models in the Sushi quant format (EXL3 experts, affine
trunk; beamivalice/sushi).

Sushi is a single-host engine: its hot path is custom Metal kernels (EXL3
bitpacking, NAX attention) on macOS, or plain-MLX fallbacks on the Omarchy
Vulkan backend, so it cannot be sharded into ranks like the llama.cpp or MLX
engines. A node instead runs one stand-alone `sushi serve` (OpenAI-compatible)
and this proxy fronts it for the gateway.

The engine serves the same `POST /generate` contract as the MLX rank 0 and the
llama.cpp engine, so the gateway does not change. Requests are greedy
(temperature 0), like every other engine path. Sushi reports timings only in
its own log, so decode tok/s here is wall-clock over the whole request; read
sushi's log for the exact prefill/decode split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _post(url, payload, timeout=3600.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(server_url, req):
    """Serve one gateway /generate request through the sushi server, greedy."""
    max_tokens = int(req.get("max_tokens", 64))
    started = time.monotonic()
    if req.get("messages"):
        res = _post(server_url + "/v1/chat/completions",
                    {"messages": req["messages"], "max_tokens": max_tokens,
                     "temperature": 0})
        text = res["choices"][0]["message"]["content"]
    else:
        res = _post(server_url + "/v1/completions",
                    {"prompt": req.get("prompt") or "", "max_tokens": max_tokens,
                     "temperature": 0})
        text = res["choices"][0]["text"]
    elapsed_ms = (time.monotonic() - started) * 1e3
    usage = res.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    tokps = completion / elapsed_ms * 1e3 if elapsed_ms and completion else 0.0
    return {"text": text, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion, "prefill_ms": 0.0,
            "decode_ms": round(elapsed_ms, 1), "tokps": round(tokps, 2),
            "finish_reason": res["choices"][0].get("finish_reason") or "stop",
            "raw_timings": usage}


class EngineHandler(BaseHTTPRequestHandler):
    server_url = None
    status = {}

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
        if self.path == "/health":
            try:
                with urllib.request.urlopen(self.server_url + "/v1/models",
                                            timeout=5) as r:
                    r.read()
                return self._json(200, {"ok": True})
            except OSError as e:
                return self._json(503, {"ok": False, "error": "sushi: %s" % e})
        if self.path == "/status":
            return self._json(200, self.status)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        try:
            return self._json(200, generate(self.server_url, req))
        except Exception as e:  # noqa: BLE001 - the gateway relays it as 502
            return self._json(200, {"error": "sushi: %s" % e})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy_cluster.sushi_engine")
    ap.add_argument("--sushi-url", default="http://127.0.0.1:12345")
    ap.add_argument("--port", type=int, default=8031)
    args = ap.parse_args(argv)
    EngineHandler.server_url = args.sushi_url
    EngineHandler.status = {"engine": "sushi", "sushi_url": args.sushi_url}
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), EngineHandler)
    srv.daemon_threads = True
    print("sushi engine on :%d (sushi at %s)" % (args.port, args.sushi_url),
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
