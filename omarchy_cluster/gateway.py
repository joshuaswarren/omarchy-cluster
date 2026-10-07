"""OpenAI-compatible gateway over the pipeline engine (:8020).

Proxies /v1/chat/completions and /v1/completions to the rank-0 engine and
answers /health for router-style checks. Not registered in LiteLLM by this
command; the router side happens under the router rules.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATEWAY_PORT = 8020


class Gateway(BaseHTTPRequestHandler):
    engine = None  # base url of rank-0 engine

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
        if self.path in ("/health", "/v1/health"):
            try:
                with urllib.request.urlopen(self.engine + "/health", timeout=2) as r:
                    r.read()
                return self._json(200, {"ok": True})
            except Exception:  # noqa: BLE001 - health check reports failure, not stack
                return self._json(503, {"ok": False})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": "omarchy-cluster", "object": "model", "owned_by": "omarchy-cluster"}]})
        if self.path in ("/status", "/v1/status"):
            try:
                with urllib.request.urlopen(self.engine + "/status", timeout=5) as r:
                    return self._json(200, json.loads(r.read()))
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": "engine: %s" % e})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        payload = {"prompt": req.get("prompt"), "messages": req.get("messages"),
                   "max_tokens": int(req.get("max_tokens", 64)), "greedy": True,
                   "timing": bool(req.get("timing"))}
        if req.get("temperature", 0) not in (0, None):
            return self._json(400, {"error": "gateway serves temperature 0 only"})
        data = json.dumps(payload).encode()
        rq = urllib.request.Request(self.engine + "/generate", data=data,
                                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(rq, timeout=600) as r:
                res = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            return self._json(502, {"error": "engine: %s" % e})
        if "error" in res:
            return self._json(502, {"error": "engine: %s" % res["error"]})
        now = int(time.time())
        common = {
            "id": "chatcmpl-omarchy-%d" % now,
            "created": now,
            "model": req.get("model", "omarchy-cluster"),
            "usage": {"prompt_tokens": res["prompt_tokens"],
                      "completion_tokens": res["completion_tokens"],
                      "total_tokens": res["prompt_tokens"] + res["completion_tokens"]},
            "timings": {"prefill_ms": res["prefill_ms"], "decode_ms": res["decode_ms"],
                        "tokens_per_s": res["tokps"],
                        **{k: res[k] for k in ("poll_ms", "step_ms") if k in res},
                        **({"llama_server": res["raw_timings"]} if "raw_timings" in res else {})},
        }
        finish = res.get("finish_reason", "stop")
        if self.path == "/v1/chat/completions":
            return self._json(200, {**common, "object": "chat.completion", "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": res["text"]},
                "finish_reason": finish}]})
        return self._json(200, {**common, "object": "text_completion", "choices": [{
            "index": 0, "text": res["text"], "finish_reason": finish}]})


def serve(port=GATEWAY_PORT, engine="http://127.0.0.1:8031"):
    Gateway.engine = engine.rstrip("/")
    srv = ThreadingHTTPServer(("0.0.0.0", port), Gateway)
    srv.daemon_threads = True
    print("omarchy-cluster gateway on :%d -> engine %s" % (port, Gateway.engine), flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster gateway")
    ap.add_argument("--port", type=int, default=GATEWAY_PORT)
    ap.add_argument("--engine", default="http://127.0.0.1:8031")
    args = ap.parse_args(argv)
    serve(port=args.port, engine=args.engine)


if __name__ == "__main__":
    main()
