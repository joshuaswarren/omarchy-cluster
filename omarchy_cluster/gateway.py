"""OpenAI-compatible gateway over the pipeline engine (:8020).

Proxies /v1/chat/completions and /v1/completions to the rank-0 engine and
answers /health for router-style checks. Not registered in LiteLLM by this
command; the router side happens under the router rules.
Malformed completion bodies receive HTTP 400 before reaching the engine.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import token as cluster_token

GATEWAY_PORT = 8020
MAX_CONCURRENT = 4  # llama-server's default slot count; the MLX ranks queue behind their own worker
QUEUE_WAIT_S = 600.0
CONCURRENCY_FLOOR_TOKPS = 154.63  # the k=1 aggregate decode tok/s from the receipts table; k>1 switches off below it
RATE_WINDOW = 8  # completions per aggregate-rate reading (two rounds of 4 clients)


class Gateway(BaseHTTPRequestHandler):
    engine = None  # base url of rank-0 engine
    max_concurrent = MAX_CONCURRENT
    slot_count = MAX_CONCURRENT  # permits in `slots` right now; the 503 text uses it
    slots = threading.BoundedSemaphore(MAX_CONCURRENT)  # engine calls in flight at once
    queue_wait_s = QUEUE_WAIT_S  # how long a request waits for a slot before the 503
    floor_tokps = CONCURRENCY_FLOOR_TOKPS
    rate_window = RATE_WINDOW
    _measured = deque(maxlen=RATE_WINDOW)  # (finish, completion tokens) of the last completions
    _running = 0
    _count_lock = threading.Lock()

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _peer_is_local(self):
        return ipaddress.ip_address(self.client_address[0]).is_loopback

    def _authorized(self):
        """Loopback callers are this machine's own. Any other caller needs the cluster token."""
        if self._peer_is_local():
            return True
        token = cluster_token.read()
        return bool(token) and hmac.compare_digest(
            (self.headers.get("Authorization") or "").encode(), ("Bearer " + token).encode())

    def _refuse(self):
        body = json.dumps({"error": "bearer token required: send the cluster token as Authorization: Bearer"}).encode()
        self.send_response(401)
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def _record_rate(cls, tokens):
        """Aggregate decode tok/s over the last rate_window completions on the gateway's
        own wall clock (per-request tok/s falls when streams share an engine, so it is the
        wrong gate signal). Above floor_tokps the configured k runs; at or below it the
        gate switches off and requests queue one at a time. The flip swaps `slots`; each
        request releases the very object it acquired."""
        with cls._count_lock:
            cls._measured.append((time.perf_counter(), tokens))
            if len(cls._measured) < 2:
                return
            span = cls._measured[-1][0] - cls._measured[0][0]
            rate = sum(t for _, t in cls._measured) / span if span > 0 else None
            if rate is None:
                return
            k = cls.max_concurrent if rate > cls.floor_tokps else 1
            if k != cls.slot_count:
                cls.slots = threading.BoundedSemaphore(k)
                cls.slot_count = k

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            try:
                with urllib.request.urlopen(self.engine + "/health", timeout=2) as r:
                    r.read()
                return self._json(200, {"ok": True})
            except Exception:  # noqa: BLE001 - health check reports failure, not stack
                return self._json(503, {"ok": False})
        if not self._authorized():
            return self._refuse()
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": "omarchy-cluster", "object": "model", "owned_by": "omarchy-cluster"}]})
        if self.path in ("/models", "/v1/models/local"):
            from . import models
            return self._json(200, {"object": "list", "data": models.list_local()})
        if self.path in ("/status", "/v1/status"):
            try:
                with urllib.request.urlopen(self.engine + "/status", timeout=5) as r:
                    status = json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                return self._json(502, {"error": "engine: %s" % e})
            with self._count_lock:
                status["running"] = Gateway._running
            return self._json(200, status)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return self._refuse()
        if self.path in ("/models/pull", "/v1/models/pull"):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                from . import models
                repo = req.get("repo") if isinstance(req, dict) else None
                if not models.valid_repo(repo):
                    raise ValueError("body must be a JSON object with a \"repo\" field like \"org/name\"")
            except (ValueError, TypeError) as e:
                return self._json(400, {"error": str(e)})
            from . import models
            try:
                path = models.pull(repo)
            except SystemExit as e:
                return self._json(500, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - a failed download is a client-visible error
                return self._json(502, {"error": "pull: %s" % e})
            total, weights = models._snapshot_info(path)
            return self._json(200, {"repo": repo, "path": path, "bytes": total, "weights": weights})
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise TypeError("request body must be a JSON object")
            payload = {"prompt": req.get("prompt"), "messages": req.get("messages"),
                       "max_tokens": int(req.get("max_tokens", 64)),
                       "timing": bool(req.get("timing")), "raw": bool(req.get("raw"))}
            sampling = _sampling(req)
            payload["greedy"] = sampling is None
            if sampling:
                payload["sampling"] = sampling
        except (ValueError, TypeError, OverflowError) as e:
            return self._json(400, {"error": str(e)})
        data = json.dumps(payload).encode()
        rq = urllib.request.Request(self.engine + "/generate", data=data,
                                    headers={"Content-Type": "application/json"})
        slots = self.slots  # a gate flip swaps the class attribute; release what we acquired
        if not slots.acquire(timeout=self.queue_wait_s):
            body = json.dumps({"error": "all %d request slots are busy; retry" % self.slot_count}).encode()
            self.send_response(503)
            self.send_header("Retry-After", "5")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        with self._count_lock:
            Gateway._running += 1
        try:
            with urllib.request.urlopen(rq, timeout=600) as r:
                res = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 400:  # the engine refused the request itself: the client needs its reason
                return self._json(400, {"error": _error_text(e)})
            return self._json(502, {"error": "engine: %s" % e})
        except Exception as e:  # noqa: BLE001
            return self._json(502, {"error": "engine: %s" % e})
        finally:
            with self._count_lock:
                Gateway._running -= 1
            slots.release()
        if res.get("client_error"):
            return self._json(400, {"error": res["error"]})
        if "error" in res:
            return self._json(502, {"error": "engine: %s" % res["error"]})
        self._record_rate(res["completion_tokens"])
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


def _error_text(http_error):
    try:
        return json.loads(http_error.read())["error"]
    except (ValueError, KeyError, TypeError):
        return str(http_error)


def _number(req, key, lo, hi, lo_open=False):
    v = req.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo < v if lo_open else lo <= v) or v > hi:
        raise ValueError("%s must be a number in %s%s, %s]" % (key, "(" if lo_open else "[", lo, hi))
    return v


def _sampling(req):
    """Sampling settings from an OpenAI-style request, or None for greedy (temperature 0 or absent)."""
    temperature = _number(req, "temperature", 0, 2)
    top_p = _number(req, "top_p", 0, 1, lo_open=True)
    seed = req.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32):
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not temperature:
        return None
    return {k: v for k, v in (("temperature", temperature), ("top_p", top_p), ("seed", seed)) if v is not None}


def serve(port=GATEWAY_PORT, engine="http://127.0.0.1:8031", max_concurrent=MAX_CONCURRENT):
    Gateway.engine = engine.rstrip("/")
    Gateway.max_concurrent = max_concurrent
    Gateway.slot_count = max_concurrent
    Gateway.slots = threading.BoundedSemaphore(max_concurrent)
    Gateway._measured.clear()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Gateway)
    srv.daemon_threads = True
    print("omarchy-cluster gateway on :%d -> engine %s" % (port, Gateway.engine), flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster gateway")
    ap.add_argument("--port", type=int, default=GATEWAY_PORT)
    ap.add_argument("--engine", default="http://127.0.0.1:8031")
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, metavar="N",
                    help="requests the gateway passes to the engine at once; more wait for a slot (default %(default)s)")
    args = ap.parse_args(argv)
    if args.max_concurrent < 1:
        ap.error("--max-concurrent must be at least 1")
    serve(port=args.port, engine=args.engine, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
