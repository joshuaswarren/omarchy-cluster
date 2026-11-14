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
PROBE_AFTER = 16  # completions below the floor at k=1 before the gate tries the configured k again


class Slots:
    """One shared admission gate whose limit can change while requests run or
    wait: a permit pool (held under a condition) plus the current limit every
    waiter re-checks. A downshift therefore also stops waiters already parked
    on it; replacing the semaphore would leave the old queue running at the
    old limit in its own permit pool. A request admitted before a downshift
    finishes at the old limit; the new limit binds future entries. Parked
    waiters hold arrival tickets and only the head of the ticket queue may
    take a freed permit, so queued requests are served in arrival order even
    when the limit changes underneath them. acquire/release match the
    semaphore calls the request path already makes."""

    def __init__(self, limit):
        self._cond = threading.Condition()
        self.limit = limit
        self.held = 0
        self._tickets = deque()  # parked waiters, arrival order

    def acquire(self, timeout):
        deadline = time.monotonic() + timeout
        ticket = object()
        with self._cond:
            self._tickets.append(ticket)
            while True:
                if self._tickets[0] is ticket and self.held < self.limit:
                    self._tickets.popleft()
                    self.held += 1
                    self._cond.notify_all()  # the next ticket may use another free permit
                    return True
                left = deadline - time.monotonic()
                if left <= 0:
                    self._tickets.remove(ticket)
                    self._cond.notify_all()  # a departed head frees the next one to re-check
                    return False
                self._cond.wait(left)

    def release(self):
        with self._cond:
            if self.held <= 0:
                raise RuntimeError("slot released without holding one")
            self.held -= 1
            self._cond.notify_all()

    def resize(self, limit):
        with self._cond:
            self.limit = limit
            self._cond.notify_all()  # a raise must re-evaluate every waiter; a lower just blocks future entries


class Gateway(BaseHTTPRequestHandler):
    engine = None  # base url of rank-0 engine
    max_concurrent = MAX_CONCURRENT
    slots = Slots(MAX_CONCURRENT)  # engine calls in flight at once
    queue_wait_s = QUEUE_WAIT_S  # how long a request waits for a slot before the 503
    floor_tokps = CONCURRENCY_FLOOR_TOKPS
    _measured = deque(maxlen=RATE_WINDOW)  # (admitted, finish, completion tokens) of the last completions
    PROBE_AFTER = PROBE_AFTER
    _off_count = 0  # completions that missed the floor since the gate last had a reason to try k>1
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
    def _aggregate_rate(cls, samples):
        """Tokens per second of engine-busy time over a window of (admitted, finish, tokens)
        samples. Busy time is the union of the [admitted, finish] intervals: concurrent
        requests overlap, so they shorten it (that is the k>1 gain), and an idle gap between
        bursts is not served work, so it adds nothing. The queue wait before admission is
        outside the interval too."""
        if len(samples) < 2:
            return None
        busy = 0.0
        first, last = None, None
        for admitted, finish, _ in sorted((a, f, t) for a, f, t in samples):
            if last is None:
                first, last = admitted, finish
            elif admitted > last:
                busy += last - first
                first, last = admitted, finish
            else:
                last = max(last, finish)
        busy += last - first
        return sum(t for _, _, t in samples) / busy if busy > 0 else None

    @classmethod
    def _record_rate(cls, admitted, tokens):
        """Aggregate completion throughput over the last RATE_WINDOW completions on the
        gateway's own wall clock (per-request tok/s falls when streams share an engine,
        so it is the wrong gate signal). Above floor_tokps the configured k runs; at or
        below it the gate switches off and requests queue one at a time. resize() keeps
        one shared permit pool: waiters already parked re-check the new limit.

        Serving at k=1 can only measure the k=1 rate, so it can never show that k>1 would
        clear the floor. After PROBE_AFTER completions below the floor with the gate off,
        the gate allows the configured k again and judges it on a fresh window: two
        completions at k>1 that clear the floor keep it on, two that do not switch it off."""
        with cls._count_lock:
            cls._measured.append((admitted, time.perf_counter(), tokens))
            rate = cls._aggregate_rate(list(cls._measured))
            if rate is None:
                return
            limit = getattr(cls.slots, "limit", None)  # a foreign gate (a test double) cannot resize
            if rate > cls.floor_tokps:
                k = cls.max_concurrent
                cls._off_count = 0
            else:
                cls._off_count += 1
                if limit == 1 and cls._off_count >= cls.PROBE_AFTER:
                    k = cls.max_concurrent  # probe: a fresh window decides
                    cls._measured.clear()
                    cls._off_count = 0
                else:
                    k = 1
            if limit is not None and k != limit:
                cls.slots.resize(k)

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
            except models.ModelsError as e:
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
        slots = self.slots  # the gate resizes in place; release the object you acquired
        if not slots.acquire(timeout=self.queue_wait_s):
            body = json.dumps({"error": "all %d request slots are busy; retry"
                               % getattr(slots, "limit", self.max_concurrent)}).encode()
            self.send_response(503)
            self.send_header("Retry-After", "5")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        admitted = time.perf_counter()  # exclude queued time; the engine starts work now
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
        self._record_rate(admitted, res["completion_tokens"])
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
    Gateway.slots = Slots(max_concurrent)
    Gateway._measured.clear()
    Gateway._off_count = 0
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
