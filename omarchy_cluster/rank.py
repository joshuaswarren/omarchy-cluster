"""omarchy-cluster rank runner (mlx-lm pipeline path).

Both ranks run the SAME generation loop, coordinated by mlx-lm's
PipelineMixin collectives — the verified cross-version pattern
(mlx-omarchy 4033f6fe8: byte-identical greedy across mac-a stock
0.32.2 and the M2 omarchy wheel). No hand-rolled ring hop.

rank 0 (mac-a) hosts the engine HTTP API and publishes each job's
prompt token ids; rank 1 polls for them, runs the identical
stream_generate on the same tokens, and discards its copy.

Requires a PipelineMixin model (mlx-lm 0.31.3: deepseek_v2/v3,
glm4_moe, glm4_moe_lite, ministral3). qwen2/qwen3 are NOT pipeline
models upstream; use a single node for those.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
import urllib.request
from urllib.parse import urlencode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RING_BASE_PORT = 52100


def write_hostfile(path, rank_ips):
    with open(path, "w") as f:
        json.dump([["%s:%d" % (ip, RING_BASE_PORT)] for ip in rank_ips], f)


def _post_engine(engine, path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(engine + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _get_engine(engine, path, params, timeout=600):
    query = urlencode({key: value for key, value in params.items()
                       if value is not None})
    req = urllib.request.Request(engine + path + ("?" + query if query else ""),
                                 method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _poll_job(engine, last_id):
    """One poll of rank 0's engine: the new job, or None if there is none."""
    job = _get_engine(engine, "/prompt/wait", {"after": last_id}, timeout=3600)
    if job.get("id") == last_id or not job.get("prompt"):
        return None
    return job


def _pipeline_split(self, group):
    """PipelineMixin.pipeline with ml-explore/mlx-lm 6c5d3a298 (#1816).

    mlx-lm 0.31.3 starts rank r at (size-r-1) * its own layer count, so an
    uneven split drops a layer (DeepSeek-V2-Lite: 27 layers / 2 ranks run
    0-12 and 14-26; layer 13 never runs and greedy output diverges)."""
    self.pipeline_rank = group.rank()
    self.pipeline_size = group.size()
    base, extra = divmod(len(self.layers), self.pipeline_size)
    split = [base + (r < extra) for r in range(self.pipeline_size)]
    self.start_idx = sum(split[self.pipeline_rank + 1:])
    self.end_idx = self.start_idx + split[self.pipeline_rank]
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


def _sharded_load(model_ref, group):
    from mlx_lm.models.pipeline import PipelineMixin
    from mlx_lm.utils import sharded_load
    # ponytail: overrides mlx-lm's split on every rank; delete once both hosts run an mlx-lm release containing #1816
    PipelineMixin.pipeline = _pipeline_split
    return sharded_load(model_ref, group, None)


def rank1_worker(model_ref, engine):
    """rank 1: poll rank0's engine for prompts and run the identical
    stream_generate so the pipeline collectives stay in lockstep."""
    import mlx.core as mx

    group = mx.distributed.init(backend="ring")
    print("rank 1 ring: rank=%d size=%d device=%s"
          % (group.rank(), group.size(), mx.default_device()), flush=True)
    model, tokenizer = _sharded_load(model_ref, group)
    print("rank 1 sharded_load done: %s" % model_ref, flush=True)
    print("rank 1 polling %s/prompt/wait" % engine, flush=True)
    last_id = None
    while True:
        try:
            job = _poll_job(engine, last_id)
        except Exception as e:  # noqa: BLE001 - keep polling through errors
            print("rank 1 poll error: %s" % e, flush=True)
            time.sleep(2)
            continue
        if job is None:
            time.sleep(0.5)
            continue
        last_id = job["id"]
        print("rank 1 job %s: %d prompt tokens" % (last_id, len(job["prompt"])), flush=True)
        parts = [r.text for r in _stream(model, tokenizer, job["prompt"], job["max_tokens"])]
        print("rank 1 completed job %s: text_sha256 %s" % (
            last_id, hashlib.sha256("".join(parts).encode()).hexdigest()), flush=True)


def _stream(model, tokenizer, prompt, max_tokens):
    from mlx_lm import stream_generate
    yield from stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)


class Engine:
    """rank 0: run mlx-lm stream_generate per request on one worker thread."""

    def __init__(self, model_ref, world):
        self.model_ref = model_ref
        self.world = world
        self.ready = threading.Event()
        self.tok = None
        self.model = None
        self._jobs = queue.Queue()
        self._id = 0
        self._pending = None  # {"id", "prompt" token ids, "max_tokens"} for rank1
        self._lock = threading.Lock()
        self._error = None
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        try:
            import mlx.core as mx
            # world 1: plain full-model load. mlx ring cannot self-connect and
            # sharded_load insists on a group; a replica needs neither.
            if self.world == 1:
                from mlx_lm.utils import load
                self.model, self.tok = load(self.model_ref)
                print("rank 0 replica load done: %s" % self.model_ref, flush=True)
            else:
                group = mx.distributed.init(backend="ring")
                if group.size() != self.world:
                    raise RuntimeError("ring size %d != %d" % (group.size(), self.world))
                print("rank 0 ring: rank=%d size=%d device=%s"
                      % (group.rank(), group.size(), mx.default_device()), flush=True)
                self.model, self.tok = _sharded_load(self.model_ref, group)
                print("rank 0 sharded_load done: %s" % self.model_ref, flush=True)
        except Exception as e:  # noqa: BLE001 - surfaced via wait_ready
            self._error = "%s: %s" % (type(e).__name__, e)
            self.ready.set()
            return
        self.ready.set()
        while True:
            messages, max_tokens, out = self._jobs.get()
            try:
                out["res"] = self._generate(messages, max_tokens)
            except Exception as e:  # noqa: BLE001
                out["res"] = {"error": "%s: %s" % (type(e).__name__, e)}
            self._jobs.task_done()

    def wait_ready(self, timeout=900):
        if not self.ready.wait(timeout):
            raise RuntimeError("engine boot timed out")
        if self.model is None:
            raise RuntimeError("engine boot failed: %s" % self._error)

    def _generate(self, messages, max_tokens):
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        with self._lock:
            self._id += 1
            self._pending = {"id": self._id, "prompt": prompt,
                             "max_tokens": max_tokens}
        t0 = time.perf_counter()
        response = None
        parts = []
        for response in _stream(self.model, self.tok, prompt, max_tokens):
            parts.append(response.text)
        wall = time.perf_counter() - t0
        text = "".join(parts)
        return {"text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.generation_tokens,
                "prefill_ms": 0.0,
                "decode_ms": round(wall * 1000, 1),
                "tokps": round(response.generation_tps, 2),
                "peak_memory_gb": round(response.peak_memory, 2)}

    def generate(self, messages, max_tokens=64):
        self.wait_ready()
        out = {}
        self._jobs.put((messages, max_tokens, out))
        self._jobs.join()
        return out["res"]

    def pending_prompt(self, after):
        with self._lock:
            p = self._pending
        if not p or p["id"] == after:
            return {"id": after if after is not None else 0}
        return p


class EngineHandler(BaseHTTPRequestHandler):
    engine = None

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
            return self._json(200, {"ok": True})
        if self.path.startswith("/prompt/wait"):
            after = None
            if "after=" in self.path:
                after = int(self.path.split("after=")[1].split("&")[0])
            return self._json(200, self.engine.pending_prompt(after))
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        messages = req.get("messages") or [{"role": "user",
                                            "content": req.get("prompt", "")}]
        res = self.engine.generate(messages, max_tokens=int(req.get("max_tokens", 64)))
        return self._json(200, res)


def serve_engine(model_ref, world, port):
    engine = Engine(model_ref, world)
    engine.wait_ready()
    EngineHandler.engine = engine
    srv = ThreadingHTTPServer(("0.0.0.0", port), EngineHandler)
    srv.daemon_threads = True
    print("rank 0 engine on :%d world %d (mlx-lm pipeline path)" % (port, world),
          flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster rank")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default=None, help="ignored: mlx-lm pipeline splits layers")
    ap.add_argument("--hostfile", default=os.environ.get("MLX_HOSTFILE"))
    ap.add_argument("--rank", type=int, default=int(os.environ.get("MLX_RANK", -1)))
    ap.add_argument("--engine-port", type=int, default=8031)
    ap.add_argument("--engine", default=None,
                    help="rank 1 only: rank 0 engine base URL")
    args = ap.parse_args(argv)
    if args.rank < 0 or not args.hostfile:
        ap.error("need --rank/--hostfile (or MLX_RANK/MLX_HOSTFILE)")
    with open(args.hostfile) as f:
        world = len(json.load(f))
    if args.rank == 0:
        serve_engine(args.model, world, args.engine_port)
    else:
        if not args.engine:
            ap.error("rank 1 needs --engine (rank 0 base URL)")
        print("rank 1 starting (mlx-lm pipeline path)", flush=True)
        rank1_worker(args.model, args.engine)


if __name__ == "__main__":
    main()
