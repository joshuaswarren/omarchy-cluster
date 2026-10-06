"""omarchy-cluster rank runner (mlx-lm pipeline path).

Both ranks run the SAME generation loop, coordinated by mlx-lm's
PipelineMixin collectives. No hand-rolled ring hop.

rank 0 (mac-a) hosts the engine HTTP API and publishes each job's
prompt token ids; rank 1 long-polls for them and runs the same
stream_generate on the same tokens. Only rank 0 samples; it all_gathers
each token so rank 1 feeds that exact token to its layers (_share_token).

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


def _get_engine(engine, path, params, timeout=600):
    query = urlencode({key: value for key, value in params.items()
                       if value is not None})
    req = urllib.request.Request(engine + path + ("?" + query if query else ""),
                                 method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _poll_job(engine, last_id):
    """One long-poll of rank 0's engine: the new job, or None if none came
    within the engine's wait window."""
    job = _get_engine(engine, "/prompt/wait", {"after": last_id}, timeout=3600)
    if job.get("id") == last_id or not job.get("prompt"):
        return None
    return job


def _pipeline_split(self, group, counts):
    """PipelineMixin.pipeline with per-rank layer `counts` (rank order; rank 0
    runs the last layers). Replaces mlx-lm 0.31.3's split, which starts rank r
    at (size-r-1) * its own layer count and so drops a layer on an uneven
    split (fixed upstream in ml-explore/mlx-lm 6c5d3a298, #1816)."""
    self.pipeline_rank = group.rank()
    self.pipeline_size = group.size()
    if len(counts) != self.pipeline_size or sum(counts) != len(self.layers) or min(counts) < 1:
        raise ValueError("layer split %s does not cover %d layers on %d ranks"
                         % (counts, len(self.layers), self.pipeline_size))
    self.start_idx = sum(counts[self.pipeline_rank + 1:])
    self.end_idx = self.start_idx + counts[self.pipeline_rank]
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


def _sharded_load(model_ref, group, counts=None):
    """Load this rank's layers. Returns model, tokenizer, the split, and each
    rank's measured ms/layer (None when the split was given)."""
    from mlx_lm.models.pipeline import PipelineMixin
    from mlx_lm.utils import sharded_load
    layer_ms = None
    if not counts:
        counts, layer_ms = _measured_split(model_ref, group)
    PipelineMixin.pipeline = lambda self, g: _pipeline_split(self, g, counts)
    model, tokenizer = sharded_load(model_ref, group, None)
    _share_token(model)
    return model, tokenizer, counts, layer_ms


def _layer_ms(model, n=4, steps=8):
    """This node's decode ms per decoder layer: 1-token steps through 1 and
    through n of the model's last layers (KV prefilled with 16 tokens),
    alternated, fastest of `steps` each. The slope leaves out the embed and
    sync cost every step pays once; alternating and taking the fastest keep
    bursts of other GPU work from landing on one side only."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    inner = model.model
    layers = inner.layers[-n:]
    mx.eval(inner.embed_tokens.parameters(), [layer.parameters() for layer in layers])
    caches = [KVCache() for _ in layers]
    h = inner.embed_tokens(mx.arange(1000, 1016)[None])
    for layer, cache in zip(layers, caches):
        h = layer(h, "causal", cache)
    mx.eval(h)

    def step_s(k):
        t0 = time.perf_counter()
        h = inner.embed_tokens(mx.array([[1000]]))
        for layer, cache in zip(layers[:k], caches[:k]):
            h = layer(h, None, cache)
        mx.eval(h)
        return time.perf_counter() - t0

    pairs = [(step_s(1), step_s(n)) for _ in range(steps + 2)][2:]  # first 2 compile kernels
    one, many = min(p[0] for p in pairs), min(p[1] for p in pairs)
    return 1000 * max(many - one, 0.0) / (n - 1)


def _fastest_layer_ms(key, ms, path="~/.local/state/omarchy-cluster/layer-ms.json"):
    """The fastest positive ms/layer this node has measured for `key`, this
    start included. Other GPU work on a node can last through a whole
    calibration (ClusterRun5: mac-a measured 9.7 ms/layer, above the M2's
    8.0, and the split flipped to 1,26); the split must follow capability.
    ponytail: never forgets a faster past; delete the file after a slowdown."""
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            seen = json.load(f)
    except (OSError, ValueError):
        seen = {}
    known = [v for v in (ms, seen.get(key)) if v and v > 0]
    if not known:
        return ms
    best = min(known)
    if best != seen.get(key):
        seen[key] = best
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(seen, f, indent=1)
    return best


def _measured_split(model_ref, group):
    """Every rank measures its ms per layer and its layer cap (free memory),
    all_gathers both, and runs planner.choose_split on the same numbers.
    The measurement runs in a child process: in-process, rank 0's later Metal
    decode ran ~12x slower in 9 of 13 sessions (ClusterRun5)."""
    import subprocess
    import sys
    import mlx.core as mx
    from . import facts, planner
    info = planner.model_info(model_ref)
    if not info:
        raise RuntimeError("model not in the local HF cache: %s" % model_ref)
    cap = planner.max_layers(facts.memory_free_bytes(), info)
    out = subprocess.run(
        [sys.executable, "-c", "import sys; from mlx_lm.utils import load; "
         "from omarchy_cluster.rank import _layer_ms; print(_layer_ms(load(sys.argv[1], lazy=True)[0]))",
         model_ref], capture_output=True, text=True, check=True).stdout
    now = float(out.split()[-1])
    ms = _fastest_layer_ms("%s %s" % (sys.executable, model_ref), now)
    rows = mx.distributed.all_gather(mx.array([[ms, cap]], mx.float32), group=group,
                                     stream=mx.cpu).tolist()
    split = planner.choose_split([r[0] for r in rows], [int(r[1]) for r in rows], info["layers"])
    print("rank %d measured %.2f ms/layer (fastest seen %.2f), cap %d layers; all ranks %s -> split %s"
          % (group.rank(), now, ms, cap, rows, split), flush=True)
    if split is None:
        raise RuntimeError("ranks cannot hold %d layers: %s" % (info["layers"], rows))
    return split, [r[0] for r in rows]


def _share_token(model):
    """Only rank 0 samples. It all_gathers its token and every other rank
    swaps lm_head for that all_gather: ranks on different backends can argmax
    a near-tie differently (ClusterRun5: 26,1 split, rank texts diverged), and
    a rank that feeds its own token to its layers corrupts rank 0's output."""
    import mlx.core as mx
    if model.model.pipeline_size == 1 or "lm_head" not in model:
        return
    gather = mx.distributed.all_gather  # bound now: _stream's timing must not force it during prefill
    if model.model.pipeline_rank == 0:
        def sampler(logprobs):
            y = mx.argmax(logprobs, axis=-1)
            return mx.depends(y, gather(y.astype(mx.float32)))
        model.omarchy_sampler = sampler
        return
    vocab = model.lm_head.weight.shape[0]

    def head(h):
        # depends: keep this collective after the forward's all_gather of h
        tok = gather(mx.depends(mx.zeros((1,), mx.float32), h))[0]
        hot = (mx.arange(vocab) == tok.astype(mx.int32)).astype(mx.float32)
        return mx.broadcast_to(hot, h.shape[:-1] + (vocab,))
    model.lm_head = head


def rank1_worker(model_ref, engine, counts=None):
    """rank 1: poll rank0's engine for prompts and run the identical
    stream_generate so the pipeline collectives stay in lockstep."""
    import mlx.core as mx

    group = mx.distributed.init(backend="ring")
    print("rank 1 ring: rank=%d size=%d device=%s"
          % (group.rank(), group.size(), mx.default_device()), flush=True)
    model, tokenizer, _, _ = _sharded_load(model_ref, group, counts)
    print("rank 1 sharded_load done: %s layers %d-%d" % (
        model_ref, model.model.start_idx, model.model.end_idx - 1), flush=True)
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
            continue
        last_id = job["id"]
        print("rank 1 job %s: %d prompt tokens" % (last_id, len(job["prompt"])), flush=True)
        marks = [] if job.get("timing") else None
        parts = [r.text for r in _stream(model, tokenizer, job["prompt"], job["max_tokens"], marks)]
        print("rank 1 completed job %s: text_sha256 %s" % (
            last_id, hashlib.sha256("".join(parts).encode()).hexdigest()), flush=True)
        if marks is not None:
            print("rank 1 job %s step_ms %s" % (last_id, json.dumps(_step_ms(marks))), flush=True)


_RING_OPS = ("recv_like", "send", "all_gather")


def _stream(model, tokenizer, prompt, max_tokens, marks=None, every=1, on_sample=None):
    """mlx-lm stream_generate. Given a `marks` list, the ring ops of every
    `every`-th decode forward are evaluated on entry and exit and timestamped,
    so that step splits into own compute vs ring wait (the forced evals
    serialise it). `on_sample` gets each sampled forward's marks. Prefill
    forwards are never forced: a chunked prefill leaves its all_gather
    unevaluated on every rank, so forcing it on one rank alone (rank 0's
    watch) runs a collective the other ranks never join and the ring hangs."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    kw = {"max_tokens": max_tokens, "sampler": getattr(model, "omarchy_sampler", None)}
    if marks is None:
        yield from stream_generate(model, tokenizer, prompt, **kw)
        return
    dist = mx.distributed
    saved = {name: getattr(dist, name) for name in _RING_OPS}
    state = {"n": 0, "first": len(marks), "tokens": 0}

    def timed(name, op):
        def call(x, *a, **k):
            sampled = state["tokens"] > 0 and state["n"] % every == 0
            if not sampled:
                y = op(x, *a, **k)
            else:
                mx.eval(x)
                marks.append((name + "<", time.perf_counter()))
                y = op(x, *a, **k)
                mx.eval(y)
                marks.append((name + ">", time.perf_counter()))
            if name == "all_gather":  # the last ring op of every pipeline forward
                if sampled and on_sample:
                    on_sample(marks[state["first"]:])
                state["n"] += 1
                state["first"] = len(marks)
            return y
        return call

    for name, op in saved.items():
        setattr(dist, name, timed(name, op))
    marks.append(("start", time.perf_counter()))
    state["first"] = len(marks)
    try:
        for r in stream_generate(model, tokenizer, prompt, **kw):
            marks.append(("token", time.perf_counter()))
            state["tokens"] += 1
            yield r
    finally:
        for name, op in saved.items():
            setattr(dist, name, op)


def _step_ms(marks):
    """Mean ms per interval between consecutive ring marks after the first
    token (decode steps), keyed "a>b"; "token" is the mean inter-token gap
    and "prefill" the start -> first token wall."""
    tokens = [t for name, t in marks if name == "token"]
    if not tokens:
        return {}
    first = [name for name, _ in marks].index("token")
    ring = [(name, t) for name, t in marks[first:] if name != "token"]
    sums, counts = {}, {}
    for (a, ta), (b, tb) in zip(ring, ring[1:]):
        key = a + b
        sums[key] = sums.get(key, 0.0) + tb - ta
        counts[key] = counts.get(key, 0) + 1
    out = {key: round(1000 * sums[key] / counts[key], 2) for key in sums}
    out["prefill"] = round(1000 * (tokens[0] - marks[0][1]), 2)
    if len(tokens) > 1:
        out["token"] = round(1000 * (tokens[-1] - tokens[0]) / (len(tokens) - 1), 2)
    return out


class _Watch:
    """Trips once when `n` consecutive samples exceed `factor` x the reference
    (the lower of `expected`, from calibration or None, and the fastest sample
    seen so far) and the reference by at least `floor_ms`: a sub-ms recv wait
    tripling is noise, not contention (mac-a loopback: 0.07 -> 0.33 ms)."""

    def __init__(self, expected=None, factor=3.0, n=4, floor_ms=5.0):
        self.ref, self.factor, self.n, self.floor_ms, self.run = expected, factor, n, floor_ms, 0

    def add(self, ms):
        slow = self.ref is not None and ms > self.factor * self.ref and ms - self.ref >= self.floor_ms
        self.ref = ms if self.ref is None else min(self.ref, ms)
        self.run = self.run + 1 if slow else 0
        return self.run == self.n


class Engine:
    """rank 0: run mlx-lm stream_generate per request on one worker thread."""

    def __init__(self, model_ref, world, counts=None):
        self.model_ref = model_ref
        self.counts = counts
        self.world = world
        self.ready = threading.Event()
        self.tok = None
        self.model = None
        self._jobs = queue.Queue()
        self._id = 0
        self._pending = None  # {"id", "prompt" token ids, "max_tokens", "timing"} for rank1
        self._published = self._fetched = None  # perf_counter of publish / rank1's first fetch
        self._cond = threading.Condition()  # guards _pending/_published/_fetched; wakes rank1's long-poll
        self._error = None
        # contention watch: rank 0's own compute and its wait on ranks 1.. per sampled step
        self._watch = {}
        self._status = {"split": None, "layer_ms": None, "every": self.WATCH_EVERY,
                        "parts": {}, "events": []}
        self._status_lock = threading.Lock()
        threading.Thread(target=self._boot, daemon=True).start()

    WATCH_EVERY = 8  # sample one decode step in 8 (forced evals on rank 0 only)

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
                self.model, self.tok, split, layer_ms = _sharded_load(self.model_ref, group, self.counts)
                print("rank 0 sharded_load done: %s layers %d-%d" % (
                    self.model_ref, self.model.model.start_idx, self.model.model.end_idx - 1),
                    flush=True)
                self._start_watch(split, layer_ms)
        except Exception as e:  # noqa: BLE001 - surfaced via wait_ready
            self._error = "%s: %s" % (type(e).__name__, e)
            self.ready.set()
            return
        self.ready.set()
        while True:
            messages, max_tokens, timing, out = self._jobs.get()
            try:
                out["res"] = self._generate(messages, max_tokens, timing)
            except Exception as e:  # noqa: BLE001
                out["res"] = {"error": "%s: %s" % (type(e).__name__, e)}
            self._jobs.task_done()

    def wait_ready(self):
        """Block until boot ends. No timeout: rank 1 can sit in its node's
        gpu-turn queue far past 15 min (the old 900 s limit killed rank 0 at
        04:12Z and rank 1 then lost the ring); the runner's ENGINE_WAIT_S or
        `omarchy-cluster stop` bounds the wait."""
        self.ready.wait()
        if self.model is None:
            raise RuntimeError("engine boot failed: %s" % self._error)

    def _generate(self, messages, max_tokens, timing=False):
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        marks = [] if timing or self._watch else None
        with self._cond:
            self._id += 1
            self._pending = {"id": self._id, "prompt": prompt,
                             "max_tokens": max_tokens, "timing": timing}
            self._published, self._fetched = time.perf_counter(), None
            self._cond.notify_all()
        response = None
        parts = []
        for response in _stream(self.model, self.tok, prompt, max_tokens, marks,
                                every=1 if timing else self.WATCH_EVERY,
                                on_sample=self._sample if self._watch else None):
            parts.append(response.text)
        wall_ms = 1000 * (time.perf_counter() - self._published)
        prefill_ms = 1000 * response.prompt_tokens / response.prompt_tps
        text = "".join(parts)
        res = {"text": text,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_tokens": response.prompt_tokens,
               "completion_tokens": response.generation_tokens,
               "prefill_ms": round(prefill_ms, 1),
               "decode_ms": round(wall_ms - prefill_ms, 1),
               "tokps": round(response.generation_tps, 2),
               "peak_memory_gb": round(response.peak_memory, 2)}
        if self._fetched is not None:
            res["poll_ms"] = round(1000 * (self._fetched - self._published), 1)
        if timing:
            res["step_ms"] = _step_ms(marks)
        if self._watch:
            with self._status_lock:
                res["watch"] = json.loads(json.dumps(self._status["parts"]))
        return res

    def _start_watch(self, split, layer_ms):
        """Expected ms per step part from calibration. With --split there is
        none; rank 0 then uses its node's fastest-seen ms/layer (layer-ms.json)
        so a serve that starts during contention still flags it, and ranks
        1.. fall back to the fastest step seen."""
        import sys
        own = up = None
        if layer_ms:
            own = split[0] * layer_ms[0]
            up = sum(c * m for c, m in zip(split[1:], layer_ms[1:]))
        else:
            best = _fastest_layer_ms("%s %s" % (sys.executable, self.model_ref), 0.0)
            own = split[0] * best if best > 0 else None
        self._watch = {"rank0": _Watch(own), "rank1+": _Watch(up)}
        with self._status_lock:
            self._status.update(split=split, layer_ms=layer_ms)

    def _sample(self, forward_marks):
        """One sampled forward on rank 0: recv wait = ranks 1.. computing,
        recv exit -> all_gather entry = rank 0's own layers."""
        t = dict(forward_marks)
        if not {"recv_like<", "recv_like>", "all_gather<"} <= t.keys():
            return
        parts = {"rank1+": 1000 * (t["recv_like>"] - t["recv_like<"]),
                 "rank0": 1000 * (t["all_gather<"] - t["recv_like>"])}
        for name, ms in parts.items():
            w = self._watch[name]
            tripped = w.add(ms)
            with self._status_lock:
                self._status["parts"][name] = {"last_ms": round(ms, 2), "ref_ms": round(w.ref, 2),
                                               "ratio": round(ms / w.ref, 2), "slow_run": w.run,
                                               "contended": w.run >= w.n}
                if tripped:
                    event = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "part": name,
                             "ms": round(ms, 2), "ref_ms": round(w.ref, 2), "ratio": round(ms / w.ref, 2)}
                    self._status["events"] = (self._status["events"] + [event])[-20:]
            if tripped:
                print("rank 0 contention: %s %.1f ms/step, %.1fx its reference %.1f ms for %d sampled steps"
                      % (name, ms, ms / w.ref, w.ref, w.n), flush=True)

    def status(self):
        with self._status_lock:
            return json.loads(json.dumps(self._status))

    def generate(self, messages, max_tokens=64, timing=False):
        self.wait_ready()
        out = {}
        self._jobs.put((messages, max_tokens, timing, out))
        self._jobs.join()
        return out["res"]

    POLL_WAIT_S = 30.0

    def pending_prompt(self, after):
        """Block until a job other than `after` is published (or POLL_WAIT_S)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._pending and self._pending["id"] != after,
                                       self.POLL_WAIT_S):
                return {"id": after if after is not None else 0}
            if self._fetched is None:
                self._fetched = time.perf_counter()
            return self._pending


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
        if self.path == "/status":
            return self._json(200, self.engine.status())
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
        res = self.engine.generate(messages, max_tokens=int(req.get("max_tokens", 64)),
                                   timing=bool(req.get("timing")))
        return self._json(200, res)


def serve_engine(model_ref, world, port, counts=None):
    engine = Engine(model_ref, world, counts)
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
    ap.add_argument("--layers", default=None,
                    help="per-rank layer counts in rank order, e.g. 24,3 (rank 0 runs "
                         "the last layers); empty = measure ms/layer on every rank and "
                         "choose (planner.choose_split)")
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
    counts = [int(n) for n in args.layers.split(",")] if args.layers else None
    if args.rank == 0:
        serve_engine(args.model, world, args.engine_port, counts)
    else:
        if not args.engine:
            ap.error("rank 1 needs --engine (rank 0 base URL)")
        print("rank 1 starting (mlx-lm pipeline path)", flush=True)
        rank1_worker(args.model, args.engine, counts)


if __name__ == "__main__":
    main()
