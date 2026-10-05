"""Pipeline rank runner for mx.distributed ring (2 ranks: this is the
mac-a + linux-b shape; world 1 runs fully local for single-node mode).

Rank 0 = embed + first stage layers + engine HTTP API.
Rank 1 = second stage layers + norm + lm_head (decode tail).
Hop protocol over the ring: rank0 sends an int32 header [type, T, 0]
(type 0=prefill, 1=decode), then float16 activations [1, T, hidden];
rank 1 runs its stage and sends one int32 token back.

Weights load lazily and only the kept stage is ever mx.eval'd, so each
rank holds just its layers.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import load

RING_BASE_PORT = 52100


def write_hostfile(path, rank_ips):
    with open(path, "w") as f:
        json.dump([["%s:%d" % (ip, RING_BASE_PORT)] for ip in rank_ips], f)


def _first_of(model, names):
    for n in names:
        obj = model
        try:
            for seg in n.split("."):
                obj = getattr(obj, seg)
            return n, obj
        except AttributeError:
            continue
    raise AttributeError("none of %s found on model" % names)


class PipelineRank:
    def _init_ring(self):
        """mx.distributed.init for ring retries until both ends are listening."""
        import time
        deadline = time.monotonic() + 60
        last = None
        while time.monotonic() < deadline:
            try:
                g = mx.distributed.init(backend="ring")
                if g.rank() == self.rank and g.size() == self.world:
                    return g
                last = "rank/size: %s/%s vs %s/%s" % (
                    g.rank(), g.size(), self.rank, self.world)
            except Exception as e:  # noqa: BLE001 - retry until peer is up
                last = str(e)
                time.sleep(0.5)
        raise SystemExit("ring init failed after 60 s: %s" % last)

    def __init__(self, model_ref, layer_span, rank, world):
        if world not in (1, 2):
            raise SystemExit("only world size 1 or 2 is supported (ring hop protocol is 2-rank)")
        self.rank, self.world = rank, world
        self.a, self.b = layer_span
        self.model, self.tok, self.cfg = load(model_ref, lazy=True, return_config=True)
        tc = self.cfg.get("text_config", self.cfg)
        self.hidden = int(tc["hidden_size"])
        _, self.embed = _first_of(self.model, [
            "model.embed_tokens", "model.tok_embeddings", "language_model.model.embed_tokens"])
        _, self.layers = _first_of(self.model, [
            "model.layers", "language_model.model.layers"])
        # parent module of the sliced layers (builds its own masks on hybrids)
        self._inner = None
        for cand in ("model", "language_model.model"):
            obj = self.model
            try:
                for seg in cand.split("."):
                    obj = getattr(obj, seg)
            except AttributeError:
                continue
            if getattr(obj, "layers", None) is not None:
                self._inner = obj
                break
        self.layers = self.layers[self.a:self.b]
        self.norm = self.head = None
        if rank == world - 1:
            _, self.norm = _first_of(self.model, [
                "model.norm", "model.norm_f", "language_model.model.norm"])
            try:
                _, self.head = _first_of(self.model, [
                    "lm_head", "output", "language_model.lm_head"])
            except AttributeError:
                self.head = None  # tied embeddings
        self.caches = self._make_caches()
        self._hdr = mx.zeros((3,), mx.int32)
        # MLX ring requires init() before any send/recv; world=1 is a no-op.
        # For world=2 retry until the peer also binds (both ends race to be
        # the listener; retrying closes that race).
        if world > 1:
            self.group = self._init_ring()

    def _make_caches(self):
        mk = getattr(self.model, "make_cache", None)
        if mk is None:
            lm = getattr(self.model, "language_model", None)
            mk = getattr(lm, "make_cache", None)
        if mk is not None:
            return mk()[self.a:self.b]
        return [KVCache() for _ in range(self.b - self.a)]

    def reset_cache(self):
        self.caches = self._make_caches()

    def _recv_int(self, src):
        a = mx.distributed.recv((1,), mx.int32, src=src)
        return int(a.item())

    def _recv_token(self):
        return self._recv_int((self.rank - 1) % self.world)

    def serve_hops(self):
        """Rank-last event loop: each iteration is one hop driven by rank 0."""
        while True:
            try:
                hdr = mx.distributed.recv((3,), mx.int32, src=0)
                typ = int(hdr[0].item())
                n = int(hdr[1].item()) if typ == 0 else 1
                h = mx.distributed.recv((1, n, self.hidden), mx.float16, src=0)
                h = self._run_stage(h, n)
                token = self._head_token(h)
                back = mx.array([token], mx.int32)
                mx.eval(back)
                mx.distributed.send(back, 0)
            except Exception:  # noqa: BLE001 - keep serving; drop request state
                import traceback
                traceback.print_exc()
                self.reset_cache()


    def _run_stage(self, h, n_tokens):
        # mirror the model's own mask logic (qwen3_5-style hybrids build
        # per-layer masks; plain arches use the "causal"/None convention)
        inner = self._inner
        if inner is not None and hasattr(inner, "ssm_idx"):
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask
            ssm_local = inner.ssm_idx - self.a
            fa_local = inner.fa_idx - self.a
            ssm_c = self.caches[ssm_local] if 0 <= ssm_local < len(self.caches) else self.caches[0]
            fa_c = self.caches[fa_local] if 0 <= fa_local < len(self.caches) else self.caches[-1]
            ssm_mask = create_ssm_mask(h, ssm_c)
            fa_mask = create_attention_mask(h, fa_c)
            for layer, cache in zip(self.layers, self.caches):
                mask = ssm_mask if layer.is_linear else fa_mask
                h = layer(h, mask=mask, cache=cache)
            return h
        mask = "causal" if n_tokens > 1 else None
        for layer, cache in zip(self.layers, self.caches):
            h = layer(h, mask=mask, cache=cache)
        return h

    def _head_token(self, h):
        h = self.norm(h[:, -1:, :])
        if self.head is not None:
            logits = self.head(h)
        else:
            logits = self.embed.as_linear(h)  # tied embeddings
        return int(mx.argmax(logits[:, -1, :], axis=-1).item())

    def _recv_token(self):
        a = mx.distributed.recv((1,), mx.int32, src=(self.rank - 1) % self.world)
        return int(a.item())

    def prefill(self, ids):
        """One prefill hop; rank0 returns the first generated token (or None at world 1)."""
        n = len(ids)
        h = self.embed(mx.array([ids])).astype(mx.float16)
        h = self._run_stage(h, n)
        mx.eval(h)
        if self.world == 1:
            return self._head_token(h)
        self._hdr = mx.array([0, n, 0], mx.int32)
        mx.distributed.send(self._hdr, 1)
        mx.distributed.send(h, 1)
        return self._recv_token()

    def decode_step(self, token):
        """One decode hop; rank0 returns the next token (or None at world 1)."""
        h = self.embed(mx.array([[token]])).astype(mx.float16)
        h = self._run_stage(h, 1)
        mx.eval(h)
        if self.world == 1:
            return self._head_token(h)
        self._hdr = mx.array([1, 1, 0], mx.int32)
        mx.distributed.send(self._hdr, 1)
        mx.distributed.send(h, 1)
        return self._recv_token()


class Engine:
    """Single-request engine on rank 0 (ponytail: no batching; add when it matters).

    ALL mlx work — lazy load, cache build, generation — runs on one persistent
    worker thread: mlx streams are thread-local and the HTTP server handles
    each request on its own thread.
    """

    def __init__(self, model_ref, layer_span, rank, world):
        self.rank = rank
        self.pr = None
        self.tok = None
        self.eos = set()
        self.ready = threading.Event()
        self._jobs = queue.Queue()
        self._span = layer_span
        self._ref = model_ref
        self._world = world
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        import mlx.core as mx
        try:
            self.pr = PipelineRank(self._ref, self._span, self.rank, self._world)
            self.tok = self.pr.tok
            eos = self.pr.tok.eos_token_id
            self.eos = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
            mx.eval(mx.zeros(1))  # bind this thread's default stream
            self.ready.set()
        except Exception as e:  # noqa: BLE001 - surface boot failure to callers
            self._boot_error = "%s: %s" % (type(e).__name__, e)
            self.ready.set()
            return
        while True:
            prompt_ids, max_tokens, out = self._jobs.get()
            try:
                out["res"] = self._generate(prompt_ids, max_tokens)
            except Exception as e:  # noqa: BLE001 - return the failure
                out["res"] = {"error": "%s: %s" % (type(e).__name__, e)}
            self._jobs.task_done()

    def wait_ready(self, timeout=600):
        if not self.ready.wait(timeout):
            raise RuntimeError("engine boot timed out")
        if self.pr is None:
            raise RuntimeError("engine boot failed: %s" % self._boot_error)

    def generate(self, prompt_ids, max_tokens=64):
        self.wait_ready()
        out = {}
        self._jobs.put((prompt_ids, max_tokens, out))
        self._jobs.join()
        return out["res"]

    def _generate(self, prompt_ids, max_tokens):
        self.pr.reset_cache()
        t0 = time.perf_counter()
        token = self.pr.prefill(prompt_ids)
        prefill_ms = (time.perf_counter() - t0) * 1000
        out = []
        decode_ms = 0.0
        if token is not None and token not in self.eos:
            out.append(token)
            t1 = time.perf_counter()
            for _ in range(max_tokens - 1):
                token = self.pr.decode_step(token)
                if token is None or token in self.eos:
                    break
                out.append(token)
            decode_ms = (time.perf_counter() - t1) * 1000
        text = self.tok.decode(out)
        return {"text": text, "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out),
                "prefill_ms": round(prefill_ms, 1),
                "decode_ms": round(decode_ms, 1),
                "tokps": round(len(out) / (decode_ms / 1000), 2) if out else 0.0}


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

    def do_POST(self):
        if self.path != "/generate":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        msgs = req.get("messages")
        if msgs:
            prompt = self.engine.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            ids = self.engine.tok.encode(prompt)
        else:
            ids = self.engine.tok.encode(req.get("prompt", ""))
        if req.get("greedy", True) is False:
            return self._json(400, {"error": "only greedy (temperature 0) is served"})
        res = self.engine.generate(ids, max_tokens=int(req.get("max_tokens", 64)))
        return self._json(200, res)


def serve_engine(model_ref, layer_span, rank, world, port):
    engine = Engine(model_ref, layer_span, rank, world)
    engine.wait_ready()
    EngineHandler.engine = engine
    srv = ThreadingHTTPServer(("0.0.0.0", port), EngineHandler)
    srv.daemon_threads = True
    print("rank 0 engine on :%d layers [%d:%d) world %d"
          % (port, layer_span[0], layer_span[1], world), flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster rank")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", required=True, help="stage layer span a:b")
    ap.add_argument("--hostfile", default=os.environ.get("MLX_HOSTFILE"))
    ap.add_argument("--rank", type=int, default=int(os.environ.get("MLX_RANK", -1)))
    ap.add_argument("--engine-port", type=int, default=8031)
    args = ap.parse_args(argv)
    if args.rank < 0 or not args.hostfile:
        ap.error("need --rank/--hostfile (or MLX_RANK/MLX_HOSTFILE)")
    a, b = (int(x) for x in args.layers.split(":"))
    world_len = len(json.load(open(args.hostfile)))
    if args.rank == 0:
        serve_engine(args.model, (a, b), 0, world_len, args.engine_port)
    else:
        pr = PipelineRank(args.model, (a, b), args.rank, world_len)
        print("rank %d ready, layers [%d:%d) of %s" % (args.rank, a, b, args.model),
              flush=True)
        pr.serve_hops()


if __name__ == "__main__":
    main()
