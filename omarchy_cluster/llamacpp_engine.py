"""llama.cpp engine for GGUF models, with optional RPC devices (an iPhone on USB).

The engine serves the same `POST /generate` contract as the MLX rank 0, so the
gateway does not change. Behind it runs one `llama-server`; when the model does
not fit on this host, the last layers go to an RPC device with
`--rpc HOST:PORT -ngl N`.

Placement rule: the RPC device gets layers only when the model does not fit
here. A split is slower than this host alone for any model this host holds
(measured: an M1 Max laptop alone 71.1 tok/s; with 7 of 28 layers on an
iPhone 15 Pro Max 35.5 tok/s).

Two settings matter for speed and are always applied:
- `-ot "^output\\.weight=CPU"` keeps the lm_head here. Without it, llama.cpp
  ran a tied lm_head on the RPC device and 593 KB of logits crossed USB per
  token. The `^` matters: unanchored, it also matches blk.N.attn_output.weight.
- `OMP_WAIT_POLICY=ACTIVE` keeps this host's OpenMP workers awake while the
  RPC device computes (21.1 -> 35.1 tok/s with every layer on the phone).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LM_HEAD_ON_HOST = r"^output\.weight=CPU"
HOST_FIT_FRACTION = 0.9

# GGUF metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        n = struct.calcsize(fmt)
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        return self.f.read(self.unpack("<Q")).decode("utf-8")

    def value(self, vtype):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, count = self.unpack("<I"), self.unpack("<Q")
            return [self.value(itype) for _ in range(count)]
        raise ValueError("unknown GGUF value type %d" % vtype)


def _header(path):
    """(metadata, tensor name -> byte size) of one GGUF file. Sizes come from the gaps
    between data offsets, so no quant-type table is needed."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        version = r.unpack("<I")
        if version < 2:
            raise ValueError("GGUF version %d is not supported" % version)
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            for _ in range(n_dims):
                r.unpack("<Q")
            r.unpack("<I")  # ggml type
            tensors.append((r.unpack("<Q"), name))
        align = int(meta.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    tensors.sort()
    sizes = {}
    for i, (offset, name) in enumerate(tensors):
        end = tensors[i + 1][0] if i + 1 < len(tensors) else file_size - data_start
        sizes[name] = end - offset
    return meta, sizes


def _shard_paths(path, count):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if count <= 1:
        return [path]
    if not m or int(m.group(2)) != count:
        raise ValueError("%s: split.count %d but the name is not -NNNNN-of-%05d.gguf"
                         % (path, count, count))
    return [path[:m.start()] + "-%05d-of-%05d.gguf" % (i, count) for i in range(1, count + 1)]


def gguf_info(path):
    """Layer count and byte sizes from GGUF headers; for a split model pass the first
    shard and every shard is read. Returns n_layer, layer_bytes (largest block),
    other_bytes (embeddings, output, norms), total_bytes, shards and
    kv_bytes_per_token_layer (f16 K and V)."""
    meta, sizes = _header(path)
    shards = _shard_paths(path, int(meta.get("split.count", 1)))
    for p in shards[1:]:
        sizes.update(_header(p)[1])
    arch = meta.get("general.architecture", "")
    if "%s.block_count" % arch not in meta:
        raise ValueError("%s: no general.architecture/block_count metadata" % path)
    n_layer = int(meta["%s.block_count" % arch])
    per_layer = [0] * n_layer
    other = 0
    for name, size in sizes.items():
        if name.startswith("blk."):
            per_layer[int(name.split(".")[1])] += size
        else:
            other += size
    n_kv_head = meta.get("%s.attention.head_count_kv" % arch,
                         meta.get("%s.attention.head_count" % arch, 0))
    if isinstance(n_kv_head, list):
        n_kv_head = max(n_kv_head)  # per-layer counts: charge the largest (conservative)
    n_embd = int(meta.get("%s.embedding_length" % arch, 0))
    n_head = int(meta.get("%s.attention.head_count" % arch, 1) or 1)
    k_dim = int(meta.get("%s.attention.key_length" % arch, n_embd // n_head))
    v_dim = int(meta.get("%s.attention.value_length" % arch, k_dim))
    return {"arch": arch, "n_layer": n_layer, "layer_bytes": max(per_layer),
            "other_bytes": other, "total_bytes": sum(sizes.values()), "shards": len(shards),
            "kv_bytes_per_token_layer": int(n_kv_head) * (k_dim + v_dim) * 2}


def rpc_layers(info, host_free_bytes, rpc_cap_bytes, ctx_tokens, force=None):
    """How many of the last layers go to the RPC device: 0 when the model and
    its KV cache fit in HOST_FIT_FRACTION of host free memory, else the fewest
    that make the rest fit. Raises when host + device cannot hold the model."""
    layer = info["layer_bytes"] + info["kv_bytes_per_token_layer"] * ctx_tokens
    n = info["n_layer"]
    host_budget = host_free_bytes * HOST_FIT_FRACTION
    if force is not None:
        if not 0 <= force <= n:
            raise ValueError("--rpc-layers %d is outside 0..%d" % (force, n))
        return force
    for k in range(n + 1):
        on_host = info["other_bytes"] + (n - k) * layer
        if on_host > host_budget:
            continue
        if k * layer > rpc_cap_bytes:
            break
        return k
    raise ValueError("model needs %.1f GB; host has %.1f GB usable and the RPC device %.1f GB"
                     % ((info["other_bytes"] + n * layer) / 1e9, host_budget / 1e9,
                        rpc_cap_bytes / 1e9))


def tensor_split(info, caps_bytes, ctx_tokens):
    """--tensor-split weights for N RPC devices: each device's memory budget, so the
    layers spread in proportion and none gets more than its share of the total.
    Raises when all layers plus their KV cache exceed the summed budgets. The
    embeddings and lm_head stay on the host CPU and are not charged here."""
    need = info["n_layer"] * (info["layer_bytes"] + info["kv_bytes_per_token_layer"] * ctx_tokens)
    if need > sum(caps_bytes):
        raise ValueError("layers need %.1f GB; the RPC devices have %.1f GB (%s)" % (
            need / 1e9, sum(caps_bytes) / 1e9, ", ".join("%.1f" % (c / 1e9) for c in caps_bytes)))
    return [round(c / 1e9, 2) for c in caps_bytes]


def server_cmd(llama_server, model, port, ctx_tokens, threads, rpc=None, n_rpc_layers=0,
               split=None):
    """llama-server argv. With `split`, `rpc` is a list of N endpoints and every layer
    is spread over RPC0..RPCn by those weights. Otherwise layers go to one RPC device
    only when n_rpc_layers > 0 (llama.cpp offloads the LAST n layers). The lm_head
    always stays here. `-dev RPC...` keeps a local GPU backend from taking layers."""
    cmd = [llama_server, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx_tokens), "-t", str(threads)]
    if split is not None:
        if not rpc or len(rpc) != len(split):
            raise ValueError("%d tensor-split weights for %d RPC endpoints" % (len(split), len(rpc or [])))
        cmd += ["--rpc", ",".join(rpc), "-dev", ",".join("RPC%d" % i for i in range(len(rpc))),
                "-ngl", "999", "--tensor-split", ",".join("%g" % w for w in split),
                "-ot", LM_HEAD_ON_HOST]
    elif n_rpc_layers:
        if not rpc:
            raise ValueError("layers placed on an RPC device but no --rpc endpoint")
        cmd += ["--rpc", rpc, "-dev", "RPC0", "-ngl", str(n_rpc_layers), "-ot", LM_HEAD_ON_HOST]
    else:
        cmd += ["-dev", "none", "-ngl", "0"]
    return cmd


def server_env(base=None):
    env = dict(os.environ if base is None else base)
    env["OMP_WAIT_POLICY"] = "ACTIVE"
    return env


def _post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(server_url, req):
    """Serve one gateway /generate request through llama-server, greedy."""
    max_tokens = int(req.get("max_tokens", 64))
    if req.get("messages"):
        res = _post(server_url + "/v1/chat/completions",
                    {"messages": req["messages"], "max_tokens": max_tokens,
                     "temperature": 0})
        text = res["choices"][0]["message"]["content"]
    else:
        res = _post(server_url + "/completion",
                    {"prompt": req.get("prompt") or "", "n_predict": max_tokens,
                     "temperature": 0, "cache_prompt": False})
        text = res["content"]
    t = res.get("timings") or {}
    decode_ms = float(t.get("predicted_ms", 0.0))
    completion = int(t.get("predicted_n", 0))
    return {"text": text, "prompt_tokens": int(t.get("prompt_n", 0)),
            "completion_tokens": completion, "prefill_ms": float(t.get("prompt_ms", 0.0)),
            "decode_ms": decode_ms,
            "tokps": round(completion / decode_ms * 1e3, 2) if decode_ms else 0.0}


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
                with urllib.request.urlopen(self.server_url + "/health", timeout=2) as r:
                    r.read()
                return self._json(200, {"ok": True})
            except OSError as e:
                return self._json(503, {"ok": False, "error": "llama-server: %s" % e})
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
            return self._json(200, {"error": "llama-server: %s" % e})


def wait_http(url, deadline_s=600.0, proc=None):
    """Poll url until it answers 200. Raises at the deadline, or at once when
    proc (the process that should serve url) has exited."""
    end = time.monotonic() + deadline_s
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("%s: process exited with status %d before it answered"
                               % (url, proc.returncode))
        if time.monotonic() > end:
            raise TimeoutError("%s did not answer within %.0f s" % (url, deadline_s))
        time.sleep(1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy_cluster.llamacpp_engine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-server", default="llama-server")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--server-port", type=int, default=8032)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--rpc", default=None, help="endpoint, or a comma list with --tensor-split")
    ap.add_argument("--rpc-layers", type=int, default=0)
    ap.add_argument("--tensor-split", default=None,
                    help="weights for the --rpc endpoints; spreads every layer over them")
    ap.add_argument("--load-timeout", type=float, default=3600.0,
                    help="seconds to wait for llama-server (RPC loads stream the weights)")
    args = ap.parse_args(argv)
    if args.tensor_split:
        cmd = server_cmd(args.llama_server, args.model, args.server_port, args.ctx, args.threads,
                         rpc=args.rpc.split(",") if args.rpc else None,
                         split=[float(w) for w in args.tensor_split.split(",")])
    else:
        cmd = server_cmd(args.llama_server, args.model, args.server_port, args.ctx,
                         args.threads, args.rpc, args.rpc_layers)
    print("llama-server: %s" % " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd, env=server_env(), stdin=subprocess.DEVNULL)
    server_url = "http://127.0.0.1:%d" % args.server_port
    try:
        wait_http(server_url + "/health", deadline_s=args.load_timeout, proc=server)
    except (TimeoutError, RuntimeError):
        server.terminate()
        server.wait(10)
        raise
    EngineHandler.server_url = server_url
    EngineHandler.status = {"engine": "llama.cpp", "model": args.model, "rpc": args.rpc,
                            "rpc_layers": args.rpc_layers, "tensor_split": args.tensor_split}
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), EngineHandler)
    srv.daemon_threads = True
    print("llama.cpp engine on :%d (rpc %s, split %s)" % (args.port, args.rpc, args.tensor_split),
          flush=True)
    try:
        srv.serve_forever()
    finally:
        server.terminate()
        server.wait(10)


if __name__ == "__main__":
    sys.exit(main())
