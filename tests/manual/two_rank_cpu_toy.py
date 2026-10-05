#!/usr/bin/env python3
"""Two-rank MLX ring test: a single hop carrying a 4-token float16 activation
between two processes on the same machine (mac-a, stock MLX 0.32.2 + nix-venv
MLX 0.32.2). Replicates the omarchy-cluster hand-rolled pattern:

  rank0: embed(ids) -> stage -> mx.eval(h) -> send(h) -> recv(token)
  rank1: recv(h) -> stage -> head -> send(token)

Reports the wall time per step so a deadlock is obvious.
"""
import argparse
import json
import os
import time
import socket


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--hostfile", required=True)
    ap.add_argument("--hidd", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.hostfile) as f:
        world = len(json.load(f))
    assert world == 2 and args.rank in (0, 1), f"world {world} rank {args.rank}"

    import mlx.core as mx

    g = mx.distributed.init(backend="ring")
    if g.rank() != args.rank or g.size() != world:
        raise SystemExit(f"ring rank/size mismatch: {g.rank()}/{g.size()}")
    print(f"CPU_TOY rank={args.rank} world={world} mlx={mx.__version__} device={mx.default_device()}",
          flush=True)

    hidden = args.hidd
    layers = args.layers
    peer = 1 - args.rank

    # no model — just a synthetic 2-layer "stage" (matmul + small non-linearity)
    rng = mx.random.key(42)
    for li in range(layers):
        w = (mx.random.normal(shape=(hidden, hidden), key=rng) * 0.1).astype(mx.float16)
        mx.eval(w)
    def stage(x, idx):
        for li in range(layers):
            x = x @ w + 0.0
        return x

    t0 = time.perf_counter()
    if args.rank == 0:
        # send activations (8 tokens) to rank1
        h = mx.random.normal(shape=(1, 8, hidden)).astype(mx.float16)
        mx.eval(h)
        print(f"RANK0 sending activations shape={h.shape} dtype={h.dtype}", flush=True)
        s1 = mx.distributed.send(h, peer)
        mx.eval(s1)
        print(f"RANK0 sent activations in {time.perf_counter()-t0:.3f}s", flush=True)
        # recv back a single int32 token
        t1 = time.perf_counter()
        token_arr = mx.distributed.recv((1,), mx.int32, src=peer)
        mx.eval(token_arr)
        token = int(token_arr.item())
        print(f"RANK0 got token {token} in {time.perf_counter()-t1:.3f}s; total {time.perf_counter()-t0:.3f}s",
              flush=True)
    else:
        t1 = time.perf_counter()
        h = mx.distributed.recv((1, 8, hidden), mx.float16, src=peer)
        mx.eval(h)
        print(f"RANK1 received activations shape={h.shape} in {time.perf_counter()-t1:.3f}s", flush=True)
        h2 = stage(h, 0)
        mx.eval(h2)
        # "head" -> argmax over a fake vocab
        logits = h2.mean(axis=-1)  # (1, 8)
        token = int(mx.argmax(logits, axis=-1).item())
        mx.eval(mx.array([token], mx.int32))
        print(f"RANK1 computed token {token}; sending back", flush=True)
        s = mx.distributed.send(mx.array([token], mx.int32), peer)
        mx.eval(s)
        print(f"RANK1 sent token in {time.perf_counter()-t0:.3f}s", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
