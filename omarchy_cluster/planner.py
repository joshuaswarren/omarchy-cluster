"""Placement planner: which nodes form the pipeline, and the layer split.

`place` orders ranks so the decode tail runs on the biggest node, checks that
every stage can hold at least one layer and that together they hold the model
(weights + KV) in free memory, and weighs each boundary by the pinned route's
measured bandwidth and RTT. The split itself is chosen at rank start
(`choose_split`) from each rank's measured ms per layer.
"""
from __future__ import annotations

import json
import os

from . import probe

BYTES_PER_KV_ELEMENT = 2  # KV stays fp16/bf16 even for 4-bit weights
DEFAULT_CTX_TOKENS = 2048
PREFILL_STEP = 2048  # mlx-lm stream_generate's default prefill_step_size; rank._stream leaves it unset


def find_model_path(model_ref):
    """Local dir, or a HF repo id resolved in the default HF cache."""
    if os.path.isdir(model_ref):
        return model_ref
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    safe = "models--" + model_ref.replace("/", "--")
    base = os.path.join(cache, safe)
    if not os.path.isdir(base):
        return None
    snaps = os.path.join(base, "snapshots")
    for entry in sorted(os.listdir(snaps)):
        snap = os.path.join(snaps, entry)
        if os.path.isfile(os.path.join(snap, "config.json")):
            return snap
    return None


def attention_pattern(tc):
    """One flag per decoder layer: True when the layer grows a per-token KV cache.

    Hybrid checkpoints (qwen3_next, qwen3_5) mix them with linear-attention layers, which keep a
    fixed-size recurrent state instead. `layer_types` names every layer; without it mlx-lm makes every
    `full_attention_interval`-th layer a full-attention layer. Anything unclear flags every layer, which
    over-reserves rather than under-reserves."""
    total = int(tc.get("num_hidden_layers", 0))
    types = tc.get("layer_types")
    if isinstance(types, list) and len(types) == total and all(isinstance(t, str) for t in types):
        pattern = [t != "linear_attention" for t in types]
    else:
        interval = int(tc.get("full_attention_interval") or 0)
        pattern = [(i + 1) % interval == 0 for i in range(total)] if interval > 1 else [True] * total
    return pattern if any(pattern) else [True] * total


def attention_layers(tc):
    """Decoder layers that grow a per-token KV cache."""
    return sum(attention_pattern(tc))


def window_pattern(tc):
    """(flags, window): flags[i] is True when layer i is a sliding-window layer, whose cache mlx-lm caps at `window`
    tokens (RotatingKVCache). Needs `layer_types` and `sliding_window`; otherwise no layer is a window layer."""
    total = int(tc.get("num_hidden_layers", 0))
    types = tc.get("layer_types")
    window = int(tc.get("sliding_window") or 0)
    if window and isinstance(types, list) and len(types) == total:
        return [t == "sliding_attention" for t in types], window
    return [False] * total, 0


def model_info(model_ref):
    path = find_model_path(model_ref)
    if not path:
        return None
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)  # qwen3_5-style multimodal wrappers nest the LM
    weights = sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".safetensors"))
    return {
        "ref": model_ref,
        "path": path,
        "layers": int(tc.get("num_hidden_layers", 0)),
        "hidden": int(tc.get("hidden_size", 0)),
        "kv_heads": int(tc.get("num_key_value_heads", tc.get("num_attention_heads", 0))),
        "head_dim": int(tc.get("head_dim", 0) or
                        (tc.get("hidden_size", 0) // max(tc.get("num_attention_heads", 1), 1))),
        "kv_layers": attention_layers(tc),
        "kv_pattern": attention_pattern(tc),
        "kv_window_layers": window_pattern(tc)[0],
        "kv_window": window_pattern(tc)[1],
        "weights_bytes": weights,
    }


def kv_bytes_per_token(info):
    """Per-token KV bytes of the whole model: only the attention layers grow a cache."""
    return 2 * (info.get("kv_layers") or info["layers"]) * info["kv_heads"] * info["head_dim"] * BYTES_PER_KV_ELEMENT


def worst_attention_layers(pattern, n):
    """Most attention layers that any n contiguous layers of the model can hold."""
    best = cur = sum(pattern[:n])
    for i in range(n, len(pattern)):
        cur += pattern[i] - pattern[i - n]
        best = max(best, cur)
    return best


def max_layers(free_bytes, info, ctx_tokens=DEFAULT_CTX_TOKENS):
    """Decoder layers (weights + KV for ctx_tokens) that fit in 90 % of free_bytes.

    A pipeline stage owns contiguous layers, and in a hybrid model their attention layers are not spread evenly,
    so the KV of n layers is charged for the n contiguous layers that hold the most cached tokens. A sliding-window
    layer holds at most its window (mlx-lm's RotatingKVCache), a full-attention layer the whole context."""
    layers = max(info["layers"], 1)
    pattern = info.get("kv_pattern")
    if pattern and len(pattern) == info["layers"]:
        per_token = 2 * info["kv_heads"] * info["head_dim"] * BYTES_PER_KV_ELEMENT
        flags = info.get("kv_window_layers") or [False] * layers
        window = info.get("kv_window") or 0
        peak = window + PREFILL_STEP - 1  # RotatingKVCache._update_concat keeps up to max_size + S - 1 entries during a prefill chunk
        tokens = [(min(ctx_tokens, peak) if flags[i] else ctx_tokens) if pattern[i] else 0 for i in range(layers)]
        budget = free_bytes * 0.9
        for n in range(layers, 0, -1):
            if n * info["weights_bytes"] / layers + per_token * worst_attention_layers(tokens, n) <= budget:
                return n
        return 0
    per_layer = (info["weights_bytes"] + kv_bytes_per_token(info) * ctx_tokens) / layers
    return int(free_bytes * 0.9 // per_layer)


def choose_split(layer_ms, caps, n_layers):
    """Per-rank layer counts (rank order) that minimise the decode step, the
    sum over ranks of layers x measured ms per layer: every rank keeps one
    layer and the rest go to the cheapest ranks first, up to each rank's
    layer cap. None when the caps cannot hold n_layers."""
    size = len(layer_ms)
    if n_layers < size or min(caps) < 1 or sum(caps) < n_layers:
        return None
    split, left = [1] * size, n_layers - size
    for r in sorted(range(size), key=lambda r: (layer_ms[r], r)):
        split[r] += min(left, caps[r] - 1)
        left -= split[r] - 1
    return split


def pinned_route(links, a, b):
    """The route between two nodes that carried traffic in both directions, else None.
    A stored pin that lacks one direction (written before probe.measured_gbps required both)
    is dropped and the pair's stored routes are ranked again."""
    for pair in links.get("pairs", []):
        names = {pair.get("a"), pair.get("b")}
        if names == {a, b}:
            stored = pair.get("pinned")
            if stored and probe.measured_gbps(stored) > 0:
                return stored
            return probe.pin_pair(pair.get("routes", []))
    return None


def plan_placement(nodes, links, info, ctx_tokens=DEFAULT_CTX_TOKENS, no_decode=(), max_stages=None):
    """nodes: {name: facts}; links: links dict. Returns the plan dict."""
    kv_tok = kv_bytes_per_token(info)

    # Keep every unified-memory node available for prefill; only the final rank
    # must be eligible for decode.
    ranks = [n for n, f in nodes.items() if f.get("unified_memory", True)]
    ranks.sort(key=lambda n: nodes[n]["memory_total_bytes"])
    no_decode = set(no_decode)
    if not any(n not in no_decode for n in ranks):
        return {"mode": "none", "reason": "no decode-eligible nodes"}

    L = info["layers"]
    plan = None
    if max_stages:
        # exact request (e.g. --stages 2): use the requested stage count.
        counts = [min(max_stages, len(ranks))]
    else:
        counts = list(range(len(ranks), 0, -1))
    for n_stages in counts:
        chosen = None
        for tail in reversed(ranks):
            if tail in no_decode:
                continue
            prefix = [n for n in ranks if n != tail]
            if len(prefix) >= n_stages - 1:
                chosen = prefix[-(n_stages - 1):] + [tail] if n_stages > 1 else [tail]
                break
        if chosen is None:
            continue
        caps = {n: max_layers(nodes[n]["memory_free_bytes"], info, ctx_tokens) for n in chosen}
        if min(caps.values()) < 1 or sum(caps.values()) < L:
            continue
        stages = [{"node": n, "max_layers": min(caps[n], L - n_stages + 1)} for n in chosen]
        ok = True
        boundaries = []
        for i in range(n_stages - 1):
            route = pinned_route(links, stages[i]["node"], stages[i + 1]["node"])
            if not route:
                ok = False
                break
            gbps = probe.measured_gbps(route)
            act_bytes = 2 * info["hidden"] * 2  # 1 token, bf16 hidden states
            rtt = route.get("rtt_tcp_ms") or 0.0
            boundaries.append({
                "from": stages[i]["node"], "to": stages[i + 1]["node"],
                "route": "%s:%s -> %s:%s" % (route["a_iface"], route["a_ip"],
                                             route["b_iface"], route["b_ip"]),
                "gbps": gbps, "rtt_ms": round(rtt, 3),
                "per_token_ms": round(act_bytes * 8 / 1e9 / gbps * 1000 + rtt, 4),
            })
        if not ok:
            continue
        plan = {"mode": "pipeline" if n_stages > 1 else "replica",
                "model": info["ref"], "layers": L, "ctx_tokens": ctx_tokens,
                "kv_layers": info.get("kv_layers") or L,
                "kv_bytes_per_token": kv_tok,
                "decode_tail": stages[-1]["node"],
                "stages": stages, "boundaries": boundaries}
        break
    if plan is None:
        plan = {"mode": "none",
                "reason": "no feasible plan: the candidate stages cannot hold the model in "
                          "free memory, or a boundary lacks a measured pinned route"}
    return plan


def plan_table(plan):
    if plan["mode"] == "none":
        return "NO PLAN: %s" % plan.get("reason")
    kv_layers = plan.get("kv_layers", plan["layers"])
    held = "" if kv_layers == plan["layers"] else ", %d with KV" % kv_layers
    lines = ["%s: %s (%d layers%s, KV %.1f KB/token)"
             % (plan["mode"].upper(), plan["model"], plan["layers"], held,
                plan["kv_bytes_per_token"] / 1024)]
    for i, s in enumerate(plan["stages"]):
        lines.append("  rank %d  %-16s fits up to %d layers%s"
                     % (i, s["node"], s["max_layers"],
                        "  (engine; samples, runs the last layers)" if i == 0 else ""))
    if plan["mode"] == "pipeline":
        lines.append("  split: chosen at rank start from measured ms/layer (serve --split overrides)")
    for b in plan["boundaries"]:
        lines.append("  boundary %s -> %s via %s  %.1f Gb/s rtt %.2f ms  ~%.3f ms/token"
                     % (b["from"], b["to"], b["route"], b["gbps"], b["rtt_ms"],
                        b["per_token_ms"]))
    return "\n".join(lines)
