"""Placement planner: memory- and bandwidth-weighted pipeline split.

Implements build task 4's `place` from the heterogeneous-inference proposal:
order ranks so the decode tail runs on the biggest node, cut layers so every
stage's weights + KV fit its rank's free memory, and weigh each boundary by
the pinned route's measured bandwidth and RTT.
"""
from __future__ import annotations

import json
import os

# ponytail: rank capability proxy is free memory (bigger unified memory ~= bigger
# chip ~= faster); replace with measured per-layer times when mx exists on the
# planner host.

BYTES_PER_KV_ELEMENT = 2  # KV stays fp16/bf16 even for 4-bit weights


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
        "weights_bytes": weights,
    }


def kv_bytes_per_token(info):
    return 2 * info["layers"] * info["kv_heads"] * info["head_dim"] * BYTES_PER_KV_ELEMENT


def _pinned_route(links, a, b):
    for pair in links.get("pairs", []):
        names = {pair.get("a"), pair.get("b")}
        if names == {a, b}:
            return pair.get("pinned")
    return None


def plan_placement(nodes, links, info, ctx_tokens=2048, no_decode=(), max_stages=None):
    """nodes: {name: facts}; links: links dict. Returns the plan dict."""
    kv_tok = kv_bytes_per_token(info)
    kv_total = kv_tok * ctx_tokens
    layer_bytes = info["weights_bytes"] / max(info["layers"], 1)

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
        # proportional cut: layers per stage follow free-memory share
        free = {n: max(nodes[n]["memory_free_bytes"], 1) for n in chosen}
        total_free = sum(free.values())
        cuts, acc = [0], 0.0
        for n in chosen[:-1]:
            acc += free[n] / total_free
            cuts.append(min(L - 1, max(1, round(acc * L))))
        cuts = sorted(set(cuts + [L]))
        if len(cuts) - 1 != n_stages:
            continue
        stages = []
        ok = True
        for i, n in enumerate(chosen):
            a, b = cuts[i], cuts[i + 1]
            stage_w = (b - a) * layer_bytes
            stage_kv = kv_total * (b - a) / L
            if stage_w + stage_kv > nodes[n]["memory_free_bytes"] * 0.9:
                ok = False
                break
            stages.append({"node": n, "layers": [a, b],
                           "weights_bytes": int(stage_w), "kv_bytes": int(stage_kv)})
        if not ok:
            continue
        boundaries = []
        for i in range(n_stages - 1):
            route = _pinned_route(links, stages[i]["node"], stages[i + 1]["node"])
            if not route or not (route.get("gbps_a2b") or route.get("gbps_b2a")):
                ok = False
                break
            gbps = min(v for v in (route.get("gbps_a2b"), route.get("gbps_b2a")) if v)
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
                "kv_bytes_per_token": kv_tok,
                "decode_tail": stages[-1]["node"],
                "stages": stages, "boundaries": boundaries}
        break
    if plan is None:
        plan = {"mode": "none",
                "reason": "no feasible split: every candidate stage overflowed free memory "
                          "or lacked a measured pinned route"}
    return plan


def plan_table(plan):
    if plan["mode"] == "none":
        return "NO PLAN: %s" % plan.get("reason")
    lines = ["%s: %s (%d layers, KV %.1f KB/token)"
             % (plan["mode"].upper(), plan["model"], plan["layers"],
                plan["kv_bytes_per_token"] / 1024)]
    for i, s in enumerate(plan["stages"]):
        lines.append("  rank %d  %-16s layers [%3d:%3d)  weights %5.2f GB  kv@%dt %4.2f GB%s"
                     % (i, s["node"], s["layers"][0], s["layers"][1],
                        s["weights_bytes"] / 1e9, plan["ctx_tokens"],
                        s["kv_bytes"] / 1e9,
                        "  (decode tail)" if s["node"] == plan["decode_tail"] else ""))
    for b in plan["boundaries"]:
        lines.append("  boundary %s -> %s via %s  %.1f Gb/s rtt %.2f ms  ~%.3f ms/token"
                     % (b["from"], b["to"], b["route"], b["gbps"], b["rtt_ms"],
                        b["per_token_ms"]))
    return "\n".join(lines)
