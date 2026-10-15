"""Print the KV reservation of a GGUF model with and without its sliding-window layers.

Usage: python3 measure.py FIRST_SHARD.gguf [ctx ...]
Reads only the GGUF headers of every shard, so a sparse file with the right size and the real header bytes is enough.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from omarchy_cluster import llamacpp_engine as le

info = le.gguf_info(sys.argv[1])
full = {k: v for k, v in info.items() if not k.startswith("swa")}
ctxs = [int(c) for c in sys.argv[2:]] or [2048, 8192, 32768, 131072]
print("arch", info["arch"], "layers", info["n_layer"], "window layers", sum(info.get("swa_layers", [])),
      "window", info.get("swa_window"), "KV bytes per token per layer", info["kv_bytes_per_token_layer"])
print("weights %.2f GB in layers, %.2f GB elsewhere" % (sum(info["per_layer"]) / 1e9, info["other_bytes"] / 1e9))
print("| context | KV, full context on every layer | KV, window layers capped | ")
print("|---|---|---|")
for ctx in ctxs:
    print("| %d | %.2f GB | %.2f GB |" % (ctx, sum(le.layer_kv_bytes(full, ctx)) / 1e9, sum(le.layer_kv_bytes(info, ctx)) / 1e9))
