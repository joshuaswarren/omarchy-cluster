# Sliding-window KV reservation for GGUF models (2026-10-15)

The llama.cpp engine planner (`gguf_info`, `rpc_layers`, `tensor_split`, `place`) charged every layer the KV cache of the full
context. A model with sliding-window layers keeps only a window of tokens in those layers. llama.cpp sizes such a layer at
the window times the number of server slots, plus `n_ubatch` tokens: `llama-server` starts 4 slots on one unified KV cache
when `--parallel` is not given (auto mode), and `serve` does not pass it, so the charge is 4 x 2048 + 512 = 8704 tokens per
window layer (llama.cpp `tools/server/server.cpp` auto mode and `src/llama-kv-cache-iswa.cpp`, read at commit 65840ed53). On
Muse Spark Open (62 layers, 16 full attention, 46 window layers of 2048 tokens) the old charge over-reserved by 2.2x at 32768
tokens and 3.3x at 131072 tokens; at 8192 tokens and below the window cache is not smaller than the context, so nothing changes.

## Source

| Item | Value |
|---|---|
| Model | `unsloth/Muse-Spark-Open-816B-A42B-GGUF`, `UD-IQ1_S`, 5 shards |
| GGUF keys read | `muse-spark.attention.sliding_window` = 2048, `muse-spark.attention.sliding_window_pattern` (62 booleans, True = window layer), `attention.head_count_kv` = 16, `key_length` = `value_length` = 64 |
| Window layers | 46 (layers 1, 5, 9, ... 61 are the 16 full-attention layers) |
| KV per token per layer | 16 x (64 + 64) x 2 = 4096 bytes (f16) |
| Weights | 173.34 GB in 62 layers of 2,795,767,296 bytes each, 2.29 GB in embeddings and the output matrix |
| Method | `measure.py` on the shard headers (public file list and header bytes from the Hugging Face repository; no weights downloaded) |

## Result (output of `measure.py`)

| Context (tokens) | KV, full context on every layer | KV, window layers capped |
|---|---|---|
| 2048 | 0.52 GB | 0.52 GB |
| 8192 | 2.08 GB | 2.08 GB |
| 32768 | 8.32 GB | 3.79 GB |
| 131072 | 33.29 GB | 10.23 GB |

## Effect on the placement

An example pool, chosen for this receipt and not measured on the machines: budgets of 40, 70, 50 and 8 GB for four RPC
devices and a gateway host that keeps 30 GB of layers in its page cache. `place()` result:

| Context | Layers kept on the gateway host, old charge | Layers kept on the gateway host, window-aware | GB paged from disk, old / new |
|---|---|---|---|
| 8192 | 7 | 7 | 0 / 0 |
| 32768 | 9 | 7 | 0 / 0 |
| 131072 | 12 | 10 | 3.5 / 0 |

The smaller KV charge leaves room for more layers on the devices, so fewer layers stay on the gateway host and nothing is
paged from disk at 131072 tokens. This changes only the plan. It does not change what llama.cpp allocates, and no run was
made with this model for this receipt.

## Not covered

- Hybrid models that mix attention and recurrent layers in a GGUF (the Qwen3.5 family) still get a KV charge on every layer.
  The MLX planner handles those from `config.json` (receipt `2026-10-15-hybrid-kv-reservation`).
- 512 is llama.cpp's default `n_ubatch`; `serve` does not pass `-ub`. A caller that starts llama-server with `--parallel 1` would
  need 2048 + 512 tokens per window layer; this planner would then over-reserve.
- llama.cpp may round the window cache up to its own alignment; this planner does not. Not checked against a running server.
