# Sliding-window KV reservation for GGUF models (2026-10-15)

The llama.cpp engine planner (`gguf_info`, `rpc_layers`, `tensor_split`, `place`) charged every layer the KV cache of the full
context. A model with sliding-window layers keeps only the window in those layers: llama.cpp sizes a sliding-window layer at
the window plus `n_ubatch` tokens (512 by default). On Muse Spark Open (62 layers, 16 full attention, 46 window layers of 2048
tokens) the old charge over-reserved by 2.0x at 8192 tokens and 3.2x at 32768 tokens.

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
| 8192 | 2.08 GB | 1.02 GB |
| 32768 | 8.32 GB | 2.63 GB |
| 131072 | 33.29 GB | 9.07 GB |

## Effect on the placement

An example pool, chosen for this receipt and not measured on the machines: budgets of 40, 70, 50 and 8 GB for four RPC
devices and a gateway host that keeps 30 GB of layers in its page cache. `place()` result:

| Context | Layers kept on the gateway host, old charge | Layers kept on the gateway host, window-aware | GB paged from disk, old / new |
|---|---|---|---|
| 8192 | 7 | 7 | 0 / 0 |
| 32768 | 9 | 7 | 0 / 0 |
| 131072 | 12 | 9 | 3.5 / 0 |

The smaller KV charge leaves room for more layers on the devices, so fewer layers stay on the gateway host and nothing is
paged from disk at 131072 tokens. This changes only the plan. It does not change what llama.cpp allocates, and no run was
made with this model for this receipt.

## Not covered

- Hybrid models that mix attention and recurrent layers in a GGUF (the Qwen3.5 family) still get a KV charge on every layer.
  The MLX planner handles those from `config.json` (receipt `2026-10-15-hybrid-kv-reservation`).
- The 512-token `n_ubatch` is llama.cpp's default. `serve` does not pass `-ub`.
- llama.cpp may round the window cache up to its own alignment; this planner does not. Not checked against a running server.
