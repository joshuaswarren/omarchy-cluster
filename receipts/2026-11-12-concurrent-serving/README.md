# Concurrent requests through the gateway (issue #46)

Question: does the llama.cpp engine finish more tokens per second when the gateway
passes several requests to llama-server at once?

Setup:

- One Mac Studio (Apple M1 Ultra, macOS 26.6.2), llama.cpp 0.9.0 build 12110, CPU path
  (`-dev none -ngl 0`, 8 threads), Qwen2.5-0.5B-Instruct Q4_K_M.
- Engine `omarchy_cluster.llamacpp_engine` with llama-server's default 4 slots. Three
  gateways, one per `--max-concurrent` value (1, 2, 4), all on the same engine.
- 4 client threads, one greedy 96-token completion each, per run. Runs alternate between
  the three gateways, 6 rounds, 18 runs.
- The machine was not idle: load average 12 to 16 from other work during the runs.

Result (aggregate completion tokens per second, median of 6, with min and max):

| `--max-concurrent` | median | min | max |
|---|---|---|---|
| 1 (the old one-at-a-time behavior) | 154.63 | 144.71 | 161.17 |
| 2 | 187.50 | 174.44 | 195.48 |
| 4 | 243.99 | 231.90 | 251.54 |

Four at once finish 1.58 times as many tokens per second as one at a time. The ranges do
not overlap.

Not measured: the MLX ring engine (it keeps its single worker, so it stays one request
at a time), llama.cpp over RPC across several machines, and larger models. A first try
on a shared Linux CPU host (load average 33 to 72) was discarded: the same run varied
from 19 to 110 tokens per second.

Reproduce: `bench.py REPO MODEL LLAMA_SERVER`, with `CLIENTS=4 MAX_TOKENS=96 THREADS=8
LIMITS=1,2,4 REPS=6`. Raw output: `bench.jsonl`.
