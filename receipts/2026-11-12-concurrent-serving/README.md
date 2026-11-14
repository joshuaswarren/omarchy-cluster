# Concurrent requests through the gateway (issue #46)

Question: does the llama.cpp engine finish more tokens per second when the gateway
passes several requests to llama-server at once?

Setup (both paths):

- One Mac Studio (Apple M1 Ultra, macOS 26.6.2), llama.cpp 0.9.0 build 12110,
  Qwen2.5-0.5B-Instruct Q4_K_M, 8 threads.
- Engine `omarchy_cluster.llamacpp_engine` with llama-server's default 4 slots. Three
  gateways, one per `--max-concurrent` value (1, 2, 4), all on the same engine.
- 4 client threads, one greedy 96-token completion each, per run. Runs alternate between
  the three gateways, 6 rounds, 18 runs per path.
- The machine was not idle: load average 11 to 16 from other work during the runs.

CPU path (`-dev none -ngl 0`, the engine's default), `bench.jsonl`. Aggregate completion
tokens per second, median of 6 with min and max:

| `--max-concurrent` | median | min | max |
|---|---|---|---|
| 1 (the old one-at-a-time behavior) | 154.63 | 144.71 | 161.17 |
| 2 | 187.50 | 174.44 | 195.48 |
| 4 | 243.99 | 231.90 | 251.54 |

GPU path (Metal: `LLAMA_ARGS=-dev,MTL0,-ngl,999`, passed as `--llama-arg`; llama-server
logged that the last `-dev` and `-ngl` values were used), `bench-gpu.jsonl`:

| `--max-concurrent` | median | min | max |
|---|---|---|---|
| 1 | 146.56 | 131.05 | 163.15 |
| 2 | 206.12 | 187.57 | 213.47 |
| 4 | 594.69 | 351.59 | 658.74 |

Four at once finish 1.58 times as many tokens per second as one at a time on the CPU
path and 4.06 times on the Metal path (medians). On Metal the first round at 4 was 351.59
and the other five rounds were 547 to 659. The Metal run with one request at a time is no
faster than the CPU run (146.56 against 154.63). The likely reason is that one stream
does not keep the GPU busy with a 0.5B model; that was not measured.

Not measured: the MLX ring engine (it keeps its single worker, so it stays one request
at a time), the Vulkan path on an Omarchy Mac, llama.cpp over RPC across several
machines, and models larger than 0.5B. A first try on a shared Linux CPU host (load
average 33 to 72) was discarded: the same run varied from 19 to 110 tokens per second.

Reproduce: `bench.py REPO MODEL LLAMA_SERVER`, with `CLIENTS=4 MAX_TOKENS=96 THREADS=8
LIMITS=1,2,4 REPS=6` (add `LLAMA_ARGS=-dev,MTL0,-ngl,999` for the Metal path). Stop the
leftover llama-server between runs: the engine's exit leaves it running.

## k=1 against k=2, machine and command per row (DH-P7-03)

Same runs as the tables above (`bench.jsonl`, `bench-gpu.jsonl`); medians with min and max,
recomputed from those files, nothing new measured.

| machine | command | k=1 tok/s (median, min–max) | k=2 tok/s (median, min–max) |
|---|---|---|---|
| Mac Studio, Apple M1 Ultra, macOS 26.6.2; llama.cpp 0.9.0 build 12110, Qwen2.5-0.5B-Instruct Q4_K_M, 8 threads; CPU path (`-dev none -ngl 0`); load average 11–16 | `bench.py REPO MODEL LLAMA_SERVER` with `CLIENTS=4 MAX_TOKENS=96 THREADS=8 LIMITS=1,2,4 REPS=6` | 154.63 (144.71–161.17) | 187.50 (174.44–195.48) |
| Same machine, Metal path (`LLAMA_ARGS=-dev,MTL0,-ngl,999`) | Same command plus `LLAMA_ARGS=-dev,MTL0,-ngl,999` | 146.56 (131.05–163.15) | 206.12 (187.57–213.47) |

The on/off floor from step 3 lives in one place: `CONCURRENCY_FLOOR_TOKPS` in
`omarchy_cluster/gateway.py`, set to the CPU k=1 median above (154.63). While the gateway's
own completion throughput (tokens per wall second, from each request's engine admission to
its finish, over the last 8 completions) is above the floor, the configured `--max-concurrent`
runs; at or below it the gate switches off and requests queue one at a time. Serving at
k=1 measures at about the floor (it is the k=1 median), so the gate stays off until a k>1
stretch clears it; idle gaps count against the window and keep it off. A request admitted
before a downshift finishes at the old limit; the new limit binds future entries. `GET
/status` carries the running count.

### Not run here

Two concurrent short chats on the real lab pool. The exact command, once a pooled model is
served (`omarchy-cluster serve --model REPO --max-concurrent 2`), from the serve node:

    ( echo 'Count from one to twelve.' | omarchy-cluster chat --gateway http://127.0.0.1:8020 ) &
    ( echo 'Count from twelve down to one.' | omarchy-cluster chat --gateway http://127.0.0.1:8020 ) &
    curl -s http://127.0.0.1:8020/status   # "running": 2 while both chats are mid-turn
    wait

Reason not run: this session has no lab-pool machines attached (the assignment excludes the
real-pool step); run it where the pool is served. Issue 62's 30x4 two-node batch stays open.
