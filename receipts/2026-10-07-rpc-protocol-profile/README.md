# Where llama.cpp's RPC protocol cost goes per token, and one fix that did not help

2026-10-07, 07:42 to 08:33Z. MacBook Pro 16-inch M1 Max on Omarchy Linux, llama.cpp `65840ed`,
Qwen3-1.7B Q4_K_M, everything on 127.0.0.1 so the network is out of the picture. Each timing config
is 1 cold request, then 5 warm; time per token is llama-server's `predicted_per_token_ms`.

## The cost

| config, same session | warm median ms/token |
|---|---|
| M1 Max GPU, no RPC | 19.15 |
| same GPU through an RPC server on 127.0.0.1 | 23.89 |
| same, output layer pinned to the host | 29.84 |

Going through the RPC server costs 4.7 ms per token here. Pinning the output layer to the host makes
it worse: the host CPU then computes the 151,936-wide logits.

## What one decode token does

A recording proxy between llama-server and the RPC server logged every command
([src/rpcproxy.py](src/rpcproxy.py), [data/proxy-commands.jsonl](data/proxy-commands.jsonl),
summary [data/per-token.txt](data/per-token.txt)). Per decode token, median of 125:

- 9 small writes before the graph (inputs, positions, attention mask; 27.9 KB), none of which wait
  for a reply;
- one `GRAPH_RECOMPUTE`: the server reuses its stored graph, so it is not rebuilt per token;
- one `GET_TENSOR` that waits: 607,752 bytes of logits with the output layer on the device, 8.2 KB of
  hidden state with it on the host.

So there is one blocking round trip per token, and graph reuse already works. Upstream llama.cpp has
no RPC change after `65840ed` other than `-sm tensor` (a protocol bump).

## The fix that did not help

In the RPC server each of the 9 writes is a synchronous `ggml_backend_tensor_set`, which on a GPU
means a staging copy, a submit and a wait. The prototype queued writes of 1 MiB or less with
`ggml_backend_tensor_set_async` and synchronized once before any other command
([src/rpcasync.diff](src/rpcasync.diff), 40 lines, server only).

| order, same session | warm median ms/token |
|---|---|
| M1 Max GPU, no RPC | 17.70 |
| stock RPC server | 20.78 |
| patched RPC server | 22.57 |
| stock RPC server | 24.88 |
| patched RPC server | 24.20 |

The stock server alone moved from 20.78 to 24.88 between its two runs, more than any difference
between stock and patched. The patch shows no measurable gain, so the per-write wait is not where
most of the cost is. No upstream change was proposed.

Correctness: the patched server's greedy output equals stock's. The check written into the A/B
script hashed the `content` field, which was empty for every request (Qwen3 puts its thinking in
`reasoning_content`), so its PASS meant nothing. Recomputed from the recorded responses with
`reasoning_content` + `content`: all 30 answers (local, stock, patched) are the same 233 characters,
one digest. The measurement scripts now hash both fields and refuse an empty answer.

## Not settled

- A CPU comparison (CPU locally vs a CPU RPC server) put the protocol at 2.2 ms per token, but the
  RPC server ran 5 threads (its default) against llama-server's 8, so that figure mixes protocol and
  compute ([data/rpcprof-cpu.jsonl](data/rpcprof-cpu.jsonl)).
- Where the 4.7 ms goes is still open. The next measurement is an RPC server that logs its own time
  per command: deserializing, writing, computing, reading back.
