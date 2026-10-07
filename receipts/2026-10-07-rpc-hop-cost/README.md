# What one llama.cpp RPC hop costs per decoded token

2026-10-07, 04:15 to 04:22Z. Same devices as the GLM-5.3 runs that night, small model, so each
change can be measured in minutes.

## Setup

- Host: MacBook Pro 16-inch M1 Max on Omarchy Linux, llama-server at llama.cpp `65840ed` (CPU build
  for RPC configs, Vulkan build for the local one).
- Model: Qwen3-1.7B Q4_K_M, 28 layers, all layers and the output layer on the device(s) under test.
- Each config: 1 cold request, then 5 warm (3 in the interleaved repeat). Temperature 0,
  `cache_prompt` false, 64 tokens. Time per token is llama-server's `predicted_per_token_ms`.
- Script: [src/hopcost.py](src/hopcost.py), [src/hoprun.sh](src/hoprun.sh). Raw responses:
  [data/hopcost.jsonl](data/hopcost.jsonl).

## Results

| config | warm median ms/token | range |
|---|---|---|
| M1 Max GPU, no RPC | 18.58 | 18.14 to 19.16 |
| same GPU through an RPC server on 127.0.0.1 | 21.99 | 18.94 to 22.25 |
| M2 Max GPU, one wired hop | 20.41 | 18.60 to 20.82 |
| same, repeated 6 minutes later | 20.75 | 20.54 to 21.35 |
| M2 Max then Mac Studio, 14 + 14 layers | 22.60 | 22.11 to 22.95 |

1. The RPC protocol costs 3.4 ms per token on one device with no network (18.58 to 21.99, 18
   percent). In these runs every token brings back the full logits, about 0.6 MB, because the
   output layer sits on the device. The GLM cluster runs keep the output layer on the host, so
   each device boundary carries a hidden state instead (24 KB on GLM-5.3).
2. One wired hop to the M2 Max adds little on top: its TCP connect RTT was 0.24 to 0.34 ms.

## The Mac Studio could not be measured

Mac Studio Metal, one hop, in back-to-back runs:

| order | route | warm median ms/token |
|---|---|---|
| 1 | wired | 27.05 |
| 2 | Thunderbolt | 94.21 |
| 3 | wired, graph reuse off | 23.06 |
| 4 | wired | 92.73 |
| 5 | Thunderbolt | 92.68 |
| 6 | wired | 89.26 |
| 7 | Thunderbolt | 56.39 |

Single warm requests ranged from 18 to 98 ms per token. Something outside the test changed the Mac
Studio's speed; it was not identified (a CPU snapshot at 04:21Z showed nothing busy, and other GPU
work does not show there). That swing swamps any route or graph reuse effect, so no Thunderbolt or
graph reuse result is claimed.

The TCP connect RTT column reads about 1024 ms for the Mac: its RPC server takes one client at a
time, so the probe's extra connections wait for a 1 s retry. That column is valid for the Linux
endpoints only. The network measurements earlier that night put the Mac at 378 us over wired and
148 us over Thunderbolt.

## What this says about the 0.39 tok/s GLM-5.3 run

That run had six device boundaries. At no more than 3.4 ms each, they cost at most about 20 ms of a
2560 ms token, under 1 percent. Hop count and route are not where the time goes. The open candidates
are compute: the host's layers mapped from disk (paging not yet measured), the Vulkan kernels on the
Omarchy Macs, and the Mac Studio's own variable speed.
