# GLM-5.3 (753B, UD-IQ1_S, 216.7 GB) on five machines

2026-10-07. One llama.cpp endpoint for a model bigger than every machine here, spread over a Mac
Studio on macOS, three MacBook Pros on Omarchy Linux and an x86 laptop with a 4 GB NVIDIA GPU,
through llama.cpp RPC.

## Setup

Layers in order; each device holds one contiguous block.

| device | hardware | backend | layers | GB |
|---|---|---|---|---|
| omarchy-host | MacBook Pro 13-inch M1 (2020), Omarchy | CPU, llama-server host, layers mmapped from the GGUF | 0-10 | 24.79 |
| x86-laptop GPU | Dell workstation laptop, Quadro M1200 4 GB, Omarchy | CUDA | 11 | 2.67 |
| x86-laptop CPU | same laptop | CPU | 12-19 | 21.35 |
| omarchy-m1 | MacBook Pro 16-inch M1 Max (2021), Omarchy | Vulkan (Honeykrisp) | 20-33 | 37.77 |
| mac-ultra | Mac Studio M1 Ultra 128 GB, macOS 26.6.2 | Metal | 34-48 | 41.64 |
| omarchy-m2 CPU | MacBook Pro 14-inch M2 Max (2023), Omarchy | CPU (second rpc-server) | 49-56 | 22.16 |
| omarchy-m2 GPU | same laptop | Vulkan (Honeykrisp) | 57-78 | 61.51 |

- Model: unsloth GLM-5.3 UD-IQ1_S, 6 shards, 216.71 GB, 79 layers. Layer 78 is the MTP layer;
  llama.cpp does not load it.
- llama.cpp `65840ed` everywhere. omarchy-cluster `54ef8b1` on the host (split by per-layer size).
- Network: wired LAN; the host's own link is 2.5 Gb Ethernet. The x86 laptop sits on another subnet, RPC
  hello RTT 3.5 and 5.9 ms.
- Load: [src/glm53.sh](src/glm53.sh) (`serve`). Requests: [src/measure.py](src/measure.py),
  straight to llama-server.

## Results

1 cold request, then 5 warm. Same prompt ("Write a haiku about shared memory."), temperature 0,
max_tokens 64, `cache_prompt` false: every request evaluated all 20 prompt tokens (`cache_n` 0).
Decode rate is llama-server's `predicted_per_second`.

| | value |
|---|---|
| warm decode, n = 5 | median 0.39 tok/s, range 0.36 to 0.40 |
| cold decode, first request after load | 0.34 tok/s |
| prompt, 20 tokens, n = 6 | median 28.8 s, range 25.1 to 34.4 |
| load, launch to healthy | 2724 s |
| stopped by | max_tokens on all 6, inside the `<think>` reasoning |

Raw responses: [data/measure.jsonl](data/measure.jsonl). llama-server's timing lines:
[data/llamacpp-engine.log](data/llamacpp-engine.log). Recordings:
[data/load.cast](data/load.cast), [data/measure.cast](data/measure.cast).

## The six texts differ

The six greedy outputs are not identical. First differing token per pair, re-tokenized with the
GLM-5.3 tokenizer ([data/divergence.txt](data/divergence.txt), [src/diverge.py](src/diverge.py)):

- The cold request differs from every warm one at token 3 (" is" against " wants").
- Warm request 2 of 5 differs from the other warm ones at token 22 (" has" against " could").
- The other warm pairs differ at tokens 34 to 61.

Ruled out by the data: prompt cache and KV reuse (`cache_n` 0 and `prompt_n` 20 on all six), and
different inputs or sampling (same prompt, temperature 0, same server build). The cause is not
determined: nothing per device was recorded in this run, and its requests did not log token
probabilities.

### Per-device check with a small model

Same night, same llama.cpp build: Qwen3-1.7B Q4_K_M with every layer on one device at a time, 3
greedy runs each, `cache_prompt` false, 64 tokens, top-2 logprobs
([src/determinism.py](src/determinism.py), [src/detcompare.py](src/detcompare.py), raw:
[data/determinism.jsonl](data/determinism.jsonl)).

| device | tokens identical, 3 runs | largest logprob change between runs |
|---|---|---|
| x86 laptop, CUDA | yes | 0 |
| x86 laptop, CPU | yes | 0 |
| M1 Max, CPU | yes | 0 |
| Mac Studio, Metal | yes | 0 |
| M2 Max, Vulkan | yes | 0 |
| M1 Max, Vulkan | yes | 0 |

Every device repeated itself bit for bit. The devices differ from each other: logprob gaps up to
0.33 between CPU and Metal, and 0 between the two Vulkan GPUs
([data/determinism-compare.txt](data/determinism-compare.txt)). No token differed on this prompt;
the smallest top-1 over top-2 margin in any run was 0.29.

So no single device path repeats inconsistently on a dense Q4_K model. This check does not cover
what the 753B run adds: IQ1_S kernels, mixture-of-experts routing, the sparse attention indexer,
layers mapped from disk on the host, and six devices chained over RPC.

## Conditions

- During the load the Mac Studio's swap grew by 4.3 to 4.5 GB (two samplers), then held flat;
  macOS reported 29 to 33 percent of memory free.
- Peak memory use during the run, sampled about every 5 minutes: M2 Max 89.1 percent (03:15 to
  03:30Z, during the requests). No Linux node logged an out-of-memory kill.
- Another job ran on the Mac Studio from 03:04 to 03:08Z, before the first request (03:12:56Z).
- The load started under a two-request script. It was stopped at 02:41Z, mid-load, and replaced
  by the 1 cold + 5 warm measurement; the server kept loading. The load time spans both.
