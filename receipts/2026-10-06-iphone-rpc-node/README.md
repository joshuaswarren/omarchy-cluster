# iPhone as a ggml RPC node, driven from Omarchy Linux (2026-10-06)

An iPhone 15 Pro Max (A17 Pro, 8 GB, iOS 27.0.1) runs llama.cpp's `rpc-server` as an app.
A 16-inch M1 Max laptop on Omarchy Linux (m1max-linux) offloads model layers to it over USB.
Everything was built, signed and installed from Linux. No Mac and no Xcode took part.

## Result

- The phone holds and computes real layers. Greedy output with the phone on CPU is byte-identical to m1max-linux alone at 7, 14 and 28 phone layers.
- The phone GPU works through Metal. The shader source is embedded and compiled on the phone, so Linux needs no Metal compiler.
- For a model that fits on m1max-linux, a split is slower than m1max-linux alone. One phone layer costs more than one m1max-linux layer, and every token waits for the phone.
- Speculative decoding with the draft model on the phone is slower than with the draft on m1max-linux.

## Table

Qwen3-1.7B Q4_K_M, `llama-bench -t 8 -p 128 -n 64 -r 3`, m1max-linux quiet window 18:13-18:17Z.
For every split, `-ot "^output\.weight=CPU"` keeps the lm_head on m1max-linux (see "Fix" below).

| Setup | pp128 t/s | tg64 t/s | ms/token |
|---|---:|---:|---:|
| m1max-linux alone (CPU) | 348.57 ± 0.22 | 71.09 ± 0.21 | 14.1 |
| m1max-linux + phone Metal, 7 of 28 layers on phone | 334.44 ± 6.55 | 35.46 ± 0.79 | 28.2 |
| m1max-linux + phone Metal, 14 layers | 341.30 ± 3.90 | 31.64 ± 0.34 | 31.6 |
| m1max-linux + phone Metal, 28 layers | 344.73 ± 8.93 | 21.06 ± 0.30 | 47.5 |
| m1max-linux + phone CPU (-t 4), 7 layers | 163.93 ± 12.60 | 33.84 ± 2.30 | 29.6 |
| m1max-linux + phone CPU, 14 layers | 106.62 ± 1.54 | 30.04 ± 1.35 | 33.3 |
| m1max-linux + phone CPU, 28 layers | 57.60 ± 0.48 | 20.77 ± 0.29 | 48.1 |

Per-token breakdown at 14 phone layers, median of 45 decode tokens, measured by `src/rpc-trace.py`
(a timing proxy between llama.cpp and the USB forward):

| Phone backend | m1max-linux side ms | send ms | phone compute + RTT ms | cycle ms | RPC cmds/token | bytes up / down |
|---|---:|---:|---:|---:|---:|---:|
| Metal | 18.25 | 0.93 | 12.30 | 32.24 | 9 | 19.4 KB / 8.0 KB |
| CPU -t 4 | 18.15 | 1.16 | 11.19 | 30.71 | 9 | 19.4 KB / 8.0 KB |

Transport is not the cost. RPC round trip over the usbmuxd forward: 0.61 ms p50, 0.83 ms p99
(300 x `DEVICE_COUNT`, `src/rpc-ping.py`). `iproxy` gave the same p50 but a 40.9 ms p99.

Speculative decoding, Qwen3-8B Q4_K_M target on m1max-linux CPU, Qwen3-0.6B Q8_0 draft, 128 tokens, 3 drafted per step:

| Setup | t/s | accept |
|---|---:|---:|
| target alone | 18.81 | - |
| draft on m1max-linux CPU | 25.12 | 64.4% |
| draft on phone Metal | 6.60 | 64.4% |

The two speculative outputs are byte-identical to each other.

## Numerics gate

Full offload to the phone (`-ngl 99`) against m1max-linux alone, `llama-perplexity --kl-divergence`, 8 x 512 tokens:

| Phone backend | mean KLD | 99% KLD | same top token | PPL ratio |
|---|---:|---:|---:|---:|
| Metal | 0.007879 | 0.071 | 95.98% | 1.018 |
| CPU | 0.009681 | 0.099 | 95.49% | 1.019 |

With the phone on Metal, greedy text matches m1max-linux at 7 layers and diverges at 14 and 28 layers
(near-tie tokens). The m1max-linux build uses gcc, the phone build uses Apple-target clang. Both use
`-march=armv8.2-a+dotprod+fp16`.

## Fix found on the way: the lm_head ran on the phone

Qwen3 ties its output matrix to the token embedding. llama.cpp placed that matmul on the RPC device
at every `-ngl` we tried, so 593.5 KB of f32 logits (151936 x 4 bytes) crossed USB per token.
`-ot "^output\.weight=CPU"` keeps it on m1max-linux and the phone returns an 8 KB hidden state instead.
The anchor matters: without `^`, the pattern also matches every `blk.N.attn_output.weight`, and the
scheduler then makes one RPC round trip per layer.

## How the app is built (all on Linux)

1. `src/build-ios.sh` or `src/build-ios-metal.sh`: cross-build the ggml static libraries with clang
   against the iPhoneOS 27.0 SDK. `GGML_METAL_EMBED_LIBRARY=ON` embeds the Metal source.
2. `src/build-app.sh` (`METAL=1` for the GPU build): compile `rpc-server.cpp` with
   `-Dmain=rpc_server_main` and link it with `src/main.m`, a UIKit host.
3. `xtool install --usb <app>`: provision, sign and install. xtool's signature executes on iOS 27.
   An rcodesign 0.29.0 signature of the same Mach-O fails at exec with `EBADEXEC` (85).
4. `pymobiledevice3 developer dvt launch "<bundle> -H 127.0.0.1 -p 50052 -t 4 [-d CPU]"`. The
   launch arguments reach `rpc-server`.

iOS 27 adds two rules that a bare `rpc-server` main breaks:

- SpringBoard kills a process that does not finish UIKit launch in about 20 s (SIGKILL while in `accept()`).
- UIKit traps an app that does not adopt the scene lifecycle (`EXC_BREAKPOINT` in
  `_UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption_block_invoke`).

`src/main.m` runs `UIApplicationMain` with a scene delegate on the main thread and starts the RPC
server on a worker thread. It also disables the idle timer.

## Reading the result

- m1max-linux needs about 0.5 ms per layer of this model (14.1 ms / 28 layers, sampling included). The
  phone needs about 0.8 ms per layer at batch 1, on CPU or Metal (11.2 to 12.3 ms for 14 layers).
  A split cannot beat m1max-linux alone for a model that m1max-linux holds.
- At batch 128 the phone GPU matches m1max-linux (pp128 345 vs 349 t/s with all 28 layers on the phone).
- The m1max-linux side of a split token takes 18 ms against 14 ms for a whole token on m1max-linux alone.
  That overhead is open.
- The case where the phone adds something real is memory: about 5.3 GiB of Metal working set
  or 7.5 GiB on CPU for layers that do not fit on the host.

## Files

- `src/`: app source, build scripts, measurement scripts. `final.sh` produced the tables.
- `data/final-1813/`: raw llama-bench tables, greedy outputs, logs and model hashes.

Model sha256 prefixes: Qwen3-1.7B-Q4_K_M d2387ca2dbfee2ff, Qwen3-8B-Q4_K_M d98cdcbd03e17ce4,
Qwen3-0.6B-Q8_0 9465e63a22add535. llama.cpp commit 65840ed, RPC protocol 7.0.0.
