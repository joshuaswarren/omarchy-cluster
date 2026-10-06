# iPhone as a ggml RPC node, driven from Omarchy Linux (2026-10-06)

An iPhone 15 Pro Max (A17 Pro, 8 GB, iOS 27.0.1) runs llama.cpp's `rpc-server` as an app.
A 16-inch M1 Max laptop on Omarchy Linux offloads model layers to it over USB.
Everything was built, signed and installed from Linux. No Mac and no Xcode took part.

## Result

- The phone holds and computes real layers. Greedy output with the phone on CPU is byte-identical to the laptop alone at 7, 14 and 28 phone layers.
- The phone GPU works through Metal. The shader source is embedded and compiled on the phone, so Linux needs no Metal compiler.
- For a model that fits on the laptop, a split is slower than the laptop alone. One phone layer costs more than one laptop layer, and every token waits for the phone.
- Speculative decoding with the draft model on the phone is slower than with the draft on the laptop.

## Table

Qwen3-1.7B Q4_K_M, `llama-bench -t 8 -p 128 -n 64 -r 3`, quiet window on the laptop 18:13-18:17Z.
For every split, `-ot "^output\.weight=CPU"` keeps the lm_head on the laptop (see "Fix" below).

| Setup | pp128 t/s | tg64 t/s | ms/token |
|---|---:|---:|---:|
| Laptop alone (CPU) | 348.57 ± 0.22 | 71.09 ± 0.21 | 14.1 |
| Laptop + phone Metal, 7 of 28 layers on phone | 334.44 ± 6.55 | 35.46 ± 0.79 | 28.2 |
| Laptop + phone Metal, 14 layers | 341.30 ± 3.90 | 31.64 ± 0.34 | 31.6 |
| Laptop + phone Metal, 28 layers | 344.73 ± 8.93 | 21.06 ± 0.30 | 47.5 |
| Laptop + phone CPU (-t 4), 7 layers | 163.93 ± 12.60 | 33.84 ± 2.30 | 29.6 |
| Laptop + phone CPU, 14 layers | 106.62 ± 1.54 | 30.04 ± 1.35 | 33.3 |
| Laptop + phone CPU, 28 layers | 57.60 ± 0.48 | 20.77 ± 0.29 | 48.1 |

Per-token breakdown at 14 phone layers, median of 45 decode tokens, measured by `src/rpc-trace.py`
(a timing proxy between llama.cpp and the USB forward):

| Phone backend | laptop side ms | send ms | phone compute + RTT ms | cycle ms | RPC cmds/token | bytes up / down |
|---|---:|---:|---:|---:|---:|---:|
| Metal | 18.25 | 0.93 | 12.30 | 32.24 | 9 | 19.4 KB / 8.0 KB |
| CPU -t 4 | 18.15 | 1.16 | 11.19 | 30.71 | 9 | 19.4 KB / 8.0 KB |

Transport is not the cost. RPC round trip over the usbmuxd forward: 0.61 ms p50, 0.83 ms p99
(300 x `DEVICE_COUNT`, `src/rpc-ping.py`). `iproxy` gave the same p50 but a 40.9 ms p99.

### Second fix: keep the laptop OpenMP workers awake

The laptop build uses OpenMP. While the laptop waits for the phone, its worker threads go to sleep, and
each token then pays to wake them. `OMP_WAIT_POLICY=ACTIVE` keeps them spinning. Phone Metal,
`llama-bench -t 8 -n 64 -r 5`, lm_head on the laptop, 18:31-18:33Z (laptop load average about 2, not a
quiet window):

| Phone layers | tg64 t/s, `OMP_WAIT_POLICY=PASSIVE` | tg64 t/s, `ACTIVE` |
|---:|---:|---:|
| 0 (laptop only) | 43.79 ± 1.26 | 71.27 ± 0.06 |
| 4 | 17.10 ± 0.27 | 49.66 ± 0.62 |
| 7 | 17.23 ± 0.12 | 44.56 ± 0.50 |
| 14 | 18.66 ± 0.34 | 44.95 ± 0.24 |

The table above used the default policy (31.64 t/s at 14 layers). With `ACTIVE`, the traced laptop
side of a 14-layer token fell from 17.4 to 10.9 ms and from 12.9 to 9.6 ms in two A/B pairs.

### Phone memory ceiling

The app has no increased-memory-limit entitlement. Qwen3-8B Q4_K_M with 18 of 36 layers on phone
Metal ran: pp128 59.64, tg32 11.00 t/s (the laptop alone 72.96 / 19.01). With all 36 layers (about
4.4 GB) the load drove iOS into memory pressure: jetsam killed system daemons for
`vm-compressor-thrashing`, widget extensions crashed, and the RPC connection closed. On this 8 GB
phone, 18 layers (about 2.2 GB) worked and 36 layers did not. The 5461 MiB that Metal reports is
not usable headroom.

### Phone alone: all 28 layers on the phone GPU

`src/metal-latency.sh`, 19:10Z, laptop load average 1.7, `OMP_WAIT_POLICY=ACTIVE` on the laptop,
`llama-bench -t 8 -p 128 -n 64 -r 5`:

| Placement | pp128 t/s | tg64 t/s |
|---|---:|---:|
| 28 layers on phone, lm_head on the laptop | 424.44 ± 6.64 | 35.13 ± 1.07 |
| everything on phone (`-ngl 99`), logits return over USB | 413.49 ± 5.51 | 26.97 ± 3.41 |

The same placement ran at 21.06 t/s in the quiet-window table with the default OpenMP policy.
Traced per-token cost (median ms):

| Phone layers | laptop side | send | phone compute + RTT |
|---:|---:|---:|---:|
| 1 | 14.71 | 0.47 | 1.79 |
| 28 | 7.39 | 0.47 | 23.28 |

The fixed cost of a phone Metal step is small (1.79 ms for one layer, including the 0.6 ms
round trip). 28 layers read about 1 GB of weights in 23 ms, about 43 GB/s, which is close to the
phone's memory bandwidth. With the lm_head on the phone and on-phone sampling (`-bs`), only a
token id returns, but greedy decode ran at 30.38 t/s against 32.24 t/s with the lm_head on the
laptop. The two `-ngl 99` outputs (with and without `-bs`) are byte-identical to each other.

Speculative decoding, Qwen3-8B Q4_K_M target on the laptop CPU, Qwen3-0.6B Q8_0 draft, 128 tokens, 3 drafted per step:

| Setup | t/s | accept |
|---|---:|---:|
| target alone | 18.81 | - |
| draft on the laptop CPU | 25.12 | 64.4% |
| draft on phone Metal | 6.60 | 64.4% |

The two speculative outputs are byte-identical to each other.

## Numerics gate

Full offload to the phone (`-ngl 99`) against the laptop alone, `llama-perplexity --kl-divergence`, 8 x 512 tokens:

| Phone backend | mean KLD | 99% KLD | same top token | PPL ratio |
|---|---:|---:|---:|---:|
| Metal | 0.007879 | 0.071 | 95.98% | 1.018 |
| CPU | 0.009681 | 0.099 | 95.49% | 1.019 |

With the phone on Metal, greedy text matches the laptop at 7 layers and diverges at 14 and 28 layers
(near-tie tokens). The laptop build uses gcc, the phone build uses Apple-target clang. Both use
`-march=armv8.2-a+dotprod+fp16`.

## Fix found on the way: the lm_head ran on the phone

Qwen3 ties its output matrix to the token embedding. llama.cpp placed that matmul on the RPC device
at every `-ngl` we tried, so 593.5 KB of f32 logits (151936 x 4 bytes) crossed USB per token.
`-ot "^output\.weight=CPU"` keeps it on the laptop and the phone returns an 8 KB hidden state instead.
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

- The laptop needs about 0.5 ms per layer of this model (14.1 ms / 28 layers, sampling included). The
  phone needs about 0.8 ms per layer at batch 1, on CPU or Metal (11.2 to 12.3 ms for 14 layers).
  A split cannot beat the laptop alone for a model that the laptop holds.
- At batch 128 the phone GPU is faster than the laptop CPU: pp128 424 vs 349 t/s with all 28 layers
  on the phone (laptop OpenMP workers awake; 345 t/s with the default policy).
- The laptop side of a split token took 18 ms with the default OpenMP policy, against 14 ms for a
  whole token on the laptop alone. `OMP_WAIT_POLICY=ACTIVE` removes most of that (44.95 t/s at 14 phone
  layers instead of 31.64).
- The phone adds memory, not speed, and less memory than it reports: about 2.2 GB of layers worked
  on this 8 GB phone, and 4.4 GB pushed iOS into memory pressure.

## Files

- `src/`: app source, build scripts, measurement scripts. `final.sh` produced the quiet-window
  tables; `metal-latency.sh` produced the phone-alone numbers.
- `data/final-1813/`, `data/metal-lat-1910/`: raw llama-bench tables, greedy outputs, logs and
  model hashes. `data/traces/`: per-command RPC timing traces (`src/trace-summary.py` reads them).

Model sha256 prefixes: Qwen3-1.7B-Q4_K_M d2387ca2dbfee2ff, Qwen3-8B-Q4_K_M d98cdcbd03e17ce4,
Qwen3-0.6B-Q8_0 9465e63a22add535. llama.cpp commit 65840ed, RPC protocol 7.0.0.
