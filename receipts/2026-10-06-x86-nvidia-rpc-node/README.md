# An x86 + NVIDIA laptop as a node (experiment, llama.cpp RPC)

2026-10-06. Can a 2017-era laptop GPU — a 4 GB Quadro M1200 (Maxwell, compute
capability 5.0) in a Dell workstation laptop running Omarchy (Arch) Linux — join
the cluster as a llama.cpp RPC node, and what does it do for a small model?

Setup: llama.cpp at the same commit (`65840ed`) on both machines, so the RPC
protocol matches by construction. The laptop runs `ggml-rpc-server` built with
`-DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=50`. Two things made
the build non-obvious on current Arch:

- Current Arch CUDA is 13.x, which dropped Maxwell. CUDA 12.9.1 (last 12.x, from
  the Arch Linux Archive) still compiles `sm_50`, and the 580-series driver still
  runs it. The archive `cuda` package itself depends on `gcc14`, which is also
  archive-only now and ships in two pieces (`gcc14` + `gcc14-libs`).
- The bootstrap does all of this idempotently
  ([src/esper-node.sh](src/esper-node.sh)): install, build, open only the two RPC
  ports to the cluster LANs while running (ufw, removed again on exit), keep the
  laptop awake with `systemd-inhibit`, and serve TWO nodes: the GPU on tcp/50052
  and CPU+system-RAM on tcp/50053. A rerun skips the build and serves in seconds.

Results (Qwen3-1.7B-Q4_K_M, `llama-bench`, 5 reps, all layers offloaded, M1 Max
node on Omarchy Linux as the client):

| Nodes | pp512 t/s | tg128 t/s |
|---|---|---|
| M1 Max alone (CPU) | 268.22 ± 3.06 | 6.83 ± 1.87 |
| M1 Max + x86/CUDA node | 145.14 ± 7.99 | **9.84 ± 0.12** |
| M1 Max + x86/CUDA node + x86/CPU node | 52.95 ± 2.96 | 3.00 ± 0.99 |

Reading:

- The 4 GB Maxwell GPU lifts decode by ~44% on a model that fits it whole, but
  prefill drops: every token crosses the LAN through the RPC path.
- The two-node config trades speed for memory: 4 GB VRAM plus ~24 GB of system
  RAM join the pool, which is what a large model needs from this laptop, not
  tokens on a 1.7B.
- Tokens do NOT match the M1 Max alone under greedy decoding (fixed prompt, temp
  0, seed 42): the CUDA and CPU paths differ numerically and the text diverges
  after a few tokens. Same class as the Metal-vs-Vulkan ulp differences above;
  each config is deterministic for itself.
- Instrument caveat: `llama-cli` reports the M1 Max local decode at 56.2 t/s
  where `llama-bench` says 6.83 — the split path agrees across instruments
  (9.84 vs 9.9). The local-path discrepancy is unresolved; the split numbers
  cross-check.

Raw outputs in [data/](data/), scripts in [src/](src/).

MLX note (per the official MLX install docs, v0.32.3): the CUDA backend requires
"Nvidia architecture >= SM 7.5", so this GPU cannot run MLX at all — llama.cpp
RPC is the only path for it.
