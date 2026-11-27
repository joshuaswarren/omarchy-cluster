# omarchy-cluster

## The projects

Part of the Omarchy on Apple Silicon ecosystem:

- [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) — MLX on the Apple GPU under Linux, and Core ML models on the Neural Engine.
- [omarchy-ane](https://github.com/joshuaswarren/omarchy-ane) — Linux driver for the Apple Neural Engine: DRM kernel modules and libane.
- [mil-hwx-compiler](https://github.com/joshuaswarren/mil-hwx-compiler) — compiles Core ML MIL into ANE programs on Linux (M1/M2 targets).
- [metal2vk](https://github.com/joshuaswarren/metal2vk) — translates Metal compute kernels to Vulkan SPIR-V with open tools.
- [mesa-1](https://github.com/joshuaswarren/mesa-1) — Mesa fork carrying the Honeykrisp Vulkan driver work for Apple silicon.
- [omarchy-apple-dev](https://github.com/joshuaswarren/omarchy-apple-dev) — build and ship iOS apps from Omarchy Linux, no Xcode or Mac.
- [omarchy-egpu](https://github.com/joshuaswarren/omarchy-egpu) — eGPU setup, diagnostics and measured local-inference results on Omarchy Linux.
- [ane-linux-experiments](https://github.com/joshuaswarren/ane-linux-experiments) — Neural Engine on Linux: verified primitives, receipt-backed macOS parity.
- [coreglass](https://github.com/joshuaswarren/coreglass) — profiler that shows where local inference loses speed on Apple Silicon Linux.
- [omarchy-inference-fleet](https://github.com/joshuaswarren/omarchy-inference-fleet) — reproducible, pinned single-node MLX inference baselines.
- [remnic](https://github.com/joshuaswarren/remnic) — local-first memory and context for AI agents.
- Platform upstream: [omarchy-mac](https://github.com/omacom/omarchy-mac) — Omarchy on Apple Silicon Macs.

omarchy-cluster runs one local model across the machines on your network: Macs on
macOS, Macs on Omarchy M+ Linux, even an iPhone. The machines measure each other,
decide which node runs which layers, and serve the cluster as one
OpenAI-compatible endpoint on :8020. Memory stays per node: each machine holds
its own layers and KV cache in its own (unified) memory, and the cluster does
not pool memory across machines.

Use a split when a model does not fit on any one machine you own. In the
measured runs, a split added capacity at a cost in speed. When one machine
can hold the model, start there and measure before adding nodes.

## Engines and hardware

| Engine | Models | Gateway machine | Layer nodes |
|---|---|---|---|
| `--engine mlx` (default) | mlx-lm pipeline families: `deepseek_v2`/`v3`, `glm4_moe`, `glm4_moe_lite`, `ministral3`; `qwen2`/`qwen3` run on one node (`--stages 1`); Muse Spark Open through the omarchy-mlx adapter | any node | macOS Macs (Metal, stock MLX) and Omarchy M+ Macs (Vulkan through omarchy-mlx) |
| `--engine llamacpp` | GGUF models supported by your llama.cpp build | the machine with the GGUF | llama.cpp RPC servers: Metal on macOS, Vulkan or CPU on Omarchy Linux, CUDA or CPU elsewhere, an iPhone over USB (`ios/`) |
| `--engine sushi` | Sushi format (EXL3 experts), greedy only | the Sushi host (single host, Metal or the Vulkan fallback) | not sharded |

On the Vulkan backend, runs on these nodes cover 2-bit and 4-bit models; 5-bit and
6-bit MoE checkpoints have failed. Engine details:
[docs/cli.md](docs/cli.md), placement: [docs/placement.md](docs/placement.md).

## What it does

`discover` finds nodes over mDNS; `probe` measures every node pair, pins the
fastest wired route per pair (Wi-Fi and overlay routes are probed, never used for
decode); `place` checks the model fits and prints each node's layer cap; `serve`
plans, launches one pipeline rank per node through its token-authed agent, and
serves the gateway on :8020. `stop` stops the run; `guard` watches memory and
stops when a node crosses its safety thresholds. `status` and `hub` show nodes,
pinned routes and 1 Hz liveness; the hub also serves a browser page at `/` that draws
the nodes as cards (chip, memory, load) joined by their pinned links. `--engine llamacpp` spreads a GGUF over RPC servers with
per-node memory budgets measured from the real heaps. All commands and flags:
[docs/cli.md](docs/cli.md).

## Measured performance

Headline runs, all greedy, on real machines. Full tables, per-token costs,
provenance and the iPhone node: [docs/benchmarks.md](docs/benchmarks.md); every run
has a receipt under [receipts/](receipts).

| What | Result |
|---|---|
| GLM-5.3 (216.7 GB) across five machines: Mac Studio (Metal), three Omarchy Macs (Vulkan/CPU), an x86 CUDA laptop | one endpoint, 0.34 tok/s cold, median 0.39 after; load 2724 s ([record](receipts/2026-10-07-glm53-full-7device)) |
| GLM-5.3-Flash (156.8 GB) on four Macs | 1.36 then 1.54 tok/s, identical text; load 1055 s ([receipt](receipts/2026-10-07-glm53-flash-4node)) |
| GLM-4.5-Air (60.1 GB) split Studio + M1 Max over Thunderbolt | 0.87 tok/s, every token agreeing ([docs/demo.cast](docs/demo.cast)) |
| Muse Spark Open 816B (2-bit MLX, ~237 GB), all 62 layers on GPU across four Macs, five ranks | 2.0-2.2 tok/s, TTFT about 26 s; load 10 min 48 s ([receipt](receipts/2026-10-16-muse-spark-mlx-full62)) |
| Muse Spark Open (175.6 GB GGUF) across three machines, llama.cpp RPC | 3.04 tok/s over a 107-token answer; load 857 s ([receipt](receipts/2026-10-16-muse-spark-llamacpp-rpc)) |
| DeepSeek-Coder-V2-Lite-4bit on one M1 Max, current build | median 15.73 tok/s ([receipt](receipts/2026-10-11-single-host-baseline)) |

Realistic expectations:

- The slowest node's layers set the decode speed; the planner puts as few layers
  there as memory allows, but a naive even split can be five to ten times slower
  than a good one.
- Other GPU work on a Mac can slow its node several-fold; `GET /status` reports it.
- Concurrent requests: the llama.cpp engine serves them in llama-server's slots (4 by
  default). In a one-machine test with a 0.5B model, four at once gave 1.58 times the
  tokens per second of one at a time on the CPU path and 4.06 times on Metal
  ([receipt](receipts/2026-11-12-concurrent-serving)). Vulkan, multi-machine RPC and
  larger models are not measured yet. The MLX ring
  engine runs one request at a time; the others queue. `serve --max-concurrent N`
  (default 4) sets how many requests the gateway passes on at once; the rest wait,
  and a request that waits 10 minutes gets HTTP 503 with `Retry-After`.
- Sampling: The gateway accepts `temperature` (0-2), `top_p` (0-1] and an integer `seed`; 0 or absent is greedy. The llama.cpp engine and the MLX ranks sample; Sushi answers a sampled request with an error. With a seed, the MLX ranks give the same text for the same placement; a different placement (whole or split) can give different text, because chips differ in the last bits (receipt: `receipts/2026-11-14-seeded-split/`). On the MLX ranks, rank 0 is the only rank that samples (from one seeded key, then it all_gathers its token), so a split run needs no sampling-state agreement. On the MLX ranks with `0 < top_p < 1`, the nucleus is taken within the 4096 likeliest tokens and renormalised there (the nucleus is almost always far smaller; the exact full-vocabulary sort was measured at 13x greedy in #97); probabilities outside that set are dropped, so this is not full-vocabulary nucleus sampling even when the whole nucleus would fit inside 4096. An omitted `top_p` or `top_p = 1` bypasses the cap entirely.
- Ranks agree on every token within one backend. Metal and Vulkan can differ by a
  near-tie token (up to 1 bf16 ulp); each backend is deterministic run to run.

## Requirements

- Python 3.9+ on every node; Avahi on Linux, dns-sd on macOS (both present by
  default). All nodes on one LAN or Thunderbolt link.
- macOS nodes: stock `mlx` and `mlx-lm`.
- Omarchy M+ nodes: omarchy-mlx (MLX on the GPU through its Vulkan backend).
- llama.cpp engine: llama.cpp on the gateway; each RPC node needs `ggml-rpc-server`
  built with `-DGGML_RPC=ON` at the same llama.cpp version. An iPhone node needs
  the app from `ios/` ([ios/README.md](ios/README.md)).

## Install

One command per node. Use a venv that persists: the agent starts from it at every
boot, so a venv under /tmp breaks after a reboot (`install-agent` refuses one).

On Omarchy (ranks run in omarchy-mlx's Python, whatever install it is):

```sh
python3 -m venv ~/.local/share/omarchy-cluster/venv && ~/.local/share/omarchy-cluster/venv/bin/pip install git+https://github.com/joshuaswarren/omarchy-cluster.git && ~/.local/share/omarchy-cluster/venv/bin/omarchy-cluster install-agent
```

On macOS:

```sh
python3 -m venv ~/mlx && ~/mlx/bin/pip install mlx mlx-lm git+https://github.com/joshuaswarren/omarchy-cluster.git && ~/mlx/bin/omarchy-cluster install-agent
```

`install-agent` starts the node agent on :8025 and writes a token to
`~/.config/omarchy-cluster/token`.

From a machine that already has a token, add new nodes over ssh in one command:

```sh
omarchy-cluster bootstrap me@node2 me@node3
```

For each node, `bootstrap` detects Linux or macOS, checks `python3` is 3.9 or newer,
installs Avahi on Linux when it is missing (`sudo -n`, so it needs passwordless sudo),
copies this package and your cluster token, runs `install-agent`, waits for the agent
to answer, and prints the name the node shows up under in `omarchy-cluster discover`.
It uses the node's system `python3` and installs no MLX: macOS nodes that run MLX
ranks still need `mlx` and `mlx-lm` in a Python the agent can use. Use ssh keys;
`bootstrap` never asks for a password.

## Security

- The agent on :8025 speaks TLS with its own key (made at `install-agent`, kept in
  `~/.config/omarchy-cluster/agent.key`, 0600). `bootstrap` reads the node's certificate
  over ssh and pins its SHA-256 in `~/.local/state/omarchy-cluster/pins`; this machine sends the
  token only to an address whose key matches, so a machine that takes over a recorded address
  gets a refused handshake and no token. An address with no pin is sent nothing. For a node that
  is already installed (a hosts-file entry, or a node joined before TLS) run
  `omarchy-cluster pin user@host`; an agent that predates TLS needs `bootstrap` again.
  The agent requires the cluster token (HMAC-checked on every request);
  ranks are started only through the agent, so there is no node-to-node ssh.
- The gateway on :8020 serves callers on the same machine without a token. Any other
  caller must send `Authorization: Bearer <cluster token>`; without it the gateway
  answers 401. Only `/health` is open.
- The token file is the cluster's secret. All nodes need the same token:

  ```sh
  scp ~/.config/omarchy-cluster/token node2:~/.config/omarchy-cluster/token
  ssh node2 'omarchy-cluster install-agent --token-file ~/.config/omarchy-cluster/token'
  ```

- The agent control plane is TLS with pinned keys. The gateway (:8020), the engine and the
  rank ring are still plain HTTP over your LAN, with no per-user auth: prompts and answers
  cross the LAN unencrypted. This is a homelab trust model. Run it on a network you trust.
- llama.cpp's `ggml-rpc-server` has no authentication. For a cross-host link,
  [keep RPC on loopback and wrap it in mutual TLS](docs/secure-rpc.md). This
  does not encrypt the gateway or engine HTTP APIs.

## Quickstart: two machines

With both agents installed (above), on the machine that will run the gateway:

```sh
omarchy-cluster discover        # both nodes appear
omarchy-cluster probe           # measure routes, pin the fastest
omarchy-cluster status          # chip, memory, backend
omarchy-cluster place mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
omarchy-cluster serve mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
```

`serve` prints the plan, starts the ranks through each node's agent, and leaves
the gateway on :8020. Each node loads only its own layers from its Hugging Face
cache, so download the model on every node first:

```sh
hf download mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
```

Ask it something:

```sh
curl -s http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
       "messages":[{"role":"user","content":"Write a haiku about shared memory."}],
       "max_tokens":64,"temperature":0}'
```

Or chat in the terminal: `omarchy-cluster chat` talks to the gateway on :8020
(`--gateway URL` for another one), keeps the conversation, and prints the speed
of each reply in tokens per second. `/reset` starts over, `/exit` leaves.
Tokens stream into the terminal as they decode: send `"stream": true` to
`/v1/chat/completions` and the gateway answers SSE delta chunks ending with
`data: [DONE]`; without the flag the reply stays one JSON body.

`omarchy-cluster stop` stops everything. For a model bigger than one node, pick
the split yourself so a machine you use for other work keeps free memory, for
example `serve mlx-community/GLM-4.5-Air-4bit --split 31,15`. `--nodes` with
`--split` names the nodes in rank order, any number of ranks.
[docs/placement.md](docs/placement.md) explains how the split is chosen.

## GGUF models and larger clusters

Use `--engine llamacpp` for a model supported by llama.cpp. Put the GGUF on
the gateway and use matching llama.cpp builds with RPC enabled on the other
nodes. An iPhone can be an RPC node; see [the iPhone guide](ios/README.md).

[The CLI guide](docs/cli.md) explains `--rpc-node`, memory budgets and model
paths. The [GGUF example](receipts/2026-10-16-muse-spark-llamacpp-rpc) gives
the complete commands for a three-machine run. For MLX across four Macs,
see [the MLX example](receipts/2026-10-16-muse-spark-mlx-full62/README.md).

RPC servers have no authentication. Keep raw RPC on loopback and use the
[mutual TLS recipe](docs/secure-rpc.md) to connect hosts.

## Documentation

- [Command reference](docs/cli.md): flags, gateway API and environment switches.
- [Placement](docs/placement.md): layer allocation, KV cache and ring addresses.
- [Troubleshooting](docs/troubleshooting.md): logs, memory limits and prompts.
- [Measured results](docs/benchmarks.md): benchmark tables and supporting records.
- [Wire a Mac Studio to a DGX Spark](docs/mac-studio-spark-connectx.md): set up the link and test TCP speed.

## Contributing

PRs welcome. The tests need pytest and local loopback sockets; they run without a
GPU, model downloads or access to other hosts.

```sh
python3 -m pip install pytest==8.4.2
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

CI runs these tests on Ubuntu 24.04 and macOS 15 with Python 3.9 and 3.13.

## Milestone run matrix

Cluster milestone rows CL04-CL16 with the receipt paths that exist and the ones
still missing. Status follows the lane claims log as of 2026-11-26, folded
with the post-merge states of PRs #150, #153, #155, #156 and #157.

### CL04 — Spark DGX OS solo baseline — ran, numbers posted to the card

| Receipt path | State |
|---|---|
| `spark:~/cl02-out/llama8b/run.json` (cold) and `spark:~/cl02-out/llama8b-warm/repeat-{1,2,3}/run.json` | exists on the Spark host |
| `spark:~/cl02-out/qwen17b/run.json` (cold) and `spark:~/cl02-out/qwen17b-warm/repeat-{1,2,3}/run.json` | exists on the Spark host |
| `receipts/milestone1-status-2026-11-26.md` — per-artifact inventory with results and parity digests | exists (PR #155) |
| `schemas/cl02-baseline-1.example.json` + `tests/test_cl02_schema.py` — the `cl02-baseline/1` output schema these run.json files must keep | exists (PR #153) |
| a copy of the run artifacts under `receipts/` in this repo | missing |

### CL05 — macOS solo baseline — claimed, no run

| Receipt path | State |
|---|---|
| `receipts/milestone1-jwm2-macos-solo/run.json` | missing |
| staged command sheet + blocker chain | documented in `receipts/milestone1-status-2026-11-26.md` (PR #155) |

### CL06 — vendor-OS EXO pair (macOS + DGX OS) — dry-run only

| Receipt path | State |
|---|---|
| `receipts/milestone1-exo-pair/run-exo-pair.sh` | exists |
| `omarchy_cluster/exo_pair.py` — now fails a pair run on a wrong strict token-ID digest | exists, hardened (PR #156, CLX-3) |
| `receipts/milestone1-exo-pair/dry-run/` (`run.json`, `exo-pair-details.json`, `dry-run.log`) | exists (dry-run) |
| `receipts/milestone1-exo-pair/live-*/run.json` | missing |
| staged Spark arm + the one timed-out arm | documented in `receipts/milestone1-status-2026-11-26.md` (PR #155) |

### CL07 — unified four-device C-a serve medians — measured

| Receipt path | State |
|---|---|
| `receipts/2026-11-22-ca-serve-ab/rtx5060ti-cuda.json` | exists |
| `receipts/2026-11-22-ca-serve-ab/m2max-apple-vulkan.json` | exists |
| `receipts/2026-11-22-ca-serve-ab/spark-cuda.json` | exists (PR #144) |
| `receipts/2026-11-22-ca-serve-ab/spark-cuda-attempt-1-oom.json` | exists (recorded attempt) |
| B60 Vulkan arm JSON | measured, kept off-repo per the directory README |
| unified medians table | published on issue #124 |

### CL08 — joint-request phase separation — merged

| Artifact | State |
|---|---|
| joint engine: prefill, KV transfer, decode over llama-server slots (PR #133) | merged |
| `docs/cl08-test-inventory.md` loopback/mock coverage + gap list | merged (PR #150) |
| `docs/joint-request-path.md` sequence note for the joint-request path | merged (PR #151) |

### CL09 — joint-request A/B on Omarchy Mac + Spark — open

| Receipt path | State |
|---|---|
| receipt dir proving lower 8,192-token TTFT/total vs Mac solo with identical token IDs | missing |

### CL10 — Omarchy ARM64 bring-up kit for the Spark — merged

| Artifact | State |
|---|---|
| `spark-omarchy/` boot, network, PKGBUILDs and the update/rollback runbook (PR #132) | merged |

### CL11 — Spark Remnic backup/restore bundle — staged off-repo

| Artifact | State |
|---|---|
| backup + restore scripts with dry-run checksums and verify logs | exists, staged off-repo |
| a committed path in this repo | missing |

### CL12 — two-node-proof one-command wrapper — merged

| Artifact | State |
|---|---|
| `receipts/two-node-proof-kit/` wrapper, `kit.sha256`, `check-proof.py`, `reference-combine.json` (PRs #135, #145) | merged |

### CL13 — Omarchy ARM64 install on the Spark — pending the physical step

| Artifact | State |
|---|---|
| install + rollback runbook | exists, staged off-repo |
| SSH + boot receipt on Omarchy ARM64 with the DGX OS fallback entry | missing |

### CL14 — Milestone 2 validation (Omarchy on Spark) — blocked on CL13

| Artifact | State |
|---|---|
| `docs/omarchy-spark-runbook.md` | missing |
| recovery drill + live Remnic service receipts | missing |

### CL15 — Milestone 4 both-nodes receipts — claimed, never executed

| Receipt path | State |
|---|---|
| receipt dir comparing Omarchy-on-both against CL04-CL06 across all 8 metrics | missing |
| joint-request `run.json` and two-node `combine.json` arms | missing |

### CL16 — live node-loss and restart drills — code merged + hardened, drill not run

| Artifact | State |
|---|---|
| `omarchy_cluster/drill.py` + `tests/test_drill.py` (PR #138) | merged |
| node-loss/restart drill hardening + `cli.py` rewiring and CLI tests (PR #157) | merged |
| live drill receipt on the Omarchy-on-both cluster | missing (needs CL15) |
| [unsupported-cases matrix](docs/unsupported-cases.md) (the former `docs/mac-spark-proof.md` was removed in PR #143) | merged |

## License

MIT.
