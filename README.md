# omarchy-cluster

omarchy-cluster runs one local model across the machines on your network: Macs on
macOS, Macs on Omarchy M+ Linux, even an iPhone. The machines measure each other,
decide which node runs which layers, and serve the cluster as one
OpenAI-compatible endpoint on :8020.

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
pinned routes and 1 Hz liveness. `--engine llamacpp` spreads a GGUF over RPC servers with
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
- One request at a time. Concurrent requests queue and add no throughput.
- Sampling is greedy only: the gateway rejects requests with a temperature other
  than 0.
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

## Security

- The agent on :8025 requires the cluster token (HMAC-checked on every request);
  ranks are started only through the agent, so there is no node-to-node ssh.
- The token file is the cluster's secret. All nodes need the same token:

  ```sh
  scp ~/.config/omarchy-cluster/token node2:~/.config/omarchy-cluster/token
  ssh node2 'omarchy-cluster install-agent --token-file ~/.config/omarchy-cluster/token'
  ```

- Traffic is plain HTTP over your LAN: no TLS, no per-user auth. This is a homelab
  trust model. Run it on a network you trust.
- llama.cpp's `ggml-rpc-server` has no authentication. Bind it to one address on a
  network you trust (the engine does this when the agent starts it).

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

RPC servers have no authentication. Keep them on a trusted network and bind
them to the address you intend to expose.

## Documentation

- [Command reference](docs/cli.md): flags, gateway API and environment switches.
- [Placement](docs/placement.md): layer allocation, KV cache and ring addresses.
- [Troubleshooting](docs/troubleshooting.md): logs, memory limits and prompts.
- [Measured results](docs/benchmarks.md): benchmark tables and supporting records.

## Contributing

PRs welcome. The tests need pytest and local loopback sockets; they run without a
GPU, model downloads or access to other hosts.

```sh
python3 -m pip install pytest==8.4.2
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

CI runs these tests on Ubuntu 24.04 and macOS 15 with Python 3.9 and 3.13.

## License

MIT.
