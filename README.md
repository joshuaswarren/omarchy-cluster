<!-- UNVOICED: draft wording, voice pass pending before publication. -->
# omarchy-cluster

Run one local MLX model across the Apple Silicon machines on your network: a
Mac on macOS and Macs on Omarchy Linux, in any mix. The machines measure each
other, decide which node runs which layers, run mlx-lm's pipeline in lockstep,
and serve the whole cluster as one OpenAI-compatible endpoint.

The point is memory. GLM-4.5-Air (106B parameters, 60.1 GB at 4 bits) does not
fit on a 64 GB MacBook Pro M1 Max running Omarchy. Split across that laptop and
a Mac Studio on macOS, it runs: 15 layers on the laptop (Vulkan), 31 on the Mac
Studio (Metal), 0.87 tokens per second, with both machines agreeing on every
token. The Mac Studio has 128 GB and could hold this model alone if it were
idle; in this run it was also serving other models and had 50 to 66 GB free,
and the split left at least 22 percent of its memory free.

A split does not make a model faster than one machine that can hold it. For a
model that fits on the Mac Studio, the Mac Studio alone is faster (see the
second table). Use a split when the model is too big for any one machine you
have.

Recording of the GLM-4.5-Air run: [docs/demo.cast](docs/demo.cast)
(`asciinema play docs/demo.cast`).

## Measured on real hardware

All runs greedy (temperature 0), prompt "Write a haiku about shared memory.",
2026-10-06. "Agree" means rank 1 logged the same text SHA-256 as the text the
gateway returned.

Capacity: a model that does not fit on the Linux machine.

| Model | Weights | Nodes | Layers per node | Decode | Prefill, 13 tokens |
|---|---|---|---|---|---|
| GLM-4.5-Air-4bit (glm4_moe, 46 layers) | 60.1 GB | Mac Studio M1 Ultra 128 GB (macOS, Metal) + MacBook Pro M1 Max 64 GB (Omarchy, Vulkan), Thunderbolt | 31 + 15 | 0.87 and 0.86 tok/s, 64 tokens, agree; 0.83 in the recording | 15.5 s, 13.2 s, 16.1 s |

Per token, the laptop's 15 layers take about 1.1 s and the Mac Studio's 31
layers about 45 ms. Today the omarchy-mlx Vulkan path runs a GLM-4.5-Air layer
at about 74 ms and Metal at about 1.5 ms, so the Linux machine sets the speed.

Speed: DeepSeek-Coder-V2-Lite-Instruct-4bit (27 layers, fits on one machine).

| Nodes | Link | Layers per node | Decode |
|---|---|---|---|
| Mac Studio M1 Ultra alone (macOS, Metal) | | 27 | 68.2 and 67.6 tok/s |
| MacBook Pro M1 Max alone (Omarchy, Vulkan) | | 27 | 2.45 tok/s (earlier omarchy-mlx build) |
| Mac Studio + MacBook Pro M1 Max (Omarchy, Vulkan) | Thunderbolt, 12 Gb/s | 26 + 1, chosen automatically | 46.4 and 48.0 tok/s, agree |
| Mac Studio + MacBook Pro M2 Max (Omarchy, Vulkan), 2026-10-05 | 2.2 Gb/s wired | 26 + 1, chosen automatically | median 52.5 tok/s (52.2 to 55.4), agree |
| same two machines | same | 14 + 13 (naive even split) | 5.6 to 9.6 tok/s |
| same two machines | same | 24 + 3 | 28.6 to 31.0 tok/s |

The ring exchange between machines costs under 0.5 ms per token. Almost all of
a token's time is the layers on the slower node, which is why the planner puts
as few layers there as memory allows.

Tokens: both ranks always agree, because only rank 0 samples and sends each
token to the other rank. The text of a Metal plus Vulkan split is not always
the text of a single Mac: hidden states differ between the Metal and Vulkan
paths by up to 1 bf16 ulp on the prompt we checked, and that can flip a
near-tie token. In the runs
above the Mac Studio alone wrote "Threads in code, ...", the split wrote
"Threads in harmony, ...". Each backend is deterministic run to run.

macOS note: other GPU work on a Mac (another model server, a browser, a chat
app) can slow that node's decode several-fold. The contention watch in
`GET /status` reports it. During this release a model server on the Mac Studio
cut the small-model split from about 46 to about 22 tok/s while it was busy.

## What it does

- `omarchy-cluster discover`: find nodes over mDNS (`_omarchy-cluster._tcp`,
  Avahi on Linux, dns-sd on macOS), with a plain-text hosts file override.
- `omarchy-cluster probe`: measure RTT and bandwidth for every node pair and
  every interface route, and pin the fastest decode-eligible route per pair.
  Wi-Fi and overlay routes (Tailscale) are probed but never used for decode.
- `omarchy-cluster place MODEL`: check the model fits: every stage must hold
  at least one layer in 90 percent of its free memory, and all stages together
  must hold every layer. Prints each node's layer cap.
- `omarchy-cluster serve MODEL`: plan, launch a pipeline rank on each node
  through its token-authed agent (no node-to-node ssh), and start an
  OpenAI-compatible gateway on :8020. Each rank measures its decode ms per
  layer at start and all ranks agree on the same split, or `--split N0,N1`
  sets it. A request with `"timing": true` reports per-step ring wait and
  compute.
- `omarchy-cluster stop`: stop the gateway and every rank, sweep the ports.
- `omarchy-cluster status`, `hub`: node table with pinned routes, and a 1 Hz
  heartbeat hub.

The control plane is Python stdlib only. MLX is needed only on nodes that run
layers.

## Requirements

- Macs on macOS: stock MLX and mlx-lm.
- Macs on Linux: Omarchy M+ with omarchy-mlx (MLX on the GPU through its
  Vulkan backend). Its Vulkan backend runs 4-bit and 8-bit quantized models;
  5-bit and 6-bit MoE models fail on Linux nodes today.
- Python 3.9 or newer.
- Avahi on Linux and dns-sd on macOS (present by default on both).
- All nodes on one LAN or Thunderbolt link.
- A model mlx-lm can pipeline: the deepseek_v2/v3, glm4_moe, glm4_moe_lite and
  ministral3 families. qwen2/qwen3 are not pipeline models upstream; run those
  on one node (`--stages 1`).

## Install

One command per node, into the Python that has MLX.

On Omarchy, into the omarchy-mlx environment:

```sh
~/.local/share/mlx-omarchy/venv/bin/pip install git+https://github.com/joshuaswarren/omarchy-cluster.git && ~/.local/share/mlx-omarchy/venv/bin/omarchy-cluster install-agent
```

On macOS:

```sh
python3 -m venv ~/mlx && ~/mlx/bin/pip install mlx mlx-lm git+https://github.com/joshuaswarren/omarchy-cluster.git && ~/mlx/bin/omarchy-cluster install-agent
```

`install-agent` starts the node agent on :8025 and writes a token to
`~/.config/omarchy-cluster/token`. Each agent starts ranks with its own node's
MLX Python: `$OMARCHY_CLUSTER_PYTHON` if set, else the omarchy-mlx venv if
present, else the Python the agent runs in. `serve --python-mac` and
`--python-linux` override that.

## Quickstart: two machines

All nodes need the same token. From the machine that will run the gateway:

```sh
scp ~/.config/omarchy-cluster/token node2:~/.config/omarchy-cluster/token
ssh node2 omarchy-cluster install-agent --token-file ~/.config/omarchy-cluster/token
```

Then, on the gateway machine:

```sh
omarchy-cluster discover        # both nodes appear
omarchy-cluster probe           # measure routes, pin the fastest
omarchy-cluster status          # chip, memory, backend
omarchy-cluster place mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
omarchy-cluster serve mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
```

`serve` prints the plan, starts the ranks through each node's agent, and
leaves the gateway on :8020. Each node loads only its own layers from its
Hugging Face cache, so download the model on every node first
(`hf download mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit`).

```sh
curl -s http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
       "messages":[{"role":"user","content":"Write a haiku about shared memory."}],
       "max_tokens":64,"temperature":0}'
```

Stop everything with `omarchy-cluster stop`. Logs: `~/.local/share/omarchy-cluster/`
(`gateway.log` on the gateway machine, `rank0.log`/`rank1.log` on each node).

For a model bigger than one node, pick the split yourself so a machine you use
for other work keeps free memory, for example
`serve mlx-community/GLM-4.5-Air-4bit --split 31,15` (rank 0 gets 31 layers).

## OpenAI-compatible endpoint

The gateway speaks chat completions on `http://gateway-node:8020/v1`. With
LiteLLM:

```yaml
model_list:
  - model_name: local-cluster
    litellm_params:
      model: openai/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
      api_base: http://gateway-node:8020/v1
      api_key: local
```

```sh
litellm --config litellm.yaml --port 4000
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-cluster","messages":[{"role":"user","content":"hi"}]}'
```

`GET /status` on :8020 shows the live split, the measured ms per layer and
contention events. Engine errors come back as HTTP 502 with the engine's
message.

## How the split is chosen

At rank start, each rank times one-token decode steps through 1 and 4 of the
model's layers in a child process and keeps the fastest ms per layer it has
seen on that node. Every rank runs the same chooser: one layer per rank, the
rest on the cheapest ranks up to their memory caps. A burst of other GPU work
cannot steal the split, because a stored measurement is only ever replaced by
a faster one (delete `~/.local/state/omarchy-cluster/layer-ms.json` after a
real slowdown).

## Limits

- Two ranks. The planner can plan more stages, but `serve` runs two.
- One request at a time. Concurrent requests queue; they add no throughput.
- Pipeline-model families only (see Requirements).
- `place` sizes stages from free memory, not from the Vulkan allocation limit.
  On a 16 GB M1 on Omarchy, Vulkan allocations failed at about 8.5 GB, so
  DeepSeek-Coder-V2-Lite-4bit did not load there whole. Give such a node fewer
  layers with `--split`.
- `probe` does not pin a route between two Linux nodes yet; pin it in
  `~/.local/state/omarchy-cluster/links.json` or use a Mac in the pair.
- Plain HTTP with one shared token on your LAN: a homelab trust model, no TLS,
  no per-user auth.
- A node slowed by other GPU work is detected and reported, not preempted.

## Development

```sh
python3 -m pytest tests/
```

37 tests, stdlib only, no GPU or network needed.

## License

MIT.
