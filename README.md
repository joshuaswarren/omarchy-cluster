# omarchy-cluster

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink)](https://github.com/sponsors/joshuaswarren)

omarchy-cluster runs one local model across the machines on your network: Macs
on macOS, Macs on Omarchy Linux, even an iPhone. Any manufacturer, any OS, in
one run. The machines measure each other, decide which node runs which layers,
and serve the cluster as one OpenAI-compatible endpoint.

The point is the hardware you already own. Omarchy's promise has always been
that the machine in front of you is still great: install Omarchy Linux and give
it a second life. omarchy-cluster carries that promise to local models. The
MacBook you wrote off as too old for modern models cannot hold a 106B model. It
can hold 15 of its layers while a bigger machine holds the other 31. Use the
machines you own together and you can run models too big for any one of them.
Even a split too slow for chat can keep working on your coding projects in the
background.

On real hardware, 2026-10-07: GLM-5.3-Flash (156.8 GB, more than any one of
these machines holds) split across four Macs. A 128 GB Mac Studio on macOS
(Metal) took 14 layers. Two MacBook Pros on Omarchy Linux (Vulkan), an M2 Max
and an M1 Max, took 17 and 12. A 13-inch M1 MacBook Pro on Omarchy Linux (CPU)
took the last 3 and hosted the endpoint. It decoded at 1.39 and 1.57 tokens per
second, and two greedy requests returned the same text. Loading took 1055
seconds over 2.5 Gb Ethernet.

On 2026-10-06: GLM-4.5-Air (106B parameters, 60.1 GB at 4 bits) split across
the Mac Studio (31 layers) and the M1 Max (15 layers) over Thunderbolt and ran
at 0.87 tokens per second, with both machines agreeing on every token. Two
Omarchy Linux Macs split a smaller model and produced byte-identical text to
one machine alone. An iPhone 15 Pro Max served layers to an Omarchy Linux host
as a llama.cpp node. A three node probe measured every pair, including Linux
pairs, and pinned the fastest routes in 42.6 seconds.

A split does not make a model faster than one machine that can hold it. For a
model that fits on the Mac Studio, the Mac Studio alone is faster (see the
second table). Use a split when the model is too big for any one machine you
have. GLM-5.3-Flash fits on none of these four. GLM-4.5-Air fits on the idle
Mac Studio by itself; in that run the Mac Studio was also serving other models
and had 50 to 66 GB free, and the split left at least 22 percent of its memory
free.

exo, the closest comparison, runs its Linux nodes on CPU; omarchy-cluster runs
Vulkan on Omarchy Linux. See the FAQ below.

Try it on a node (Omarchy Linux; macOS install below):

```sh
python3 -m venv ~/.local/share/omarchy-cluster/venv && ~/.local/share/omarchy-cluster/venv/bin/pip install git+https://github.com/joshuaswarren/omarchy-cluster.git && ~/.local/share/omarchy-cluster/venv/bin/omarchy-cluster install-agent
```

Then connect two machines and serve a model: Quickstart below. The
GLM-5.3-Flash run is recorded in
[receipts/2026-10-07-glm53-flash-4node](receipts/2026-10-07-glm53-flash-4node)
and the GLM-4.5-Air run in [docs/demo.cast](docs/demo.cast)
(`asciinema play docs/demo.cast`).

## Measured on real hardware

All runs greedy (temperature 0), prompt "Write a haiku about shared memory.".
"Agree" means rank 1 logged the same text SHA-256 as the text the gateway
returned. "Same text" means two requests returned byte-identical text.

Capacity: models that do not fit on the Omarchy machines.

| Model | Weights | Nodes | Layers per node | Decode | Prompt time |
|---|---|---|---|---|---|
| GLM-5.3-Flash UD-IQ4_XS (llama.cpp RPC, 46 layers), 2026-10-07 | 156.8 GB | Mac Studio M1 Ultra 128 GB (Metal) + MacBook Pro M2 Max + MacBook Pro M1 Max (Omarchy, Vulkan) + MacBook Pro M1 13-inch (Omarchy, CPU), 2.5 Gb Ethernet | 14 + 17 + 12 + 3 | 1.39 and 1.57 tok/s, 64 tokens, same text | 13.6 s, 2.7 s (prompt cache) |
| GLM-4.5-Air-4bit (glm4_moe, 46 layers), 2026-10-06 | 60.1 GB | Mac Studio M1 Ultra 128 GB + MacBook Pro M1 Max 64 GB, Thunderbolt | 31 + 15 | 0.87 and 0.86 tok/s, 64 tokens, agree | 15.5 s, 13.2 s, 16.1 s |

The GLM-4.5-Air recording shows 0.83 tok/s. Per token, the laptop's 15 layers
take about 1.1 s and the Mac Studio's 31 layers about 45 ms. The omarchy-mlx
Vulkan path runs a GLM-4.5-Air layer at about 74 ms and Metal at about 1.5 ms,
so the Omarchy machine sets the speed.

Speed: DeepSeek-Coder-V2-Lite-Instruct-4bit (27 layers, fits on one machine).
Short names below: the Studio is the Mac Studio M1 Ultra 128 GB (macOS,
Metal). The laptops are MacBook Pros on Omarchy Linux (Vulkan).

| Machines | Link | Layers per node | Decode |
|---|---|---|---|
| Studio alone | | 27 | 68.2 and 67.6 tok/s |
| M1 Max alone | | 27 | 2.45 tok/s (earlier omarchy-mlx build) |
| M1 16 GB + M1 Max | 2.4 Gb/s wired | 1 + 26, chosen automatically | 1.79 and 1.80 tok/s, agree |
| Studio + M1 Max | Thunderbolt, 12 Gb/s | 26 + 1, chosen automatically | 46.4 and 48.0 tok/s, agree |
| Studio + M2 Max, 2026-10-05 | 2.2 Gb/s wired | 26 + 1, chosen automatically | median 52.5 tok/s (52.2 to 55.4), agree |
| same two machines | same | 14 + 13, a naive even split | 5.6 to 9.6 tok/s |
| same two machines | same | 24 + 3 | 28.6 to 31.0 tok/s |

The ring exchange between machines costs under 0.5 ms per token. Almost all of
a token's time is the layers on the slower node, which is why the planner puts
as few layers there as memory allows.

Both ranks always agree on tokens, because only rank 0 samples and sends each
token to the other rank. A split on one backend gives the same text as one
machine. The two Omarchy laptops produced byte-identical text to the M1 Max
alone. The text of a Metal plus Vulkan split is not always the text of a single
Mac. Hidden states differ between the Metal and Vulkan paths by up to 1 bf16
ulp on the prompt we checked. That can flip a near-tie token. In the runs above
the Mac Studio alone wrote "Threads in code, ...", the split wrote "Threads in
harmony, ...". Each backend is deterministic run to run.

Other GPU work on a Mac (another model server, a browser, a chat app) can slow
that node's decode several-fold. The contention watch in `GET /status` reports
it. During this release a model server on the Mac Studio cut the small-model
split from about 46 to about 22 tok/s while it was busy.

## What it does

The `discover` command finds nodes over mDNS (`_omarchy-cluster._tcp`, Avahi
on Linux, dns-sd on macOS), with a plain-text hosts file override.

The `probe` command measures RTT and bandwidth for every node pair and every
interface route, and pins the fastest decode-eligible route per pair, including
Linux pairs. Wi-Fi and overlay routes (Tailscale) are probed but never used for
decode. Each node's agent also reports whether MCDMA (RDMA) is available. `probe`
records for every pair whether MCDMA could carry its traffic. Routes still use
TCP. On the machines measured here, every node reports MCDMA as unavailable:
RDMA is off on the Mac, and the Linux nodes have no InfiniBand device.

The `place MODEL` command checks that the model fits: every stage must hold at
least one layer in 90 percent of its free memory, and all stages together must
hold every layer. It prints each node's layer cap.

The `serve MODEL` command plans, launches a pipeline rank on each node through
its token-authed agent (no node-to-node ssh), and starts an OpenAI-compatible
gateway on :8020. Each rank measures its decode ms per layer at start, and all
ranks agree on the same split; `--split N0,N1` sets it by hand. A request with
`"timing": true` reports per-step ring wait and compute.

The `--engine llamacpp` option serves a GGUF model through llama-server on the
gateway machine and spreads its layers over llama.cpp RPC servers. Each
`--rpc-node NAME=GB` has that node's agent start an RPC server: Metal on macOS,
Vulkan or CPU on Linux. The layers split in proportion to those budgets.
`--rpc-node HOST:PORT=GB` adds an RPC server started some other way, such as an
iPhone over USB or a CUDA machine. `--host-layers N` keeps the first N layers on
the gateway machine's CPU, read from the GGUF on disk. That runs a model a
little bigger than all the RPC nodes together.

The `stop` command stops the gateway and every rank, and sweeps the ports. The
`status` and `hub` commands print the node table with pinned routes and a 1 Hz
heartbeat hub.

The control plane is Python stdlib only. MLX is needed only on nodes that run
layers.

## Requirements

Macs on macOS need stock MLX and mlx-lm. Macs on Omarchy Linux need Omarchy M+
with omarchy-mlx, MLX on the GPU through its Vulkan backend. That Vulkan
backend runs 4-bit and 8-bit quantized models; 5-bit and 6-bit MoE models fail
on these nodes. The llama.cpp engine (`--engine llamacpp`) needs llama.cpp on
the gateway machine. Each RPC node needs `ggml-rpc-server`, built with
`-DGGML_RPC=ON` at the same llama.cpp version. An iPhone node needs the
rpc-server app built from `ios/` (see `ios/README.md`). Python 3.9 or newer is needed on every node. Avahi on Linux
and dns-sd on macOS are present by default on both. All nodes sit on one LAN or
Thunderbolt link.

The model must be one mlx-lm can pipeline: the deepseek_v2/v3, glm4_moe,
glm4_moe_lite and ministral3 families. qwen2/qwen3 are not pipeline models
upstream; run those on one node (`--stages 1`).

## Install

One command per node. Use a venv that persists: the agent starts from it at
every boot, so a venv under /tmp breaks after a reboot (`install-agent` refuses
one).

On Omarchy. The agent is plain Python; ranks run in omarchy-mlx's Python,
whether omarchy-mlx is a user or a package install:

```sh
python3 -m venv ~/.local/share/omarchy-cluster/venv && ~/.local/share/omarchy-cluster/venv/bin/pip install git+https://github.com/joshuaswarren/omarchy-cluster.git && ~/.local/share/omarchy-cluster/venv/bin/omarchy-cluster install-agent
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

All nodes need the same token. Copy it to the second node:

```sh
scp ~/.config/omarchy-cluster/token node2:~/.config/omarchy-cluster/token
ssh node2 omarchy-cluster install-agent --token-file ~/.config/omarchy-cluster/token
```

Then, on the machine that will run the gateway:

```sh
omarchy-cluster discover        # both nodes appear
omarchy-cluster probe           # measure routes, pin the fastest
omarchy-cluster status          # chip, memory, backend
omarchy-cluster place mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
omarchy-cluster serve mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
```

`serve` prints the plan and starts the ranks through each node's agent, and it
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

Stop everything with `omarchy-cluster stop`. Logs are in
`~/.local/share/omarchy-cluster/`: `gateway.log` on the gateway machine,
`rank0.log` and `rank1.log` on each node.

For a model bigger than one node, pick the split yourself so a machine you use
for other work keeps free memory, for example `serve
mlx-community/GLM-4.5-Air-4bit --split 31,15` (rank 0 gets 31 layers).

For a GGUF model over more than two machines, use the llama.cpp engine. Only
the gateway machine needs the GGUF file. Each RPC server receives its layers
over the network at load and caches them for the next load:

```sh
omarchy-cluster serve ~/models/model-00001-of-00004.gguf --engine llamacpp \
  --rpc-node mac-studio=50 --rpc-node linux-a=57 --rpc-node linux-b=8 \
  --rpc-binary linux-a=$HOME/src/llama.cpp/build/bin/ggml-rpc-server
```

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
model's layers in a child process. It keeps the fastest ms per layer it has
seen on that node. Every rank runs the same chooser: one layer per rank, the
rest on the cheapest ranks up to their memory caps. A burst of other GPU work
cannot steal the split, because a stored measurement is only ever replaced by a
faster one. Delete `~/.local/state/omarchy-cluster/layer-ms.json` after a real
slowdown.

## Limits

The MLX engine runs two ranks: the planner can plan more stages, but `serve`
runs two. The llama.cpp engine takes any number of RPC nodes. One request at a
time: concurrent requests queue, and they add no throughput.
Pipeline-model families only (see Requirements). `place` sizes stages from free
memory, not from the Vulkan allocation limit. On a 16 GB M1 on Omarchy, Vulkan
allocations failed at about 8.5 GB, so DeepSeek-Coder-V2-Lite-4bit did not load
there whole. Give such a node fewer layers with `--split`. Plain HTTP with one
shared token on your LAN: a homelab trust model, no TLS, no per-user auth. A
node slowed by other GPU work is detected and reported, not preempted.

## How is this different from exo?

exo is the closest project, and strong at what it targets: Macs clustered with
MLX on Metal, RDMA over Thunderbolt 5, automatic discovery, tensor parallelism.
On GPU support its README states: "On macOS, exo uses the GPU. On Linux, exo
currently runs on CPU."
(https://github.com/exo-explore/exo#hardware-accelerator-support)

omarchy-cluster runs Metal on macOS and Vulkan on Omarchy Linux, and its probe
pins the fastest route on every pair, Linux pairs included. Proven on this
hardware: a Mac Studio on macOS and a MacBook Pro on Omarchy Linux run one 106B
model together. The split was 31 + 15 layers at 0.87 tok/s, with every token
agreeing. Two Omarchy Linux Macs split a model with byte-identical text to one
machine.

## How is this different from MCDMA or TensorFold?

They work at different layers. MCDMA is a transport. Its README says "MCDMA
provides the RDMA driver and verbs transport." and "Installing MCDMA alone does
not connect an inference engine to this path."
(https://github.com/ashhart/MCDMA)

TensorFold is a server. Its README says "TensorFold serves language models on
Apple Silicon and NVIDIA GPUs through an OpenAI-compatible API." For
GLM-5.3-Flash it lists "MLX on a 256 GB Mac, CUDA with two ranks".
(https://github.com/ashhart/TensorFold)

omarchy-cluster is the layer above both. It spreads one model over several
machines that each hold only part of it. It serves them as one endpoint. Today
it moves data over TCP on the fastest probed route. MCDMA would be a faster
transport under it. Each agent already detects MCDMA and `probe` records which
pairs could use it.

## An iPhone as a node (llama.cpp RPC)

This is not part of the MLX pipeline above. It uses llama.cpp's RPC backend,
through `omarchy-cluster serve MODEL --engine llamacpp`. The app source and its
build live under `ios/` (see `ios/README.md`).

An iPhone 15 Pro Max (A17 Pro, 8 GB, iOS 27) ran llama.cpp's `rpc-server` as an
app. An M1 Max laptop on Omarchy Linux sent model layers to it over USB. The
app was built, signed and installed from Omarchy Linux, with no Mac and no
Xcode. The phone GPU runs through Metal; the shader source ships inside the app
and the phone compiles it at start.

The phone is a capacity node, not a speedup. Each phone layer costs about 0.8
ms per token, against about 0.5 ms on the laptop CPU, so no split beats the
laptop alone for a model the laptop holds. Prefill is the phone's strength:
with all 28 layers of Qwen3-1.7B on the phone GPU, it processes a 128-token
prompt at 424 tok/s, faster than the laptop CPU (349 tok/s).

Qwen3-1.7B Q4_K_M, 64-token decode, lm_head on the laptop:

| Setup | Decode tok/s |
|---|---:|
| Laptop alone (CPU) | 71.1 |
| Phone GPU, 7 of 28 layers on the phone | 35.5 |
| Phone GPU, 14 layers | 31.6 |
| Phone GPU, 14 layers, `OMP_WAIT_POLICY=ACTIVE` (separate run) | 45.0 |
| Phone GPU, 28 layers | 21.1 |
| Phone GPU, 28 layers, `OMP_WAIT_POLICY=ACTIVE` (separate run) | 35.1 |
| Phone CPU, 14 layers | 30.0 |

The lm_head stayed on the laptop in every row. With all 28 layers on the phone,
a token costs about 23 ms on the phone and about 7 ms on the laptop. The phone
part reads about 1 GB of weights per token, close to the phone's memory
bandwidth.

With the phone on its CPU, greedy output is byte-identical to the laptop alone
at 7, 14 and 28 phone layers. On the phone GPU, output matches at 7 layers. At
full offload, the mean KL divergence against the laptop is 0.008 and the top
token agrees 96 percent of the time.

About 2.2 GB of layers ran well on the 8 GB phone; 4.4 GB pushed iOS into
memory pressure. Speculative decoding with the draft model on the phone was
slower (6.6 tok/s) than with the draft on the laptop (25.1 tok/s).

Details, sources and raw data: `receipts/2026-10-06-iphone-rpc-node/`.

## Support

Every bit of support helps keep omarchy-cluster alive and free. If you are able, [sponsor on GitHub](https://github.com/sponsors/joshuaswarren) or send a Lightning donation to `joshuaswarren@strike.me` to directly fund continued development and new integrations.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?style=for-the-badge)](https://github.com/sponsors/joshuaswarren)

If financial support is not an option, you can still make a big difference: [star the repo](https://github.com/joshuaswarren/omarchy-cluster), share it, or recommend it to a colleague. Word of mouth is how most people find omarchy-cluster.

## Contributing

PRs welcome. The tests need no GPU or network:

```sh
python3 -m pytest tests/
```

54 tests, stdlib only.

## License

MIT.