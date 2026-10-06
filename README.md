# omarchy-cluster

Split one local MLX model across the Apple Silicon machines on your network: a
Mac on macOS plus Linux boxes on Omarchy, or any mix, over one LAN. The
machines measure each other, pick which node runs which layers, run mlx-lm's
pipeline in lockstep, and expose the whole cluster as one OpenAI-compatible
endpoint on the machine you choose.

On a 2.2 Gb/s wired LAN, two machines (an M1 Ultra desktop on macOS with
Metal, and an M2 Max laptop on Linux with Vulkan) run
DeepSeek-Coder-V2-Lite-Instruct-4bit at a median 52.5 tok/s, greedy decode.
The same two machines with a naive even layer split ran the same prompt and
model at 5.6 to 9.6 tok/s. The split choice is most of the speed, and the
tool picks it from measurements.

## What it does

- `omarchy-cluster discover` : find nodes over mDNS (`_omarchy-cluster._tcp`,
  Avahi on Linux, dns-sd on macOS), with a plain-text hosts file override.
- `omarchy-cluster probe` : measure RTT and bandwidth for every node pair and
  every interface route, pin the fastest decode-eligible route per pair.
  Wi-Fi and overlay routes (Tailscale) are probed but never used for decode
  ranks.
- `omarchy-cluster place MODEL` : check the model fits: every pipeline stage
  must hold at least one layer in 90 percent of free memory, and all stages
  together must hold every layer. Prints each node's layer cap.
- `omarchy-cluster serve MODEL` : plan, then launch a pipeline rank on each
  node through its token-authed agent (no node-to-node ssh), and start an
  OpenAI-compatible gateway on :8020. Each rank measures its decode ms per
  layer at start; all ranks agree on the same split, or `--split N0,N1`
  overrides it. A request with `"timing": true` reports per-step ring wait vs
  compute, and a contention watch flags a node slowed by other GPU work in
  `GET /status`.
- `omarchy-cluster stop` : stop the gateway and every rank, sweep the ports.
- `omarchy-cluster status`, `hub`: node table with pinned routes, and a
  1 Hz heartbeat hub for liveness.

The control plane is Python stdlib only; MLX itself is only needed on the
nodes that run model layers.

## Measured on real hardware

Model `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit` (27 layers),
prompt "Write a haiku about shared memory.", greedy, max_tokens 64, both
ranks agree on identical tokens:

| Nodes | Link | Split | Decode |
|---|---|---|---|
| M1 Ultra desktop (macOS, Metal) + M2 Max laptop (Linux, Vulkan) | 2.2 Gb/s wired | measured `[26,1]` | median 52.5, range 52.2 to 55.4 tok/s |
| same two machines | same | naive even `[14,13]` | 5.6 to 9.6 tok/s |
| same two machines | same | `[24,3]` | 28.6 to 31.0 tok/s |

Per decode step at `[26,1]`, the ring exchange between machines costs under
0.5 ms of a ~19 ms step; almost all the time is that slowest node's single
layer plus rank 0's 26 layers. That is why the split matters and why the
planner puts one layer on the slow node. Hidden-state numerics at layer 12
differ between the Metal and Vulkan paths by at most 1 bf16 ulp on the
checked prompt; both backends are bitwise deterministic run to run.

macOS note: a desktop app doing GPU work (a browser or chat app is enough)
can slow that node's decode several-fold mid-run. The contention watch in
`GET /status` shows when it happens, and the next start measures the split
again.

## Requirements

- Apple Silicon Macs on macOS: stock MLX (`pip install mlx mlx-lm`).
- Linux machines: Omarchy M+ with omarchy-mlx (MLX on the GPU through its
  Vulkan backend), with mlx-lm installed in the same environment.
- Python >= 3.9 (macOS system Python is fine for the control plane).
- Avahi on Linux, dns-sd on macOS (both present by default on Omarchy and
  macOS).
- All nodes on one LAN.
- A model that mlx-lm can pipeline: the deepseek_v2/v3, glm4_moe,
  glm4_moe_lite, and ministral3 families. qwen2/qwen3 are not pipeline models
  upstream; run those on a single node (`--stages 1`).

## Install

On every node:

```sh
pip install git+https://github.com/joshuaswarren/omarchy-cluster.git
```

Make sure `python3 -c "import mlx.core"` works in the Python that will run
the ranks. If MLX lives in a venv instead, point `serve` at it with
`--python-linux` and/or `--python-mac`.

## Quickstart: two machines in five minutes

On the second machine (the one that will run layers but not the gateway):

```sh
omarchy-cluster install-agent   # daemon on :8025, token at ~/.config/omarchy-cluster/token
```

On the first machine:

```sh
omarchy-cluster install-agent
# share one token so this machine may drive the other's agent
scp ~/.config/omarchy-cluster/token node2:~/.config/omarchy-cluster/token
ssh node2 omarchy-cluster install-agent --token-file ~/.config/omarchy-cluster/token
```

Then, still on the first machine:

```sh
omarchy-cluster discover                          # both nodes appear
omarchy-cluster probe                             # measure routes, pin the fastest
omarchy-cluster status                            # chip, memory, backend, up
omarchy-cluster place mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
omarchy-cluster serve mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
```

`serve` prints the plan, launches the ranks through each node's agent, waits
for both engines, and leaves the gateway on :8020. Try it:

```sh
curl -s http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
       "messages":[{"role":"user","content":"Write a haiku about shared memory."}],
       "max_tokens":64,"temperature":0}'
```

Stop everything with `omarchy-cluster stop`.

The first model download happens on the node that runs the ranks, through
whatever Python you gave `serve`.

## OpenAI-compatible endpoint

The gateway speaks chat completions on `http://gateway-node:8020/v1`, so
anything that talks OpenAI can use the cluster. With LiteLLM:

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

`GET /status` on :8020 shows the live split, calibrated ms/layer per node,
and any contention events.

## How the split is chosen

At rank start, each rank times 1-token decode steps through 1 and 4 of the
model's layers in a child process, keeps the fastest ms per layer it has ever
seen on that node, and every rank runs the same chooser: one layer per rank,
the remaining layers on the cheapest ranks up to their memory caps. A burst
of other GPU work on one node cannot steal the split, because the stored
fastest measurement is never replaced by a slower one (delete
`~/.local/state/omarchy-cluster/layer-ms.json` after a real slowdown).

## Limits

- Two ranks are exercised end to end. The planner can plan more stages, but
  more than two is not proven.
- Pipeline-model families only (see Requirements).
- Plain HTTP with one shared token on your LAN. This is a homelab trust
  model, not a multi-tenant service: no TLS, no per-user auth.
- World size above 2 is not implemented in the rank path.
- A node slowed by other GPU work is detected and reported, not preempted.

## Development

```sh
python3 -m pytest tests/
```

32 tests, stdlib only, no GPU or network needed.

## License

MIT.
