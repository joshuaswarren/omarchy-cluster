# Command reference

All commands are `omarchy-cluster COMMAND` (the package installs one entry point,
`omarchy_cluster.cli:main`). The control plane is Python stdlib only; MLX is needed
only on nodes that run layers. Run `omarchy-cluster --help` for the same list.

Files the tools read and write:

| Path | Purpose |
|---|---|
| `~/.config/omarchy-cluster/token` | the cluster token agents require (HMAC-compared on every request) and remote gateway callers send as a bearer token |
| `~/.local/share/omarchy-cluster/` | `agent.log`, `gateway.log`, `rankN.log`, the installed source copy |
| `~/.local/state/omarchy-cluster/links.json` | probe's measured routes (override with `--links` / `--out`) |
| `~/.local/state/omarchy-cluster/layer-ms.json` | fastest decode ms/layer seen per node |
| `~/.local/state/omarchy-cluster/live-env` | default `OMARCHY_LIVE_ENV` file |

## Node commands

### `install-agent [--port 8025] [--token-file PATH]`
Installs and starts the node agent as a boot-persistent daemon. Refuses a venv under
`/tmp` (it would break after a reboot). Writes the token next to the agent config.

### `bootstrap user@host [user@host ...] [--port 8025] [--token-file PATH]`
Sets up new nodes over ssh (keys only, `BatchMode`). Per node, in order: detect the OS
(Linux or macOS), check `python3` is 3.9 or newer, install Avahi on Linux when missing
(`sudo -n`; without passwordless sudo it prints the command to run), copy this package
to `~/.local/share/omarchy-cluster/src`, copy the cluster token (this machine's, or
`--token-file`), run `install-agent`, wait for `/v1/health`, and print the node's
`discover` name. A failing node is reported and the rest still run; the exit status is 1
if any failed. It overwrites the node's token file with the token it sends. It installs
no MLX.

### `agent [--port 8025] [--name NAME] [--hub URL]`
Run the agent in the foreground. Serves facts, rank start/stop and probe helpers on
its port; every request needs the cluster token. With `--hub`, POSTs a 1 s heartbeat
to the hub. The agent starts ranks with its own node's MLX Python:
`$OMARCHY_CLUSTER_PYTHON` if set, else the omarchy-mlx venv if present, else the
Python the agent runs in.

### `discover [--mdns-timeout 4.0]`
Lists nodes found over mDNS (`_omarchy-cluster._tcp`, Avahi on Linux, dns-sd on
macOS) plus any plain-text hosts-file override.

### `probe [--seconds 3.0] [--out PATH]`
Measures RTT and bandwidth for every node pair and every interface route, writes
`links.json`, and pins the fastest decode-eligible route per pair. Wi-Fi and overlay
routes (Tailscale CGNAT range) are probed but never used for decode. Records for each
pair whether MCDMA (RDMA) could carry its traffic; routes stay TCP until MCDMA is
measured. A software verbs device (Soft-RoCE `rxe`, `siw`) is flagged
`soft_transport: true`; numbers over that device measure the kernel network stack.

Each route also records `thunderbolt` (both ends are Thunderbolt interfaces: a
`thunderbolt-net` device on the Thunderbolt bus on Linux, a "Thunderbolt" hardware port
on macOS) and `rdma_capable` (both ends sit on an active hardware RDMA port: a netdev
behind an ACTIVE non-`rxe`/`siw` port in `/sys/class/infiniband` on Linux, an
`rdma_en*` interface with `rdma_ctl` enabled on macOS). These two flags only label the
route. The pinned route is still the fastest measured decode-eligible one.

### `status [--links PATH] [--hub URL]`
Node table: OS, chip, memory, backend, up-state, IP, pinned routes, with hub
liveness when `--hub` is given. Routes that carry the flags print `TB` and `RDMA` tags,
and each pair says whether RDMA is available and why
not (for example `rdma_ctl status: disabled`).

### `hub [--port 8030]`
The 1 Hz heartbeat hub agents POST to; answers node liveness. Open `http://HUB:8030/` in
a browser for the cluster view: one card per node (chip, memory used, load, up or out)
and the pinned route of each probed pair as a labelled edge with its measured Gb/s. The
page polls `GET /v1/topology` once a second; that JSON holds the heartbeat nodes, each
with the `facts` its agent sent (chip, OS, memory, load, cores), and the pinned links
from the hub machine's `links.json`. Like `/v1/nodes`, it needs no token. A node
appears when its agent runs with `--hub`.

### `stop`
Stops the gateway and every rank it started (locally and through each agent) and
sweeps the ports.

### `guard [--interval 10.0]`
Watches every node of a running serve through its agent and stops the run when a
Linux node drops under 2 GB available, a Mac's memory pressure falls under 15 percent
free or its Data volume under 20 GB, or a node stops answering. Run it on the gateway
machine. See [troubleshooting.md](troubleshooting.md#logs-and-stopping).

## Model commands

### `place MODEL [--ctx 2048] [--links PATH] [--no-decode NAME]... [--nodes N1,N2] [--stages N] [--json]`
Checks the model fits: every stage must hold at least one layer in 90 percent of its
free memory, and all stages together must hold every layer. Prints each node's layer
cap and the per-boundary route cost. Model sizing details and KV reservations:
[placement.md](placement.md).

### `serve MODEL [options]`
Plans, launches a pipeline rank on each node through its token-authed agent (no
node-to-node ssh), and starts the OpenAI-compatible gateway on `:8020`. Each rank
measures its decode ms per layer at start and all ranks agree on the same split.

Engine selection:

| Flag | Meaning |
|---|---|
| `--engine mlx` (default) | pipeline the MLX model across ranks |
| `--engine llamacpp` | GGUF through llama-server on the gateway, layers spread over llama.cpp RPC servers |
| `--engine sushi` | single-host Sushi-format model (`EXL3` experts); fronts a stand-alone `sushi serve` with `--sushi`, `--sushi-model`, `--sushi-url`, `--sushi-port`, `--sushi-arg` |

MLX pipeline flags:

| Flag | Meaning |
|---|---|
| `--split N0,N1[,...]` | layers per rank, rank order; rank 0 samples and runs the last layers |
| `--nodes N1,N2[,...]` | with `--split`: the nodes in rank order, any number of ranks, no links file or placement search |
| `--ring-ip NAME=IP` | address neighbours use to reach a node (repeat) |
| `--rank-env NAME=VALUE` | environment variable for every rank (repeat) |
| `--python-mac` / `--python-linux` | rank interpreter per OS |
| `--rank-pythonpath PATH` | extra `PYTHONPATH` for ranks |
| `--stages N` | ask the planner for N stages |
| `--no-decode NAME` | never put the decode tail on this node (repeat) |
| `--no-fit-check` | skip the 90 percent free-memory fit check |
| `--gpu-turn MINUTES` | wrap Linux ranks in `~/bin/gpu-turn` (a site-specific GPU-queue wrapper, Linux nodes only) |

llama.cpp flags (`--engine llamacpp`):

| Flag | Meaning |
|---|---|
| `--llama-server PATH` | llama-server binary (default `llama-server`) |
| `--rpc-node NAME` / `NAME=GB` / `HOST:PORT` / `HOST:PORT=GB` | an RPC node started by its agent, with an optional GB budget, or one started some other way |
| `--rpc-node-port 50060` | agent-started rpc-server port per node |
| `--rpc-binary NAME=PATH` | rpc-server binary for a node |
| `--rpc-env NAME=KEY=VALUE` | environment variable for that node's RPC server |
| `--rpc-threads NAME=N` | CPU threads for that node's RPC server |
| `--rpc-cache` | RPC nodes keep received layers on disk for the next load |
| `--host-layers N` | keep exactly N layers on the gateway machine, nodes in the order given |
| `--llama-arg=ARG` | extra llama-server argument (repeat). Write the `=`: an argument starting with `-` is otherwise read as a `serve` option. Example: `--llama-arg=--no-warmup --llama-arg=-fa --llama-arg=on` |
| `--max-concurrent N` | requests the gateway passes to the engine at once (default 4); more wait up to 10 minutes, then get HTTP 503 with `Retry-After`. llama.cpp batches them in its slots; the MLX ring runs one at a time |

iPhone node flags: `--ios-bundle`, `--pm3`, `--phone-cap-gb`, `--rpc-layers`,
`--rpc-port`. See [ios/README.md](../ios/README.md).

Gateway flags: `--port 8020`, `--engine-port 8031`, `--ctx 2048`, `--links`,
`--no-decode`. The engine starts llama-server on port `engine-port + 1` (8032 by
default).

### `chat [--gateway URL] [--model ID] [--max-tokens 256] [--system TEXT]`

A terminal chat with a running `serve`. It sends the whole conversation to
`POST /v1/chat/completions` on the gateway (default `http://127.0.0.1:8020`)
with the cluster token if this machine has one, prints each reply, and then the
gateway's `tokens_per_s` when the engine reports it. Replies are not streamed.
`/reset` clears the history; `/exit`, `/quit` or end of input leave. A request
that fails prints the error and leaves the conversation as it was.

### `app`

Opens the native GTK 4 model browser and chat window. Refresh lists models cached
on the selected gateway; enter a Hugging Face repository to download one, select
it, then chat. Requires GTK 4 and PyGObject on the desktop machine. The desktop
launcher is installed in `share/applications`.

## Gateway API

- `POST /v1/chat/completions` serves OpenAI chat completions; the MLX engine applies the
  model's chat template to `messages`. Add `"timing": true` for per-step ring wait
  and compute times. Sampling is greedy only: a request with `temperature` other
  than 0 gets HTTP 400 (`gateway serves temperature 0 only`).
- `POST /v1/completions` with `"raw": true` and a `prompt` string skips the template:
  the engine tokenizes exactly that text, without adding special tokens, and both
  ranks get the same ids. The engine adds no beginning-of-text token, so a raw prompt
  must carry its own (a Muse Spark Open prompt begins with the literal text
  `<|begin_of_text|>`; to end on the answer channel, end the prompt with
  `<|start|>assistant to=user<|message|>`).
- `GET /status` reports the live split, measured ms per layer and contention events.
- Engine errors come back as HTTP 502 with the engine's message.
- `POST /score` on the engine port (8031) runs one teacher-forced forward over the
  given text or token ids on every rank and returns rank 0's per-token negative
  log-likelihood for comparing settings on a loaded ring.

Callers on the gateway machine need no token. A client on another machine must send
`Authorization: Bearer <cluster token>` (the contents of `~/.config/omarchy-cluster/token`)
and gets HTTP 401 without it. `/health` is open to everyone.

Point an OpenAI-compatible client at the gateway and set its temperature to 0. With LiteLLM
on another machine, put the cluster token in `api_key`:

```yaml
model_list:
  - model_name: local-cluster
    litellm_params:
      model: openai/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit
      api_base: http://gateway-node:8020/v1
      api_key: <cluster token>
```

```sh
litellm --config litellm.yaml --port 4000
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-cluster","messages":[{"role":"user","content":"hi"}]}'
```

## Environment switches

- `OMARCHY_CLUSTER_PYTHON` selects the rank interpreter an agent uses.
- `OMARCHY_LIVE_ENV` names a file of `NAME=VALUE` lines each rank applies before every job
  (default `~/.local/state/omarchy-cluster/live-env`), to run the same prompts with a
  route flag off and on.
- `OMARCHY_MLX_SERVE` names an omarchy-mlx `serve` directory whose model adapters are
  registered before a rank loads a model (Muse Spark Open). Nothing happens when
  unset or absent.
- `MLX_OMARCHY_HANG_NO_PROGRESS_NS` sets the Vulkan watchdog budget; the agent sets it
  to 120 s for every rank unless already set (see
  [troubleshooting](troubleshooting.md#vulkan-watchdog)).
