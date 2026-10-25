# Troubleshooting and limits

## Logs and stopping

- Logs live in `~/.local/share/omarchy-cluster/`: `gateway.log` on the gateway
  machine, `rank0.log` / `rank1.log` / ... on each node, `agent.log` for the agent,
  `llamacpp-engine.log` for the llama.cpp engine.
- `omarchy-cluster stop` stops the gateway and every rank and sweeps the ports.
- `omarchy-cluster guard` polls memory and stops the run when a Linux node has
  under 2 GB available, a Mac has under 15 percent memory pressure-free or
  under 20 GB on its Data volume, or a node stops answering. Run it on the
  gateway. A polling guard cannot guarantee that a node will avoid an OOM.

## Vulkan watchdog

A Vulkan rank whose GPU queue waits on a long ring operation can trip the
omarchy-mlx watchdog, which throws when a timeline wait makes no progress for 10 s
by default. A 256-token prefill over 62 layers on five machines takes about 40 s,
and the first pipeline stage waits for all of it. The rank launcher therefore sets
`MLX_OMARCHY_HANG_NO_PROGRESS_NS` to 120 s for every rank unless you set it
yourself; the first wait in a process fixes the value, so it has to be in the
rank's launch environment.

## Vulkan out-of-device-memory on small nodes

`place` sizes stages from free memory only and cannot see the driver's real
allocation ceiling. On a 16 GB M1 on Omarchy, DeepSeek-Coder-V2-Lite-4bit
(8.8 GB) failed to load whole with a Vulkan out-of-device-memory error once mlx
held 8.07 GB, with 5 GB of RAM still available (2026-10-11,
[receipt](../receipts/2026-10-11-single-host-baseline)); an earlier note put the
failure at about 8.5 GB on the 2026-10-06 build. The driver reports a 16.2 GB
heap, about twice the usable size. Give such a node fewer layers with `--split`.

Related: one Honeykrisp process maps at most about 63 GiB whatever the node has.
Give a bigger laptop two rank processes (`--nodes` with the node twice) or, on the
llama.cpp engine, two rpc-servers on two ports with `=GB` budgets each
([placement](placement.md#multi-rank-on-one-node)).

## Quantization limits on the Vulkan backend

Receipts on these nodes cover 2-bit (Muse Spark Open 2-bit mixed: 2-bit experts,
4-bit attention and embedding) and 4-bit models. 5-bit and 6-bit MoE checkpoints
have failed on these nodes. If a load dies in a quantized kernel, try a different
bit width before anything else.

## GPU contention

Other GPU work on a Mac can slow that node's decode several-fold (a measured case:
about 46 to about 22 tok/s). `GET /status` reports contention events. The split
itself cannot be stolen by a burst (the chooser keeps the fastest ms/layer ever
  seen). If a driver update or a new resident app slows a machine for good,
delete `~/.local/state/omarchy-cluster/layer-ms.json` and restart `serve`.

## Ring and rank failures

- `ring connection to a peer was lost` in a rank log means another rank died or a
  node dropped; check the peer's rank log and `guard`'s stop reason. One GPU hang
  took a whole run down in the full-62 receipt (a display-session restart
  correlated with the hang; cause not proven).
- Every rank must reach its neighbours at one address. If a node's first interface
  is not the reachable one (Wi-Fi laptop, multiple NICs), set `--ring-ip NAME=IP`.
- A node may run several ranks (own ring port each: 52100, 52101, ...), but all
  its layers must fit its free memory together.

## Prompts and templates

- The MLX engine applies the model's chat template to `messages` and to `prompt`.
  Send `"raw": true` with a `prompt` string to `/v1/completions` to skip the
  template; the engine adds no beginning-of-text token, so the raw prompt must
  carry its own.
- Muse Spark Open reasons before it answers unless the prompt ends on the answer
  channel (`<|start|>assistant to=user<|message|>`); rendered examples live in
  [receipts/2026-10-16-muse-spark-llamacpp-rpc/prompts](../receipts/2026-10-16-muse-spark-llamacpp-rpc).
  With the reasoning channel left open, the receipt's model looped to its token
  limit. Treat long restatements as a soft loop and force the answer channel.
- llama-server warm-up reads every expert of a mixture-of-experts model; the
  receipt runs pass `--llama-arg=--no-warmup`. Remember `=` in `--llama-arg=`.

## Loading

- Each node loads only its own layers from its Hugging Face cache: download the
  model on every node first (`hf download ...`). The llama.cpp engine is the
  exception: only the gateway needs the GGUF, and `--rpc-cache` lets RPC nodes
  keep their received layers for the next load.
- Load over one slow link dominates: a 156.8 GB model over one 2.5 Gb/s link took
  about 1055 s. Plan around the slowest node's link.
- The venv must persist: the agent starts from it at every boot, and
  `install-agent` refuses a venv under `/tmp`.

## Comparing settings

Two switches help on a loaded ring (see [cli.md](cli.md#environment-switches)):
`OMARCHY_LIVE_ENV` applies `NAME=VALUE` lines to every rank before each job, and
`POST /score` on the engine port returns rank 0's per-token negative
log-likelihood for one forward, a teacher-forced check that needs no generation.

## Trust model

Plain HTTP with one shared token on your LAN: a homelab trust model, no TLS, no
per-user auth. A node slowed by other GPU work is detected and reported, not
preempted. One request at a time: concurrent requests queue, and they add no
throughput.
