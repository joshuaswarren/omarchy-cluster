# Troubleshooting and limits

## Logs and stopping

- Logs live in `~/.local/share/omarchy-cluster/`: `gateway.log` on the gateway
  machine, `rank0.log` / `rank1.log` / ... on each node, `agent.log` for the agent,
  `llamacpp-engine.log` for the llama.cpp engine, `joint-engine.log` and
  `joint-decode-llama.log` for the joint engine.
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

## Joint engine (`--engine joint`)

- `... needs --slot-save-path for slot save/restore`: the prefill or decode
  llama-server was started without `--slot-save-path`; `/slots/N?action=save`
  answers 400 then. On the decode side `serve` passes it itself. On the
  prefill host point it at that node's agent kv dir
  (`~/.local/state/omarchy-cluster/kv`) so `/v1/kv` can reach the file.
- A `transport` error naming the agent means the pinned-TLS fetch to
  `--prefill-kv HOST:PORT` failed — check the pin (`pin` / `bootstrap`), not
  the llama-server. A missing file answers 404.
- Both servers must load the same GGUF and `-c`; `serve` warns on a `/props`
  model-name mismatch but cannot see the remote context size.
- Requests serialize on one slot file. If decode stalls, look for another
  client on the same slot or a stale `joint-kv.bin`.

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

## Two-node proof kit

The two-machine split proof lives in
[receipts/two-node-proof-kit](../receipts/two-node-proof-kit); its README is the
run recipe. These are the first-run blockers an outside tester hit (2026-11-21,
#128) and how the kit now behaves:

- Verify the kit before anything else: `cd receipts/two-node-proof-kit && sha256sum -c --strict kit.sha256` prints one `OK` line for each of the four kit files, and the preflight form above exits 2 with `CHECKS <n>` on a bare host — re-run outputs are recorded in [troubleshooting-receipt-20261122.md](../receipts/two-node-proof-kit/troubleshooting-receipt-20261122.md).
- The verifier checks that the three arms agree with each other, not that they reproduce our reference token digest. An outside tester whose hardware produces different tokens still passes; `combine.json` reports `matches_reference` per arm so a run that also matches ours stays visible.
- The two hosts must run llama.cpp builds from the same upstream commit. The kit records each host's `llama-server --version` into `llama-version.txt` and the verifier fails when they differ — a mismatched pair used to fail deep inside the run.
- Run the preflight before a real arm: `./two-node-proof.sh --check mac SOURCE GGUF OUT LLAMA_SERVER TLS M2_IP RPC_SERVER SPARK_IP` prints one PASS/FAIL line per prerequisite and exits 2 with `CHECKS <n>` when items fail, 0 when clean, and creates no run directory. All arguments are positional; nothing is read from the environment except the optional `SOCAT` override.
- The Mac certificate needs both SANs, `san=DNS:cluster-m2-rpc,IP:$M2_ADDR`: a cert with only the IP SAN stops the first arm at hostname mismatch ([secure RPC](secure-rpc.md)).
- Both hosts need socat with OpenSSL: `socat -V | grep WITH_OPENSSL`. A missing socat used to exit the preflight silently.
- macOS has no GNU `realpath -m`; the script resolves paths in Bash, so plain `realpath` calls must not come back.
- The model is the exact `Qwen3-1.7B-Q4_K_M.gguf` (1.3 GB, keep 2 GB free); the Mac's `/tmp` is a tmpfs, so the model there costs RAM. Reserve ports 24820, 24831-24832 and 24860-24864.

## Trust model

The agent control plane (:8025) is TLS with pinned keys, guarded by one shared
cluster token. The gateway (:8020), the engine and the rank ring are plain HTTP
on your LAN: a homelab trust model with no per-user auth. A node slowed by other
GPU work is detected and reported, not preempted. The MLX ring engine runs one request at a time and queues the rest. The
llama.cpp engine serves concurrent requests in llama-server's slots (see the README).
When all `--max-concurrent` gateway slots stay busy for 10 minutes, a request gets
HTTP 503 with `Retry-After`: retry it.
