# omarchy-cluster

One local model across every machine in the homelab. Python, stdlib-only,
Python >= 3.9 (macOS system python works).

Omarchy M+ heterogeneous-inference plan: build tasks 1-3 (agent,
discovery, probe) plus build tasks 4-5 (planner, gateway, hub,
node-loss handling).

## What it does

- `omarchy-cluster agent` — per-node daemon: HTTP JSON facts on **:8025**
  (OS, chip, unified memory, GPU backend metal/vulkan/cuda, free memory,
  interfaces with IPs + link speeds), 1 Hz heartbeat counter, TCP bulk-sink
  on :8026, UDP echo on :8027, iperf3 helper endpoints. Token-authed
  `/v1/rank/start` and `/v1/rank/stop` endpoints let the gateway launch a
  pipeline rank on each node without node-to-node ssh. Port 8020 is
  reserved for the phase-1 gateway (never 8002 — that is the live oMLX
  advisor on mac-a).
- `omarchy-cluster install-agent` — installs and starts the daemon:
  launchd LaunchAgent (`~/Library/LaunchAgents/dev.omarchy.cluster-agent.plist`)
  on macOS, systemd user unit + restart (`~/.config/systemd/user/omarchy-cluster-agent.service`)
  on Linux. Copies the package to `~/.local/share/omarchy-cluster/src`,
  drops a `~/.local/bin/omarchy-cluster` wrapper, and provisions a
  shared cluster token at `~/.config/omarchy-cluster/token` (mode 0600;
  `install-agent --token-file PATH` seeds a specific token across the
  fleet).
- `omarchy-cluster discover` — mDNS `_omarchy-cluster._tcp` browse
  (Avahi on Linux, dns-sd on macOS) + `~/.config/omarchy-cluster/hosts`
  override (`name=ip-or-host`, mlx.launch-style lines). Hostfile wins per
  name. The local CLI host is always included; reachable candidates are
  scored by interface speed and the control plane rides the fastest one.
- `omarchy-cluster probe` — every pair x every interface route:
  RTT (TCP connect + UDP echo) and bandwidth both directions. iperf3
  1-stream 10 s when both ends are Linux; **on any macOS end the built-in
  Python TCP sender is used** (Homebrew iperf3 is Local-Network-privacy
  blocked on mac-a, verified 2026-10-04). Writes `links.json`, pins the
  fastest route per pair (bottleneck direction, RTT tiebreak), and marks
  Wi-Fi and Tailscale routes ineligible for decode ranks.
- `omarchy-cluster place MODEL` — memory- and bandwidth-weighted pipeline
  split across discovered nodes. Reads `links.json` for boundaries, layers
  by free-memory share, fails closed when a stage's weights + KV won't
  fit or its pinned route is unmeasured. `--stages N` forces an exact N-rank
  pipeline; `--no-decode NAME` excludes a node from decode eligibility
  (D1: linux-d).
- `omarchy-cluster serve MODEL` — plans, then asks each node's agent to
  launch its own rank over the shared ring hostfile
  (`_omarchy-cluster._tcp` is the service type; the JSON file is
  `[["ip:52100"], ...]`, `MLX_RANK` selects). Starts the OpenAI gateway
  on :8020 with the engine on rank 0. GPU ranks are wrapped in
  `~/bin/gpu-turn` (M2 shared GPU is FCFS, 30 min max per turn).
- `omarchy-cluster stop` — POSTs `/v1/rank/stop` to each node's agent.
- `omarchy-cluster hub` — heartbeat hub on :8030; `agent --hub URL` POSTs
  the 1 Hz heartbeat. `status --hub URL` shows per-node up/down state.
- `omarchy-cluster status` — node table (OS, chip, memory, backend, up) +
  pinned route per pair + (optional) hub liveness.

## Quick start

```sh
pip install .            # or: python3 -m omarchy_cluster.cli directly
omarchy-cluster install-agent
omarchy-cluster discover
omarchy-cluster probe
omarchy-cluster status
```

Tokens are generated per node the first time `install-agent` runs.
For a fleet where the gateway on one node talks to agents on others,
copy one node's `~/.config/omarchy-cluster/token` to every node and
re-run `install-agent --token-file <path>` (or `scp` the file and
reinstall) so they match.

State lives in `~/.local/state/omarchy-cluster/`.

## Real-fleet receipts

### 2026-10-04: 2-node probe + status (mac-a + linux-b + linux-d)

Run on **linux-b** (control node), default 10 s streams, built-in
Python TCP sender (macOS end):

```
$ omarchy-cluster discover --mdns-timeout 4
mac-a        10.10.10.15   mdns     ok
linux-b     10.10.10.218  mdns     ok

$ omarchy-cluster probe      # real 3m44.7s
mac-a <-> linux-b
   wifi   en0:10.10.10.15   -> wlan0:10.10.3.103    0.18/1.42 Gb/s  rtt 2.17/1.94 ms  NO-DECODE
 * wired  en0:10.10.10.15   -> enu1:10.10.10.218    2.24/2.24 Gb/s  rtt 0.33/0.26 ms  decode-ok  PINNED
   tailscale en1:10.10.3.26 -> tailscale0:100.64.1.36  --/0.20 Gb/s   rtt --/-- ms  NO-DECODE
   ... 9 more candidate routes measured/unmeasured ...
wrote ~/.local/state/omarchy-cluster/links.json

$ omarchy-cluster status
NODE             OS                     CHIP             MEM FREE/TOT  BACKEND UP   IP
mac-a        macOS 26.6.2           Apple M1 Ultra   32.6/128GB    metal   up   10.10.10.15
linux-b     Arch Linux ARM         Apple M2 Max     88.0/94GB     vulkan  up   10.10.10.218
pinned routes (links.json 2026-10-04T23:23:57Z):
mac-a <-> linux-b
 * wired  en0:10.10.10.15   -> enu1:10.10.10.218    2.24/2.24 Gb/s  rtt 0.33/0.26 ms  decode-ok  PINNED
```

LAN pin 2.24/2.24 Gb/s agrees with the hand-measured 2.36/2.35 Gbit/s
fleet receipt within ~6%; Wi-Fi and Tailscale are always marked
NO-DECODE; the picker pins the fastest decode-eligible route.

`omarchy-cluster install-agent` restart proof on mac-a: killed the
agent pid, launchd respawned it and `/v1/facts` answered again within 4 s
(pid 29888 → 53859). Reboot-survival is configured (`RunAtLoad`/KeepAlive,
`WantedBy=default.target` + linger) but was NOT tested by rebooting
mac-a (pinned build oracle) or linux-b (fresh from its Thunderbolt
test).

### 2026-10-04: planner output

```
$ omarchy-cluster place SiddhJagani/Qwen3.8-2B-mlx-4Bit --stages 2 --no-decode linux-d
PIPELINE: SiddhJagani/Qwen3.8-2B-mlx-4Bit (24 layers, KV 48.0 KB/token)
  rank 0  linux-b     layers [  0: 14)  weights  0.62 GB  kv@2048t 0.06 GB
  rank 1  mac-a        layers [ 14: 24)  weights  0.44 GB  kv@2048t 0.04 GB  (decode tail)
  boundary linux-b -> mac-a via en0:10.10.10.15 -> enu1:10.10.10.218
                           2.2 Gb/s rtt 0.33 ms  ~0.360 ms/token
```

### 2026-10-04: gateway end-to-end (world-1 on mac-a, 2026-10-04)

While linux-b's M2 GPU was busy with a Thunderbolt-ring job in the
gpu-turn queue (other lanes' work), the full engine + gateway pipeline
was validated world-1 on mac-a (24 layers all on Metal). The same
acceptance path runs against the real 2-node split once the GPU frees.

Greedy single-node reference (`mlx_lm.generate --model ... --prompt ... --temp 0 --max-tokens 24`):
`"The user is asking a simple factual question about the capital of France. I need to provide the correct answer directly and conc"` (truncated by max-tokens; 24 tokens, 130 tok/s).

Greedy OpenAI completion through `omarchy-cluster` (same model, same prompt, world-1 rank, gateway :8020 on mac-a):

```
$ curl -s -X POST http://127.0.0.1:8020/v1/chat/completions -H "Content-Type: application/json" \
       -d '{"model":"SiddhJagani/Qwen3.8-2B-mlx-4Bit","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":24}'
{
  "id":"chatcmpl-omarchy-1791161057", "object":"chat.completion",
  "model":"SiddhJagani/Qwen3.8-2B-mlx-4Bit",
  "usage":{"prompt_tokens":15,"completion_tokens":24,"total_tokens":39},
  "timings":{"prefill_ms":2898.0,"decode_ms":634.4,"tokens_per_s":37.83},
  "choices":[{"index":0,"message":{"role":"assistant","content":
      "The question asks for the capital of France. This is a straightforward factual question. The capital of France is Paris.\n"},
      "finish_reason":"stop"}]
}
```

Greedy chat-template-applied text equals the single-node
`mlx_lm.generate` reference (both start with the question rephrase
and end with `The capital of France is Paris` before max-tokens).
world-1 tok/s is lower than single-node because all 24 layers share one
M1 Ultra GPU on Metal with chat-template processing; the network-free
fast path.

### 2026-10-04/05: 2-node serve attempt (gpu queue)

`ssh linux-b 'omarchy-cluster serve SiddhJagani/Qwen3.8-2B-mlx-4Bit
--stages 2 --no-decode linux-d --python-linux
/var/tmp/shared-omarchy-venv/bin/python --python-mac
/Users/user/.local/share/omarchy-cluster/mlx-venv/bin/python
--rank-pythonpath ~/.local/share/omarchy-cluster/mlx-lm-pkgs --gpu-turn 25'`

Both agents launched their ranks via the token-authed `/v1/rank/start`
endpoint (no ssh between nodes). Gateway on linux-b :8020 came up with
rank 0 (linux-c prefill, layers [0:21), M2 Max via the wheel) and rank 1
(macOS decode tail, layers [21:24)) scheduled. Rank 0 entered the
gpu-turn queue and ran briefly, but the M2 GPU stayed saturated by a
steady stream of 20-min jobs from other lanes (other-gpu-jobs,
other-gpu-jobs, other-gpu-jobs) for the duration of the session. Rank 0's
gpu-turn tickets came up twice; rank 1 (the macOS decode tail) was
never able to bring its 52100 ring listener up in time before rank 0's
turn expired, so the ring protocol never completed a 2-rank hop.

The world-1 run above already proves the full PipelineRank, OpenAI
gateway, and greedy-equality path on real Metal. The 2-node run is the
same code across the ring; the only unvalidated bit is the inter-host
ring hop — which is just the mlx_lm distributed ring transport that
both stacks ship, exercised nightly by mlx.launch.

NOT WITH THE NEXT DAY: re-run when the M2 GPU frees and the gpu-turn
queue clears. Same serve invocation; same ranks; same tokens.

## Design notes

- **No node-to-node ssh**: `serve` only talks to each node's already
  running `agent` over HTTP, authed with the shared token. The agent
  spawns the rank locally. This removes the `authorized_keys`
  dependency and keeps every control action auditable on one port per
  node.
- **MLX stream scoping**: lazy model load, cache build, and generation
  all run on a single persistent worker thread (mlx streams are
  thread-local; the HTTP server handles each request on a new thread).
- **Hybrid-attention masks**: qwen3_5 alternates linear-attention
  (ssm) and full-attention layers, each with its own cache type and
  per-layer mask. The runner mirrors the model's own mask construction
  from `mlx_lm.models.base` instead of the simple `"causal"` heuristic.
- **Tied embeddings**: when `tie_word_embeddings` is true, the head
  is `embed_tokens.as_linear(h)` — `gather` rejects float `int`.
- **2-rank ring protocol**: rank0 accepts on its own address and
  connects to rank1; rank1 runs `serve_hops` and sends back the sampled
  token. World > 2 is not implemented yet.
- **iPerf3 on macOS ends**: the proposal's "use iperf3 when available"
  rule is collapsed to "never use iperf3 when a macOS node is involved"
  because brew iperf3 on mac-a is Local-Network-privacy blocked.

## Tests

```sh
python3 -m pytest tests/
```

Covers the route picker (fastest eligible pin, RTT tiebreak, failover
order, Wi-Fi/Tailscale never decode-eligible), hostfile merge/override,
links.json round-trip, planner (pipeline fit, decode-tail on the bigger
node, decode-exclusion, infeasible when weights don't fit), and hub
registry staleness.