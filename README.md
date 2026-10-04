# omarchy-cluster

One local model across every machine in the homelab. Python, stdlib-only,
Python >= 3.9 (macOS system python works).

Part of the Omarchy M+ heterogeneous-inference plan: build tasks 1-3
(node agent + lifecycle units, discovery, pairwise probe + route table).

## What it does

- `omarchy-cluster agent` — per-node daemon: HTTP JSON facts on **:8025**
  (OS, chip, unified memory, GPU backend metal/vulkan/cuda, free memory,
  interfaces with IPs + link speeds), 1 Hz heartbeat counter, TCP bulk-sink
  on :8026, UDP echo on :8027, iperf3 helper endpoints. Port 8020 is
  reserved for the phase-1 gateway (never use 8002 — that is the live oMLX
  advisor on mac-a).
- `omarchy-cluster install-agent` — installs and starts the daemon:
  launchd LaunchAgent (`~/Library/LaunchAgents/dev.omarchy.cluster-agent.plist`)
  on macOS, systemd user unit (`~/.config/systemd/user/omarchy-cluster-agent.service`)
  on Linux. Copies the package to `~/.local/share/omarchy-cluster/src` and
  drops a `~/.local/bin/omarchy-cluster` wrapper.
- `omarchy-cluster discover` — mDNS `_omarchy-cluster._tcp` browse
  (Avahi on Linux, dns-sd on macOS) + `~/.config/omarchy-cluster/hosts`
  override (`name=ip-or-host`, mlx.launch-style lines). Hostfile wins per name.
- `omarchy-cluster probe` — every pair x every interface route:
  RTT (TCP connect + UDP echo) and bandwidth both directions. iperf3
  1-stream 10 s when both ends are Linux; **on any macOS end the built-in
  Python TCP sender is used** (brew iperf3 is Local-Network-privacy blocked
  on mac-a, verified 2026-10-04). Writes `links.json`, pins the fastest
  route per pair (bottleneck direction, RTT tiebreak), and marks Wi-Fi and
  Tailscale routes ineligible for decode ranks whatever they measure.
- `omarchy-cluster status` — node table (OS, chip, memory, backend, up) +
  pinned route per pair from `links.json`.

State lives in `~/.local/state/omarchy-cluster/links.json`.

## Quick start

```sh
pip install .            # or: python3 -m omarchy_cluster.cli directly
omarchy-cluster install-agent   # on every node
omarchy-cluster discover
omarchy-cluster probe           # from any node
omarchy-cluster status
```

## Real-fleet receipts

(filled from the 2026-10-04 run; see below)

## Tests

```sh
python3 -m pytest tests/
```

Covers the route picker (fastest eligible pin, RTT tiebreak, failover
order, Wi-Fi/Tailscale never decode-eligible), hostfile merge/override,
links.json round-trip, and route-candidate dedup.
