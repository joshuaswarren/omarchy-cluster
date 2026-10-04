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

## Real-fleet receipts — 2026-10-04

Agents installed and running as daemons on **mac-a** (macOS 26.6.2, M1
Ultra, launchd LaunchAgent, KeepAlive) and **linux-b** (Arch Linux ARM,
M2 Max, systemd user unit, linger on). Both advertise and browse
`_omarchy-cluster._tcp` natively (dns-sd / Avahi) — zero-config discovery.
Code SHA at time of run: `d5fff8c` (+ marker fix in the final commit).

`omarchy-cluster install-agent` restart proof on mac-a: killed the agent
pid, launchd respawed it and `/v1/facts` answered again within 4 s
(pid 29888 → 53859).

Ran on **linux-b** (control node), against mac-a, default 10 s
streams, built-in Python TCP sender (macOS end; brew iperf3 is
Local-Network-blocked there):

```
$ omarchy-cluster discover --mdns-timeout 4
mac-a        10.10.10.15   mdns     ok
linux-b     10.10.10.218  mdns     ok

$ time omarchy-cluster probe          # real 3m44.7s
mac-a <-> linux-b
   wifi   en0:10.10.10.15   -> wlan0:10.10.3.103    0.18/1.42 Gb/s  rtt 2.17/1.94 ms  NO-DECODE
 * wired  en0:10.10.10.15   -> enu1:10.10.10.218    2.24/2.24 Gb/s  rtt 0.33/0.26 ms  decode-ok  PINNED
   tailscale en1:10.10.3.26 -> tailscale0:100.64.1.36  --/0.20 Gb/s   rtt --/-- ms  NO-DECODE
   ... 9 more candidate routes measured/unmeasured, all recorded in links.json ...
wrote /home/user/.local/state/omarchy-cluster/links.json

$ omarchy-cluster status
NODE             OS                     CHIP             MEM FREE/TOT  BACKEND UP   IP
mac-a        macOS 26.6.2           Apple M1 Ultra   32.6/128GB    metal   up   10.10.10.15
linux-b     Arch Linux ARM         Apple M2 Max     88.0/94GB     vulkan  up   10.10.10.218
pinned routes (links.json 2026-10-04T23:23:57Z):
mac-a <-> linux-b
 * wired  en0:10.10.10.15   -> enu1:10.10.10.218    2.24/2.24 Gb/s  rtt 0.33/0.26 ms  decode-ok  PINNED
```

Measured LAN pin 2.24/2.24 Gb/s agrees with the earlier hand-measured
2.36/2.35 Gbit/s fleet receipt (the wired-218 "hangs after TCP connect"
return-route issue did not reproduce — full 10 s streams flowed both ways).
Wi-Fi and Tailscale routes are measured but always marked NO-DECODE; the
picker cannot pin them while an eligible route exists.

Second smoke on a third Linux box (control + agent, hostfile override
`mac-a=10.10.10.15` while its mDNS answer rode another interface):
LAN pinned at 6.46/5.78 Gb/s (10GbE en0), RTT 0.48/0.45 ms, Wi-Fi measured
0.20/0.20 Gb/s and correctly excluded from decode.

Notes and limits:

- Reboot re-registration is configured (launchd RunAtLoad/KeepAlive; systemd
  `WantedBy=default.target` + linger) but was NOT tested by rebooting
  mac-a (pinned build oracle) or linux-b (fresh from its Thunderbolt
  test). Restart-under-keepalive was proven on mac-a instead.
- The 1 s heartbeat today updates per-agent state (`heartbeat_seq` /
  `heartbeat_age_s`, fetched with facts; `--hub URL` POSTs it to a control
  hub). The hub-side registry protocol lands with build task 4.
- mDNS may answer with any of a machine's addresses; discovery verifies
  candidates and moves the control plane to the fastest answering interface
  (here: en0 10GbE).

## Tests

```sh
python3 -m pytest tests/
```

Covers the route picker (fastest eligible pin, RTT tiebreak, failover
order, Wi-Fi/Tailscale never decode-eligible), hostfile merge/override,
links.json round-trip, and route-candidate dedup.
