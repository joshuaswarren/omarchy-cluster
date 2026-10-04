"""omarchy-cluster CLI: agent, install-agent, discover, probe, status."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

from . import AGENT_VERSION
from . import discover, probe as probe_mod

GB = 1024.0 ** 3


def _agent_src_dir():
    """Directory containing the omarchy_cluster package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- agent ----

def cmd_agent(args):
    from . import agent
    agent.main(["--port", str(args.port)] + (["--name", args.name] if args.name else [])
               + (["--hub", args.hub] if args.hub else []))


# ---- install-agent ----

_UNIT_LINUX = """\
[Unit]
Description=omarchy-cluster agent
After=network-online.target

[Service]
ExecStart={python} -m omarchy_cluster.agent --port {port}
Environment=PYTHONPATH={src}
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
"""

_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.omarchy.cluster-agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>-m</string>
    <string>omarchy_cluster.agent</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONPATH</key><string>{src}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def cmd_install_agent(args):
    home = os.path.expanduser("~")
    src = os.path.join(home, ".local/share/omarchy-cluster/src")
    pkg_dst = os.path.join(src, "omarchy_cluster")
    shutil.rmtree(pkg_dst, ignore_errors=True)
    os.makedirs(src, exist_ok=True)
    shutil.copytree(_agent_src_dir(), pkg_dst)

    wrapper = os.path.join(home, ".local/bin/omarchy-cluster")
    os.makedirs(os.path.dirname(wrapper), exist_ok=True)
    py = sys.executable or "python3"
    with open(wrapper, "w") as f:
        f.write("#!/bin/sh\nPYTHONPATH=%s exec %s -m omarchy_cluster.cli \"$@\"\n" % (src, py))
    os.chmod(wrapper, 0o755)

    if platform.system() == "Darwin":
        log = os.path.join(home, ".local/share/omarchy-cluster/agent.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        plist = os.path.join(home, "Library/LaunchAgents/dev.omarchy.cluster-agent.plist")
        os.makedirs(os.path.dirname(plist), exist_ok=True)
        with open(plist, "w") as f:
            f.write(_PLIST.format(python=py, port=args.port, src=src, log=log))
        subprocess.run(["launchctl", "bootout", "gui/%d/dev.omarchy.cluster-agent" % os.getuid()],
                       capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", "gui/%d" % os.getuid(), plist],
                           capture_output=True, text=True)
        if r.returncode != 0:  # older macOS
            r = subprocess.run(["launchctl", "load", "-w", plist], capture_output=True, text=True)
        print("launchd: %s" % (r.stdout.strip() or r.stderr.strip() or "loaded %s" % plist))
    else:
        unit_dir = os.path.join(home, ".config/systemd/user")
        os.makedirs(unit_dir, exist_ok=True)
        unit = os.path.join(unit_dir, "omarchy-cluster-agent.service")
        with open(unit, "w") as f:
            f.write(_UNIT_LINUX.format(python=py, port=args.port, src=src))
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", "omarchy-cluster-agent"],
                       check=True)
        linger = subprocess.run(["loginctl", "enable-linger"], capture_output=True, text=True)
        print("systemd user unit enabled; linger: %s" %
              ("on" if linger.returncode == 0 else "FAILED (%s)" % linger.stderr.strip()))
    print("agent installed; wrapper: %s" % wrapper)


# ---- discover ----

def cmd_discover(args):
    nodes = discover.discover_nodes(mdns_timeout=args.mdns_timeout)
    for name in sorted(nodes):
        n = nodes[name]
        print("%-16s %-15s %-8s %s" % (name, n.get("ip"), n.get("source"),
                                       "ok" if n.get("facts") else n.get("error")))
    return nodes


# ---- probe ----

def cmd_probe(args):
    nodes = {n: d for n, d in discover.discover_nodes().items() if d.get("facts")}
    if len(nodes) < 2:
        sys.exit("need at least 2 reachable nodes, found %d" % len(nodes))
    links = probe_mod.probe_all(nodes, seconds=args.seconds)
    path = probe_mod.save_links(links, args.out)
    print(links_table(links))
    print("wrote %s" % path)
    return links


def links_table(links):
    lines = []
    for pair in links.get("pairs", []):
        lines.append("%s <-> %s" % (pair["a"], pair["b"]))
        if pair.get("error"):
            lines.append("  ERROR: %s" % pair["error"])
            continue
        pinned = pair.get("pinned") or {}
        for r in pair["routes"]:
            mark = "*" if r is pair.get("pinned") else " "
            g1 = "%.2f" % r["gbps_a2b"] if r.get("gbps_a2b") else "  --  "
            g2 = "%.2f" % r["gbps_b2a"] if r.get("gbps_b2a") else "  --  "
            rtt = rtt_str(r)
            dec = "decode-ok" if r.get("eligible_for_decode") else "NO-DECODE"
            lines.append(" %s %-6s %s:%-15s -> %s:%-15s  %s/%s Gb/s  rtt %s  %s%s" % (
                mark, r["media"], r["a_iface"], r["a_ip"], r["b_iface"], r["b_ip"],
                g1, g2, rtt, dec,
                "" if r is not pair.get("pinned") else "  PINNED"))
        if not pair["routes"]:
            lines.append("  (no routes)")
    return "\n".join(lines)


def rtt_str(r):
    t, u = r.get("rtt_tcp_ms"), r.get("rtt_udp_ms")
    return "%.2f/%s ms" % (t, ("%.2f" % u) if u is not None else "--")


# ---- status ----

def cmd_status(args):
    nodes = discover.discover_nodes()
    print(node_table(nodes))
    links = probe_mod.load_links(args.links)
    if links:
        print()
        print("pinned routes (links.json %s):" % links.get("generated"))
        print(links_table(links))
    else:
        print("\nno links.json yet; run: omarchy-cluster probe")


def node_table(nodes):
    lines = ["%-16s %-22s %-16s %-13s %-7s %-4s %s" % (
        "NODE", "OS", "CHIP", "MEM FREE/TOT", "BACKEND", "UP", "IP")]
    for name in sorted(nodes):
        n = nodes[name]
        f = n.get("facts")
        if not f:
            lines.append("%-16s %-22s %-16s %-13s %-7s %-4s %s" % (
                name, "?", "?", "?", "?", "DOWN", "%s (%s)" % (n.get("ip"), n.get("error"))))
            continue
        up = f.get("heartbeat_age_s", 99) < 3.0
        mem = "%.1f/%.0fGB" % (f["memory_free_bytes"] / GB, f["memory_total_bytes"] / GB)
        lines.append("%-16s %-22s %-16s %-13s %-7s %-4s %s" % (
            name, f["os"], f["chip"], mem,
            f["gpu_backend"], "up" if up else "stale", n.get("ip")))
    return "\n".join(lines)


# ---- entry ----

def main(argv=None):
    ap = argparse.ArgumentParser(prog="omarchy-cluster", description=__doc__)
    ap.add_argument("--version", action="version", version=AGENT_VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("agent", help="run the node agent")
    p.add_argument("--port", type=int, default=8025)
    p.add_argument("--name", default=None)
    p.add_argument("--hub", default=None, help="control-hub base URL for 1 s heartbeats")
    p.set_defaults(fn=cmd_agent)

    p = sub.add_parser("install-agent", help="install and start the agent daemon on this node")
    p.add_argument("--port", type=int, default=8025)
    p.set_defaults(fn=cmd_install_agent)

    p = sub.add_parser("discover", help="list discovered nodes")
    p.add_argument("--mdns-timeout", type=float, default=4.0)
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("probe", help="measure every pair/route, write links.json")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("status", help="show nodes and pinned routes")
    p.add_argument("--links", default=None)
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
