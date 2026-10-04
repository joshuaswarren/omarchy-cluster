"""Discovery: mDNS advertise/browse for _omarchy-cluster._tcp + hostfile override."""
from __future__ import annotations

import ipaddress
import os
import platform
import re
import shutil
import subprocess

from .client import fetch_facts

SERVICE = "_omarchy-cluster._tcp"
DEFAULT_PORT = 8025
HOSTFILE = os.path.expanduser("~/.config/omarchy-cluster/hosts")


def _run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout + "\n" + p.stderr
    except (OSError, subprocess.SubprocessError):
        return ""


def advertise_start(name, port):
    """Start advertising; caller keeps the Popen alive for the process lifetime."""
    if platform.system() == "Darwin":
        cmd = ["dns-sd", "-R", name, SERVICE, ".", str(port)]
    else:
        cmd = ["avahi-publish", "-s", name, SERVICE, str(port), "omarchy-cluster agent"]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _browse_avahi(timeout):
    out = _run(["avahi-browse", "-rpt", "-t", SERVICE], timeout)
    found = {}
    for line in out.splitlines():
        f = line.split(";")
        if len(f) > 8 and f[0] == "=" and f[2] == "IPv4" and f[8].isdigit():
            found[f[3]] = {"name": f[3], "host": f[6], "ip": f[7], "port": int(f[8])}
    return found


def _browse_dns_sd(timeout):
    found = {}
    out = _run(["dns-sd", "-B", SERVICE, ".", str(timeout)], timeout + 3)
    names = []
    for line in out.splitlines():
        m = re.match(r"\s*\d+:\s+(\S+)", line)
        if m and SERVICE in m.group(1):
            names.append(m.group(1)[: -len(SERVICE) - len(".local.")].rstrip("."))
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out2 = _run(["dns-sd", "-L", name, SERVICE, ".", "2"], 5)
        m = re.search(r"reachable at (\S+?):(\d+)", out2)
        if not m:
            continue
        host, port = m.group(1), int(m.group(2))
        out3 = _run(["dns-sd", "-G", "v4", host, ".", "2"], 5)
        ips = re.findall(r"Add\s+\d+\s+\d+\s+\S+\s+(\d+\.\d+\.\d+\.\d+)", out3)
        if ips:
            found[name] = {"name": name, "host": host, "ip": ips[-1], "port": port}
    return found


def browse(timeout=4.0):
    if shutil.which("avahi-browse"):
        return _browse_avahi(timeout)
    return _browse_dns_sd(timeout)


def read_hostfile(path=HOSTFILE):
    """mlx.launch-style lines: `name` or `name=ip-or-host`; `#` comments."""
    entries = {}
    try:
        for raw in open(path):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, addr = line.split("=", 1)
                entries[name.strip()] = addr.strip()
            else:
                entries.setdefault(line, line)
    except FileNotFoundError:
        pass
    return entries


def _resolve(addr):
    try:
        ipaddress.ip_address(addr)
        return addr
    except ValueError:
        pass
    try:
        import socket
        return socket.gethostbyname(addr)
    except OSError:
        return addr  # leave as-is; facts fetch will fail visibly


def merge_discovered(found, overrides):
    """Hostfile wins per name; unknown hostfile names are added."""
    nodes = {name: dict(node, source="mdns") for name, node in found.items()}
    for name, addr in overrides.items():
        node = nodes.get(name, {"name": name, "host": addr, "port": DEFAULT_PORT})
        node["ip"] = _resolve(addr)
        node["source"] = "hostfile"
        nodes[name] = node
    return nodes


def discover_nodes(mdns_timeout=4.0, hostfile=HOSTFILE):
    """Browse mDNS, apply hostfile overrides, verify each node with /v1/facts.

    Returns {name: {name, ip, port, source, facts|None, error}}."""
    nodes = merge_discovered(browse(mdns_timeout), read_hostfile(hostfile))
    for node in nodes.values():
        try:
            node["facts"] = fetch_facts(node["ip"], node.get("port", DEFAULT_PORT))
            node["error"] = None
        except Exception as e:  # noqa: BLE001 - surface per-node failure
            node["facts"] = None
            node["error"] = "%s: %s" % (type(e).__name__, e)
    return nodes
