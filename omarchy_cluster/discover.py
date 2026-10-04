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
    """Run cmd, returning stdout+stderr even when it must be killed at timeout.

    dns-sd/avahi-browse stream until killed; their partial output is the result."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            return p.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            p.kill()
            return p.communicate()[0]
    except OSError:
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
            found[f[3]] = {"name": f[3], "host": f[6], "ip": f[7], "ips": [f[7]],
                           "port": int(f[8])}
    return found


def _browse_dns_sd(timeout):
    found = {}
    out = _run(["dns-sd", "-B", SERVICE, ".", str(timeout)], timeout + 3)
    names = []
    for line in out.splitlines():
        # Timestamp A/R Flags if Domain ServiceType InstanceName
        m = re.match(r"\s*\S+\s+Add\s+\d+\s+\d+\s+\S+\s+\S+\s+.+?$", line)
        if m and SERVICE in line:
            name = line.split()[-1].rstrip(".")
            if name not in names:
                names.append(name)
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out2 = _run(["dns-sd", "-L", name, SERVICE, ".", "2"], 5)
        m = re.search(r"reached at (\S+?):(\d+)", out2)
        if not m:
            continue
        host, port = m.group(1), int(m.group(2))
        out3 = _run(["dns-sd", "-G", "v4", host], 5)
        ips = [ip for ip in re.findall(r"(\d+\.\d+\.\d+\.\d+)", out3)
               if not ip.startswith("127.")]
        ips.sort(key=lambda ip: ip.startswith("169.254."))  # global first, link-local last
        if ips:
            found[name] = {"name": name, "host": host, "ip": ips[0], "ips": ips, "port": port}
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
        node["ips"] = [node["ip"]]
        node["source"] = "hostfile"
        nodes[name] = node
    return nodes


def discover_nodes(mdns_timeout=4.0, hostfile=HOSTFILE):
    """Browse mDNS, apply hostfile overrides, verify each node with /v1/facts.

    Returns {name: {name, ip, port, source, facts|None, error}}."""
    nodes = merge_discovered(browse(mdns_timeout), read_hostfile(hostfile))
    local = platform.node().split(".")[0] or "localhost"
    nodes.setdefault(local, {"name": local, "host": local, "ip": "127.0.0.1",
                             "ips": ["127.0.0.1"], "port": DEFAULT_PORT, "source": "local"})
    for node in nodes.values():
        node["facts"] = node["error"] = None
        best = None  # (score, ip, facts) — prefer the fastest interface that answers
        for ip in node.get("ips") or [node.get("ip")]:
            if not ip:
                continue
            try:
                facts = fetch_facts(ip, node.get("port", DEFAULT_PORT))
            except Exception as e:  # noqa: BLE001 - remember last failure per node
                node["error"] = "%s: %s" % (type(e).__name__, e)
                continue
            speeds = [ifc.get("speed_mbps") or 0 for ifc in facts.get("interfaces", [])
                      if ip in [a["ip"] for a in ifc.get("ips", [])]]
            score = max(speeds) if speeds else 0
            if best is None or score > best[0]:
                best = (score, ip, facts)
        if best:
            node["ip"] = best[1]
            node["facts"] = best[2]
            # Control plane rides the fastest interface that also answers.
            ranked = sorted(
                ((ifc.get("speed_mbps") or 0, not ifc.get("wireless"), a["ip"])
                 for ifc in best[2].get("interfaces", []) for a in ifc.get("ips", [])
                 if not a["ip"].startswith("169.254.")),
                reverse=True)
            for speed, _, ip in ranked:
                if speed <= best[0] and ip in (node.get("ips") or []):
                    break  # already on the best known address
                try:
                    node["facts"] = fetch_facts(ip, node.get("port", DEFAULT_PORT))
                    node["ip"] = ip
                    break
                except Exception:  # noqa: BLE001 - keep the first answerer
                    pass
    return nodes
