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
JOINED = os.path.expanduser("~/.local/state/omarchy-cluster/joined")
LEFT = os.path.expanduser("~/.local/state/omarchy-cluster/left")  # name -> ssh target; never read as trust


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


def _split_port(addr):
    """`host` or `host:port` (one colon, digits after it) -> (host, port)."""
    host, sep, port = addr.rpartition(":")
    if sep and addr.count(":") == 1 and port.isdigit():
        return host, int(port)
    return addr, DEFAULT_PORT


def _trusted_port(node, ip, joined):
    """The port to send the token to at `ip`, or None when this machine does not trust that endpoint.

    The port never comes from the mDNS advertisement: it comes from the hosts file, from the join
    record, or (this machine) from DEFAULT_PORT."""
    if node.get("source") in ("local", "hostfile"):
        return node["port"]
    for endpoint in joined.get(node["name"], ()):
        host, _, port = endpoint.rpartition(":")
        if host == ip and port.isdigit():
            return int(port)
    return None


def merge_discovered(found, overrides):
    """Hostfile wins per name; unknown hostfile names are added. A hostfile entry is `name=host[:port]`;
    its port replaces the advertised one."""
    nodes = {name: dict(node, source="mdns") for name, node in found.items()}
    for name, addr in overrides.items():
        host, port = _split_port(addr)
        node = nodes.get(name, {"name": name, "host": host})
        node["port"] = port
        node["ip"] = _resolve(host)
        node["ips"] = [node["ip"]]
        node["source"] = "hostfile"
        nodes[name] = node
    return nodes


def read_joined(path=None):
    """{name: {"ip:port", ...}}: the endpoints of the nodes `bootstrap`/`join` installed, as this machine
    recorded them."""
    joined = {}
    try:
        with open(path or JOINED) as f:
            for line in f:
                name, _, addrs = line.strip().partition(" ")
                if name:
                    joined[name] = set(a for a in addrs.split(",") if a)
    except FileNotFoundError:
        pass
    return joined


def remember_joined(name, addresses, port=DEFAULT_PORT, path=None):
    """Record a node this machine installed over ssh. One line per name; endpoints merge."""
    path = path or JOINED
    joined = read_joined(path)
    joined.setdefault(name, set()).update("%s:%d" % (a, port) for a in addresses)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for node in sorted(joined):
            f.write("%s %s\n" % (node, ",".join(sorted(joined[node]))))


def forget_joined(name, path=None):
    path = path or JOINED
    joined = read_joined(path)
    if joined.pop(name, None) is not None:
        with open(path, "w") as f:
            for node in sorted(joined):
                f.write("%s %s\n" % (node, ",".join(sorted(joined[node]))))


def discover_nodes(mdns_timeout=4.0, hostfile=HOSTFILE):
    """Browse mDNS, apply hostfile overrides, verify each node with /v1/facts.

    The cluster token goes only to nodes this machine trusts: this machine, hostfile entries (the
    operator wrote them) and nodes it joined, at an address and port recorded then. An advertiser that
    is none of these gets no request at all: anyone on the LAN can advertise the service, under any
    name, including this machine's.

    Returns {name: {name, ip, port, source, facts|None, error}}."""
    overrides = read_hostfile(hostfile)
    nodes = merge_discovered(browse(mdns_timeout), overrides)
    local = platform.node().split(".")[0] or "localhost"
    if local not in overrides:  # this machine's own advertisement is not evidence of anything
        nodes[local] = {"name": local, "host": local, "ip": "127.0.0.1", "ips": ["127.0.0.1"],
                        "port": DEFAULT_PORT, "source": "local"}
    joined = read_joined()
    for node in nodes.values():
        node["facts"] = node["error"] = None
        best = None  # (score, ip, facts) - prefer the fastest interface that answers
        for ip in node.get("ips") or [node.get("ip")]:
            if not ip:
                continue
            port = _trusted_port(node, ip, joined)
            if port is None:
                node["error"] = ("not joined: run `omarchy-cluster bootstrap` for %s, or list it in the hosts file"
                                 % node["name"])
                continue
            try:
                facts = fetch_facts(ip, port)
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
            node["port"] = _trusted_port(node, best[1], joined)
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
                port = _trusted_port(node, ip, joined)
                if port is None:
                    continue
                try:
                    node["facts"] = fetch_facts(ip, port)
                    node["ip"] = ip
                    node["port"] = port
                    break
                except Exception:  # noqa: BLE001 - keep the first answerer
                    pass
    return nodes


def read_left(path=None):
    """{name: ssh target}: peers this machine left. It only lets `leave NAME` run again once the stopped
    agent no longer advertises; discovery never trusts it."""
    left = {}
    try:
        with open(path or LEFT) as f:
            for line in f:
                name, _, target = line.strip().partition(" ")
                if name and target:
                    left[name] = target
    except FileNotFoundError:
        pass
    return left


def _write_left(left, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name in sorted(left):
            f.write("%s %s\n" % (name, left[name]))


def remember_left(name, target, path=None):
    path = path or LEFT
    left = read_left(path)
    left[name] = target
    _write_left(left, path)


def forget_left(name, path=None):
    path = path or LEFT
    left = read_left(path)
    if left.pop(name, None) is not None:
        _write_left(left, path)
