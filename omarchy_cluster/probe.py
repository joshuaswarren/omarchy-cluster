"""Pairwise route enumeration, measurement, and the route table (links.json)."""
from __future__ import annotations

import ipaddress
import json
import os
import threading
import time

from .agent import DEFAULT_HTTP_PORT, DEFAULT_IPERF_PORT, DEFAULT_SINK_PORT
from .client import post

TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
STATE_DIR = os.path.expanduser("~/.local/state/omarchy-cluster")


def _is_tailscale(ip):
    try:
        return ipaddress.ip_address(ip) in TAILSCALE_NET
    except ValueError:
        return False


def iface_ips(node_facts):
    out = []
    for ifc in (node_facts or {}).get("interfaces", []):
        for a in ifc.get("ips", []):
            out.append((ifc, a["ip"]))
    return out


def candidate_routes(a, b):
    """Every (interface, interface) route between two nodes, deduped by IP pair."""
    routes = []
    seen = set()
    for ifa, ipa in iface_ips(a):
        for ifb, ipb in iface_ips(b):
            if (ipa, ipb) in seen:
                continue
            seen.add((ipa, ipb))
            wireless = bool(ifa.get("wireless") or ifb.get("wireless"))
            ts = _is_tailscale(ipa) or _is_tailscale(ipb)
            routes.append({
                "a_node": a["name"], "b_node": b["name"],
                "a_iface": ifa["name"], "a_ip": ipa,
                "b_iface": ifb["name"], "b_ip": ipb,
                "media": "tailscale" if ts else ("wifi" if wireless else "wired"),
                "eligible_for_decode": not (wireless or ts),
                "rtt_tcp_ms": None, "rtt_udp_ms": None,
                "gbps_a2b": None, "gbps_b2a": None,
            })
    return routes


def _agent_addr(nodes, name):
    node = nodes[name]
    return node["ip"], node.get("port", DEFAULT_HTTP_PORT)


def check_live(route, nodes):
    """RTT from a's agent to b over this route. False (and an "unreachable" error) when
    no TCP connect succeeds; the agent gives up after one connect timeout."""
    a_ip, a_port = _agent_addr(nodes, route["a_node"])
    b_port = _agent_addr(nodes, route["b_node"])[1]
    rtt = post(a_ip, a_port, "/v1/rtt",
               {"ip": route["b_ip"], "port": b_port, "bind_ip": route["a_ip"]}, timeout=15)
    route["rtt_tcp_ms"] = rtt.get("tcp_ms")
    route["rtt_udp_ms"] = rtt.get("udp_ms")
    if route["rtt_tcp_ms"] is None:
        route.setdefault("errors", []).append("unreachable")
        return False
    return True


def measure_route(route, nodes, seconds=3.0, use_iperf=True):
    """Liveness, then bandwidth both ways if the route is up. Control plane runs from this CLI host."""
    if check_live(route, nodes):
        measure_bandwidth(route, nodes, seconds, use_iperf)
    return route


def measure_bandwidth(route, nodes, seconds=3.0, use_iperf=True):
    a_ip, a_port = _agent_addr(nodes, route["a_node"])
    b_ip, b_port = _agent_addr(nodes, route["b_node"])
    a_facts = nodes[route["a_node"]].get("facts") or {}
    b_facts = nodes[route["b_node"]].get("facts") or {}
    mac_involved = "macOS" in (a_facts.get("os") or "") or "macOS" in (b_facts.get("os") or "")
    iperf = (use_iperf and not mac_involved
             and a_facts.get("tools", {}).get("iperf3")
             and b_facts.get("tools", {}).get("iperf3"))
    route["method"] = "iperf3" if iperf else "builtin-tcp"

    for fwd, sender, receiver in (("gbps_a2b", (a_ip, a_port), (b_ip, b_port)),
                                  ("gbps_b2a", (b_ip, b_port), (a_ip, a_port))):
        s_ip, s_port = sender
        r_ip, r_port = receiver
        bind = route["a_ip"] if fwd == "gbps_a2b" else route["b_ip"]
        dst = route["b_ip"] if fwd == "gbps_a2b" else route["a_ip"]
        if iperf:
            th = threading.Thread(target=_iperf_round, args=(
                s_ip, s_port, r_ip, r_port, bind, dst, seconds, route, fwd))
        else:
            th = threading.Thread(target=_builtin_round, args=(
                s_ip, s_port, r_ip, r_port, bind, dst, seconds, route, fwd))
        th.start()
        time.sleep(0.3)  # let the receiver side start listening first
        th.join()
    return route


def _iperf_round(s_ip, s_port, r_ip, r_port, bind, dst, seconds, route, fwd):
    errs = []
    server_result = {}

    def server():
        try:
            server_result["r"] = post(r_ip, r_port, "/v1/iperf-server",
                                      {"bind_ip": dst, "seconds": seconds},
                                      timeout=seconds + 45)
        except Exception as e:  # noqa: BLE001
            errs.append("server: %s" % e)

    th = threading.Thread(target=server)
    th.start()
    time.sleep(0.5)
    try:
        res = post(s_ip, s_port, "/v1/iperf-client",
                   {"ip": dst, "bind_ip": bind, "seconds": seconds},
                   timeout=seconds + 45)
    except Exception as e:  # noqa: BLE001
        errs.append("client: %s" % e)
        res = None
    th.join()
    if res and res.get("ok"):
        route[fwd] = round(min(res["gbps_sent"], res["gbps_received"]), 3)
    else:
        route[fwd] = None
        if errs:
            route.setdefault("errors", []).extend(errs)


def _builtin_round(s_ip, s_port, r_ip, r_port, bind, dst, seconds, route, fwd):
    errs = []
    sink = {}

    def sink_serve():
        try:
            sink["r"] = post(r_ip, r_port, "/v1/sink-serve",
                             {"seconds": seconds}, timeout=seconds + 45)
        except Exception as e:  # noqa: BLE001
            errs.append("sink: %s" % e)

    th = threading.Thread(target=sink_serve)
    th.start()
    time.sleep(0.5)
    sent = 0
    try:
        res = post(s_ip, s_port, "/v1/blast",
                   {"ip": dst, "bind_ip": bind, "seconds": seconds},
                   timeout=seconds + 45)
        sent = res.get("sent_bytes", 0)
    except Exception as e:  # noqa: BLE001
        errs.append("blast: %s" % e)
    th.join()
    recv = sink.get("r", {}).get("received_bytes", 0)
    elapsed = max(sink.get("r", {}).get("elapsed_s", seconds), 0.001)
    route[fwd] = round(min(sent, recv) * 8 / 1e9 / elapsed, 3) if min(sent, recv) > 0 else None
    if errs:
        route.setdefault("errors", []).extend(errs)


def measured_gbps(route):
    a2b, b2a = route.get("gbps_a2b"), route.get("gbps_b2a")
    vals = [v for v in (a2b, b2a) if v]
    if not vals:
        return 0.0
    return min(vals)


def rank_routes(routes):
    """Eligible routes first, then bottleneck Gb/s desc, then RTT asc."""
    def key(r):
        rtt = r.get("rtt_tcp_ms")
        return (
            1 if r.get("eligible_for_decode") else 0,
            measured_gbps(r),
            -(rtt if rtt is not None else 1e9),
        )
    return sorted((r for r in routes if measured_gbps(r) > 0), key=key, reverse=True)


def pin_pair(routes):
    ranked = rank_routes(routes)
    return ranked[0] if ranked else None


def mcdma_pair(fa, fb):
    """Whether MCDMA could carry this pair: only when both ends report it available. "soft_transport"
    is true when either end runs a software provider (rxe, siw).
    Recorded in links.json; route choice is unchanged (TCP) until MCDMA is measured."""
    for side, f in (("a", fa), ("b", fb)):
        m = (f or {}).get("transports", {}).get("mcdma")
        if m is None:
            return {"eligible": False, "reason": "%s: agent does not report transports" % side}
        if not m.get("available"):
            return {"eligible": False, "reason": "%s: %s" % (side, m.get("reason"))}
    out = {"eligible": True, "reason": "both ends report MCDMA"}
    if any(((f or {}).get("transports", {}).get("mcdma") or {}).get("soft_transport") for f in (fa, fb)):
        out["soft_transport"] = True
    return out


def probe_all(nodes, seconds=3.0, names=None):
    """Probe every node pair; returns the links dict. Liveness for every route of every
    pair runs in parallel; bandwidth then runs one route at a time (parallel tests would
    share NICs and skew each other), on live decode-eligible routes only, or on the live
    routes of a pair that has no eligible one."""
    from concurrent.futures import ThreadPoolExecutor
    keys = sorted(names or nodes.keys())
    pairs, todo = [], []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            fa, fb = nodes[a].get("facts"), nodes[b].get("facts")
            if not fa or not fb:
                pairs.append({"a": a, "b": b, "error": "missing agent facts", "routes": [],
                              "pinned": None})
                continue
            routes = candidate_routes(fa, fb)
            pairs.append({"a": a, "b": b, "routes": routes, "pinned": None,
                          "mcdma": mcdma_pair(fa, fb)})
            todo += routes

    def live(r):
        try:
            return check_live(r, nodes)
        except Exception as e:  # noqa: BLE001 - record and keep probing
            r.setdefault("errors", []).append(str(e))
            return False

    with ThreadPoolExecutor(max_workers=max(1, min(64, len(todo)))) as ex:
        up = {id(r) for r, ok in zip(todo, ex.map(live, todo)) if ok}
    for pair in pairs:
        alive = [r for r in pair["routes"] if id(r) in up]
        for r in [r for r in alive if r["eligible_for_decode"]] or alive:
            try:
                measure_bandwidth(r, nodes, seconds)
            except Exception as e:  # noqa: BLE001 - record and keep probing
                r.setdefault("errors", []).append(str(e))
        if not pair.get("error"):
            pair["pinned"] = pin_pair(pair["routes"])
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_per_test": seconds, "pairs": pairs}


def links_path(out=None):
    return out or os.path.join(STATE_DIR, "links.json")


def save_links(links, out=None):
    path = links_path(out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(links, f, indent=2)
    return path


def load_links(out=None):
    try:
        with open(links_path(out)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
