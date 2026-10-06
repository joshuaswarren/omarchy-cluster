import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import probe
from omarchy_cluster.discover import merge_discovered, read_hostfile


def node(name, ifaces):
    return {"name": name, "interfaces": ifaces}


def ifc(name, ip, prefix=24, wireless=False, speed=1000):
    return {"name": name, "ips": [{"ip": ip, "prefix": prefix}],
            "speed_mbps": speed, "wireless": wireless}


def route(a_if, b_if, a2b=None, b2a=None, tcp=0.3):
    return {
        "a_node": "a", "b_node": "b", "a_iface": a_if["name"], "a_ip": a_if["ips"][0]["ip"],
        "b_iface": b_if["name"], "b_ip": b_if["ips"][0]["ip"],
        "media": "wifi" if (a_if["wireless"] or b_if["wireless"]) else "wired",
        "eligible_for_decode": not (a_if["wireless"] or b_if["wireless"]),
        "rtt_tcp_ms": tcp, "rtt_udp_ms": tcp,
        "gbps_a2b": a2b, "gbps_b2a": b2a,
    }


WLAN = ifc("wlan0", "10.10.1.1", wireless=True)
ETH_A = ifc("en0", "10.10.2.1", speed=10000)
ETH_B = ifc("eno1", "10.10.2.2", speed=2500)
TS_A = ifc("utun7", "100.64.1.1", prefix=10)
TS_B = ifc("tailscale0", "100.64.1.2", prefix=10)


def test_decode_eligibility():
    assert not probe.candidate_routes(node("a", [WLAN]), node("b", [ETH_B]))[0]["eligible_for_decode"]
    assert not probe.candidate_routes(node("a", [TS_A]), node("b", [TS_B]))[0]["eligible_for_decode"]
    assert probe.candidate_routes(node("a", [ETH_A]), node("b", [ETH_B]))[0]["eligible_for_decode"]
    assert probe.candidate_routes(node("a", [TS_A]), node("b", [TS_B]))[0]["media"] == "tailscale"


def test_pin_prefers_eligible_over_faster_wifi():
    routes = [
        route(ETH_A, ETH_B, a2b=1.0, b2a=1.0),
        route(WLAN, ETH_B, a2b=5.0, b2a=5.0),
    ]
    pinned = probe.pin_pair(routes)
    assert pinned["a_iface"] == "en0"


def test_pin_fastest_then_rtt_tiebreak():
    slow = route(ETH_A, ifc("eno2", "10.10.3.2"), a2b=1.0, b2a=1.0, tcp=0.5)
    fast = route(ETH_A, ifc("tb0", "169.254.10.2"), a2b=25.0, b2a=25.0, tcp=5.0)
    assert probe.pin_pair([slow, fast])["b_iface"] == "tb0"
    r1 = route(ETH_A, ifc("eno1", "10.10.2.2"), a2b=2.0, b2a=2.0, tcp=0.9)
    r2 = route(ETH_A, ifc("eno2", "10.10.4.2"), a2b=2.0, b2a=2.0, tcp=0.2)
    assert probe.pin_pair([r1, r2])["b_iface"] == "eno2"


def test_failover_rank_order():
    first = route(ETH_A, ifc("tb0", "169.254.10.2"), a2b=25.0, b2a=25.0)
    second = route(ETH_A, ETH_B, a2b=2.3, b2a=2.35)
    ranked = probe.rank_routes([first, second])
    assert [r["b_iface"] for r in ranked] == ["tb0", "eno1"]  # next measured fallback
    assert probe.pin_pair([second])["b_iface"] == "eno1"  # first failed its probe, absent


def test_unmeasured_routes_never_pinned():
    dead = route(ETH_A, ifc("tb0", "169.254.10.2"), a2b=None, b2a=None)
    live = route(ETH_A, ETH_B, a2b=2.3, b2a=2.35)
    assert probe.pin_pair([dead, live])["b_iface"] == "eno1"


def test_bottleneck_metric_uses_min_direction():
    r = route(ETH_A, ETH_B, a2b=2.36, b2a=0.05)
    assert probe.measured_gbps(r) == 0.05


def test_candidate_routes_dedupe():
    a = node("a", [ETH_A, TS_A])
    b = node("b", [ETH_B, TS_B, ifc("eno2", "10.10.2.2")])
    routes = probe.candidate_routes(a, b)
    keys = [(r["a_ip"], r["b_ip"]) for r in routes]
    assert len(keys) == len(set(keys)) == 4  # 2 a-ips x 3 b-ips - 2 duplicate routes


def test_hostfile_parse_and_merge(tmp_path):
    hf = tmp_path / "hosts"
    hf.write_text("# comment\nmac-a=10.10.2.9\nlinux-b\n")
    overrides = read_hostfile(str(hf))
    assert overrides == {"mac-a": "10.10.2.9", "linux-b": "linux-b"}
    found = {"mac-a": {"name": "mac-a", "host": "mac-a.local", "ip": "1.2.3.4",
                       "port": 8025}}
    nodes = merge_discovered(found, overrides)
    assert nodes["mac-a"]["ip"] == "10.10.2.9"
    assert nodes["mac-a"]["source"] == "hostfile"  # override wins
    assert nodes["linux-b"]["ip"] == "linux-b"  # added, unresolved until fetch
    assert nodes["linux-b"]["source"] == "hostfile"


def test_links_roundtrip(tmp_path):
    links = {"generated": "t", "pairs": [{"a": "a", "b": "b", "routes": [], "pinned": None}]}
    out = str(tmp_path / "links.json")
    assert probe.save_links(links, out) == out
    assert probe.load_links(out) == links
    assert probe.load_links(str(tmp_path / "missing.json")) is None
