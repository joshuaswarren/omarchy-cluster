import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import facts, probe
from omarchy_cluster.discover import merge_discovered, read_hostfile


def test_perf_cores_counts_only_the_fastest_cores(tmp_path, monkeypatch):
    """llama.cpp threads must skip efficiency cores: one in the team slows every op to its pace."""
    monkeypatch.setattr(facts.platform, "system", lambda: "Linux")
    for i, cap in enumerate([485, 485] + [1024] * 8):  # M1 Max under Linux: 2 E + 8 P
        (tmp_path / ("cpu%d" % i)).mkdir()
        (tmp_path / ("cpu%d" % i) / "cpu_capacity").write_text("%d\n" % cap)
    (tmp_path / "cpufreq").mkdir()
    assert facts.perf_cores(str(tmp_path)) == 8
    monkeypatch.setattr(facts.os, "cpu_count", lambda: 6)
    assert facts.perf_cores(str(tmp_path / "missing")) == 6


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


def test_iperf_client_targets_the_port_the_server_listens_on(monkeypatch):
    """iperf_server listens on DEFAULT_IPERF_PORT; a client without -p dialled iperf3's
    default 5201, every Linux<->Linux round failed, and Linux pairs never pinned."""
    from omarchy_cluster import agent
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        out = {"end": {"sum_sent": {"bits_per_second": 2e9}, "sum_received": {"bits_per_second": 2e9}}}
        return type("P", (), {"returncode": 0, "stdout": json.dumps(out)})()

    monkeypatch.setattr(agent.subprocess, "run", run)
    assert agent.iperf_client("10.10.2.2", "10.10.2.1", 1)["gbps_sent"] == 2.0
    i = seen["cmd"].index("-p")
    assert seen["cmd"][i + 1] == str(agent.DEFAULT_IPERF_PORT)


def test_probe_checks_routes_in_parallel_and_skips_dead_and_spare_routes(monkeypatch):
    """Liveness for every route runs at once (a dead route costs one connect timeout,
    not two 48 s bandwidth rounds); bandwidth runs only on live decode-eligible routes."""
    import threading
    dead = ifc("eno2", "10.10.9.9")
    facts = {"os": "Arch Linux", "tools": {"iperf3": True}}
    nodes = {"a": {"ip": "10.0.0.1", "port": 8025, "facts": dict(facts, name="a", interfaces=[ETH_A, WLAN])},
             "b": {"ip": "10.0.0.2", "port": 8025, "facts": dict(facts, name="b", interfaces=[ETH_B, dead])}}
    together = threading.Barrier(4, timeout=5)  # 4 routes; only parallel liveness gets past it
    bw = []

    def post(ip, port, path, payload, timeout=60):
        if path == "/v1/rtt":
            together.wait()
            alive = payload["ip"] != "10.10.9.9"
            return {"tcp_ms": 0.3 if alive else None, "udp_ms": 0.3 if alive else None}
        if path == "/v1/iperf-server":
            return {"ok": True}
        if path == "/v1/iperf-client":
            bw.append((payload["bind_ip"], payload["ip"]))
            return {"ok": True, "gbps_sent": 2.0, "gbps_received": 2.0}
        raise AssertionError("unexpected %s" % path)

    monkeypatch.setattr(probe, "post", post)
    monkeypatch.setattr(probe.time, "sleep", lambda s: None)
    links = probe.probe_all(nodes, seconds=0.01)
    pair = links["pairs"][0]
    assert (pair["pinned"]["a_ip"], pair["pinned"]["b_ip"]) == ("10.10.2.1", "10.10.2.2")
    assert sorted(bw) == [("10.10.2.1", "10.10.2.2"), ("10.10.2.2", "10.10.2.1")]
    dead_routes = [r for r in pair["routes"] if r["b_ip"] == "10.10.9.9"]
    assert len(dead_routes) == 2 and all("unreachable" in r.get("errors", []) for r in dead_routes)


def _mcdma_env(monkeypatch, system, outputs, ib_root="/nonexistent", peer="/nonexistent/mcdma"):
    from omarchy_cluster import facts
    monkeypatch.setattr(facts.platform, "system", lambda: system)
    monkeypatch.setattr(facts.shutil, "which", lambda b: "/usr/bin/" + b if b in outputs else None)
    monkeypatch.setattr(facts, "_run", lambda cmd, timeout=6: outputs.get(cmd[0], ""))
    monkeypatch.setattr(facts, "IB_ROOT", ib_root)
    monkeypatch.setattr(facts, "MCDMA_PEER", peer)
    return facts


def test_mcdma_on_macos_needs_rdma_enabled_and_an_rdma_interface(monkeypatch):
    f = _mcdma_env(monkeypatch, "Darwin", {"rdma_ctl": "disabled\n", "ifconfig": "lo0 en0 en2"})
    assert f.mcdma() == {"available": False, "reason": "rdma_ctl status: disabled", "devices": []}
    f = _mcdma_env(monkeypatch, "Darwin", {"rdma_ctl": "enabled\n", "ifconfig": "lo0 en2 rdma_en2"})
    assert f.mcdma() == {"available": True, "reason": "rdma_ctl enabled", "devices": ["rdma_en2"]}
    f = _mcdma_env(monkeypatch, "Darwin", {"ifconfig": "lo0"})
    assert f.mcdma()["reason"] == "no rdma_ctl"


def test_mcdma_on_linux_needs_an_active_port_and_the_peer_tool(tmp_path, monkeypatch):
    f = _mcdma_env(monkeypatch, "Linux", {}, ib_root=str(tmp_path / "ib"))
    assert f.mcdma() == {"available": False, "reason": "no /sys/class/infiniband devices", "devices": []}
    port = tmp_path / "ib" / "mlx5_0" / "ports" / "1"
    port.mkdir(parents=True)
    (port / "state").write_text("1: DOWN\n")
    assert f.mcdma()["reason"] == "no ACTIVE infiniband port"
    (port / "state").write_text("4: ACTIVE\n")
    assert f.mcdma()["reason"].startswith("no MCDMA peer tool")
    peer = tmp_path / "mcdma"
    peer.write_text("#!/bin/sh\n")
    peer.chmod(0o755)
    f = _mcdma_env(monkeypatch, "Linux", {}, ib_root=str(tmp_path / "ib"), peer=str(peer))
    assert f.mcdma() == {"available": True, "reason": "ACTIVE port and peer tool", "devices": ["mlx5_0/1"]}


def test_probe_marks_mcdma_eligible_only_when_both_ends_report_it():
    up = {"transports": {"mcdma": {"available": True, "reason": "rdma_ctl enabled", "devices": ["rdma_en2"]}}}
    down = {"transports": {"mcdma": {"available": False, "reason": "rdma_ctl status: disabled", "devices": []}}}
    assert probe.mcdma_pair(up, up) == {"eligible": True, "reason": "both ends report MCDMA"}
    assert probe.mcdma_pair(up, down) == {"eligible": False, "reason": "b: rdma_ctl status: disabled"}
    assert probe.mcdma_pair({}, up)["reason"] == "a: agent does not report transports"


def test_chip_names_follow_the_soc_codes():
    from omarchy_cluster import facts
    assert facts.CHIP_NAMES["t6000"] == "Apple M1 Pro"
    assert facts.CHIP_NAMES["t6002"].startswith("Apple M1 Ultra")
