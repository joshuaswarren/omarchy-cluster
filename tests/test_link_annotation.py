import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import cli, facts, probe


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_linux_thunderbolt_is_the_interface_bound_to_the_thunderbolt_bus(tmp_path):
    bus = tmp_path / "bus" / "thunderbolt"
    pci = tmp_path / "bus" / "pci"
    for d in (bus, pci):
        d.mkdir(parents=True)
    net = tmp_path / "net"
    for name, bus_dir in (("tb-bridge", bus), ("eth0", pci), ("thunderbolt0", pci)):
        dev = tmp_path / "devices" / name
        dev.mkdir(parents=True)
        os.symlink(bus_dir, dev / "subsystem")
        (net / name).mkdir(parents=True)
        os.symlink(dev, net / name / "device")
    assert facts._linux_thunderbolt("tb-bridge", str(net))
    assert facts._linux_thunderbolt("thunderbolt0", str(net))  # default thunderbolt-net name
    assert not facts._linux_thunderbolt("eth0", str(net))
    assert not facts._linux_thunderbolt("missing", str(net))


def test_linux_rdma_netdevs_are_active_hardware_ports_only(tmp_path):
    ib = str(tmp_path)
    for dev, state, ndev in (("mlx5_0", "4: ACTIVE", "eth2"), ("mlx5_1", "1: DOWN", "eth3"),
                             ("rxe0", "4: ACTIVE", "eth4")):
        _write("%s/%s/ports/1/state" % (ib, dev), state + "\n")
        _write("%s/%s/ports/1/gid_attrs/ndevs/0" % (ib, dev), ndev + "\n")
    assert facts._linux_rdma_netdevs(ib) == {"eth2"}
    assert facts._linux_rdma_netdevs(str(tmp_path / "none")) == set()


NETWORKSETUP = """Hardware Port: Ethernet
Device: en0

Hardware Port: Thunderbolt 5
Device: en6

Hardware Port: Thunderbolt Bridge
Device: bridge0
"""
IFCONFIG = """en0: flags=8863<UP> mtu 1500
\tinet 192.168.1.5 netmask 0xffffff00 broadcast 192.168.1.255
en6: flags=8863<UP> mtu 1500
\tinet 10.9.0.1 netmask 0xfffffffc broadcast 10.9.0.3
bridge0: flags=8863<UP> mtu 1500
\tinet 10.8.0.1 netmask 0xfffffffc broadcast 10.8.0.3
"""


def test_macos_interfaces_flag_thunderbolt_ports_and_rdma_interfaces(monkeypatch):
    monkeypatch.setattr(facts, "_run", lambda cmd, timeout=6: {"networksetup": NETWORKSETUP, "ifconfig": IFCONFIG}[cmd[0]])
    monkeypatch.setattr(facts, "mcdma", lambda: {"available": True, "devices": ["rdma_en6"]})
    got = {i["name"]: (i["thunderbolt"], i["rdma"]) for i in facts._mac_interfaces()}
    assert got == {"en0": (False, False), "en6": (True, True), "bridge0": (True, False)}


def test_macos_rdma_off_marks_no_interface(monkeypatch):
    monkeypatch.setattr(facts, "_run", lambda cmd, timeout=6: {"networksetup": NETWORKSETUP, "ifconfig": IFCONFIG}[cmd[0]])
    monkeypatch.setattr(facts, "mcdma", lambda: {"available": False, "devices": ["rdma_en6"]})
    assert not any(i["rdma"] for i in facts._mac_interfaces())


def iface(name, ip, **flags):
    return {"name": name, "ips": [{"ip": ip, "prefix": 30}], "speed_mbps": None, "wireless": False, **flags}


def route(a_flags, b_flags):
    a = {"name": "a", "interfaces": [iface("en6", "10.9.0.1", **a_flags)]}
    b = {"name": "b", "interfaces": [iface("en6", "10.9.0.2", **b_flags)]}
    return probe.candidate_routes(a, b)[0]


def test_route_is_thunderbolt_or_rdma_only_when_both_ends_are():
    both = route({"thunderbolt": True, "rdma": True}, {"thunderbolt": True, "rdma": True})
    assert both["thunderbolt"] and both["rdma_capable"]
    half = route({"thunderbolt": True, "rdma": True}, {"thunderbolt": True})
    assert half["thunderbolt"] and not half["rdma_capable"]
    plain = route({}, {"thunderbolt": True})
    assert not plain["thunderbolt"] and not plain["rdma_capable"]


def test_annotation_does_not_change_media_or_decode_eligibility():
    r = route({"thunderbolt": True, "rdma": True}, {"thunderbolt": True, "rdma": True})
    assert r["media"] == "wired" and r["eligible_for_decode"]


def test_links_table_tags_thunderbolt_and_rdma_routes():
    plain = route({}, {})
    tb = route({"thunderbolt": True}, {"thunderbolt": True})
    rdma = route({"thunderbolt": True, "rdma": True}, {"thunderbolt": True, "rdma": True})
    lines = {}
    for name, r in (("plain", plain), ("tb", tb), ("rdma", rdma)):
        r["gbps_a2b"] = r["gbps_b2a"] = 10.0
        lines[name] = cli.links_table({"pairs": [{"a": "a", "b": "b", "routes": [r]}]})
    assert "TB" not in lines["plain"] and "RDMA" not in lines["plain"]
    assert "TB" in lines["tb"] and "RDMA" not in lines["tb"]
    assert "TB" in lines["rdma"] and "RDMA" in lines["rdma"]


def test_links_table_says_why_a_pair_has_no_rdma():
    pair = {"a": "a", "b": "b", "routes": [], "mcdma": {"eligible": False, "reason": "a: rdma_ctl status: disabled"}}
    assert "rdma: not available (a: rdma_ctl status: disabled)" in cli.links_table({"pairs": [pair]})
    pair["mcdma"] = {"eligible": True, "reason": "both ends report MCDMA"}
    assert "rdma: eligible (both ends report MCDMA)" in cli.links_table({"pairs": [pair]})
