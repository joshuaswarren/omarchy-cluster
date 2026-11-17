import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import probe


def _route(media, gbps=None, thunderbolt=False, errors=None):
    r = {
        "a_node": "a", "b_node": "b",
        "a_iface": "en0", "a_ip": "10.0.0.1",
        "b_iface": "en0", "b_ip": "10.0.0.2",
        "media": media,
        "thunderbolt": thunderbolt,
        "rtt_tcp_ms": 0.5, "rtt_udp_ms": 0.4,
        "gbps_a2b": gbps, "gbps_b2a": gbps,
    }
    if errors:
        r["errors"] = errors
    return r


def test_cable_measured_30gbps_is_fastest():
    """Fixture: a Thunderbolt pair measured at 30 Gb/s labels 'cable — fastest'."""
    assert probe.link_label(_route("wired", gbps=30.0, thunderbolt=True)) == "cable — fastest"


def test_wired_measured_2_5gbps_is_fast():
    assert probe.link_label(_route("wired", gbps=2.5)) == "network — fast"


def test_wifi_measured_200mbps_is_slower():
    assert probe.link_label(_route("wifi", gbps=0.2)) == "Wi-Fi — slower"


def test_unmeasured_link_never_says_fastest():
    """'Fastest' is a measurement, never an interface-name guess."""
    label = probe.link_label(_route("wired", gbps=None, thunderbolt=True))
    assert "fastest" not in label
    assert "cable" in label


def test_failed_measurement_says_check_the_cable():
    label = probe.link_label(_route("wired", gbps=None, errors=["unreachable"]))
    assert label == "slow or broken — check the cable"


def test_zero_measured_says_check_the_cable():
    label = probe.link_label(_route("wired", gbps=0.0, errors=["blast: refused"]))
    assert label == "slow or broken — check the cable"


def test_tailscale_labels_as_network():
    label = probe.link_label(_route("tailscale", gbps=1.0))
    assert "network" in label


def test_class_attached_by_probe():
    """candidate_routes records a class on every route for the label table."""
    a = {"name": "a", "interfaces": [
        {"name": "en0", "ips": [{"ip": "10.0.0.1"}], "wireless": False, "thunderbolt": True}]}
    b = {"name": "b", "interfaces": [
        {"name": "en0", "ips": [{"ip": "10.0.0.2"}], "wireless": False, "thunderbolt": True}]}
    routes = probe.candidate_routes(a, b)
    assert len(routes) == 1
    assert routes[0]["link_class"] == "cable"
