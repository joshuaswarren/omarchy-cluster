import io
import json
import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import hub


def request(body, headers=None, path="/v1/heartbeat"):
    handler = object.__new__(hub.Handler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(body)), **(headers or {})}
    handler.rfile = io.BytesIO(body)
    handler.registry = hub.Registry()
    handler._json = Mock()
    return handler


@pytest.mark.parametrize("body", [
    b"[]", b"null", b"42", b'"node"', b"{", b"\xff",
    b'{"name": ["node"]}', b'{"name": {"node": 1}}', b'{"name": 12}',
    b'{"name": "node", "seq": null}', b'{"name": "node", "seq": []}',
    b'{"name": "node", "seq": 1e1000}', b'{"seq": 1}',
])
def test_invalid_heartbeat_returns_400_without_mutating_registry(body):
    handler = request(body)
    handler.do_POST()
    assert handler._json.call_args.args[0] == 400
    assert handler.registry.nodes() == {}


@pytest.mark.parametrize("length", ["-1", "65537", "invalid"])
def test_invalid_length_is_rejected_before_reading(length):
    handler = request(b"{}", {"Content-Length": length})
    handler.rfile = Mock()
    handler.do_POST()
    assert handler._json.call_args.args[0] == 400
    handler.rfile.read.assert_not_called()
    assert handler.registry.nodes() == {}


@pytest.mark.parametrize("body, headers, name, seq", [
    ({"name": "body-node", "seq": 7}, {}, "body-node", 7),
    ({"seq": "8"}, {"X-Node-Name": "header-node"}, "header-node", 8),
    ({"name": "body-node"}, {"X-Node-Name": "header-node"}, "body-node", 0),
])
def test_valid_heartbeat_preserves_name_and_sequence_contract(body, headers, name, seq):
    handler = request(json.dumps(body).encode(), headers)
    handler.do_POST()
    handler._json.assert_called_once_with(200, {"ok": True})
    assert handler.registry.nodes()[name]["seq"] == seq


def test_unknown_route_does_not_read_or_register():
    handler = request(b"ignored", path="/not-a-heartbeat")
    handler.rfile = Mock()
    handler.do_POST()
    handler._json.assert_called_once_with(404, {"error": "not found"})
    handler.rfile.read.assert_not_called()
    assert handler.registry.nodes() == {}


def test_registry_exact_expiry_boundary_and_refresh(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(hub.time, "monotonic", lambda: clock[0])
    registry = hub.Registry()
    assert registry.nodes() == {}
    registry.beat("node", 1)
    clock[0] = 12.999
    assert registry.up_names() == {"node"}
    clock[0] = 13.0
    assert registry.nodes()["node"] == {"age_s": 3.0, "seq": 1, "up": False}
    registry.beat("node", 2)
    assert registry.nodes()["node"] == {"age_s": 0.0, "seq": 2, "up": True}


def test_heartbeat_facts_are_stored_and_trimmed_to_what_the_page_draws():
    body = {"name": "n", "seq": 1, "facts": {"chip": "M1 Ultra", "memory_total_bytes": 64, "load1": 0.5,
                                              "memory_free_bytes": True, "secret": "x", "os": 7}}
    handler = request(json.dumps(body).encode())
    handler.do_POST()
    handler._json.assert_called_once_with(200, {"ok": True})
    assert handler.registry.nodes()["n"]["facts"] == {"chip": "M1 Ultra", "memory_total_bytes": 64, "load1": 0.5}


@pytest.mark.parametrize("facts", [[], "m1", 3, None])
def test_non_object_facts_are_refused_without_registering(facts):
    handler = request(json.dumps({"name": "n", "facts": facts}).encode())
    handler.do_POST()
    assert handler._json.call_args.args[0] == 400
    assert handler.registry.nodes() == {}


def test_agent_beat_facts_survive_the_hub_filter():
    from omarchy_cluster import agent
    sent = agent.beat_facts({})
    assert hub.clean_facts(sent) == sent
    assert set(sent) == set(hub.STR_FACTS) | set(hub.NUM_FACTS)


LINKS = {"pairs": [
    {"a": "m1", "b": "m2", "pinned": {"a_ip": "10.0.0.1", "b_ip": "10.0.0.2"}, "routes": [
        {"a_ip": "192.168.1.1", "b_ip": "192.168.1.2", "media": "wifi", "gbps_a2b": 0.4, "gbps_b2a": 0.4},
        {"a_ip": "10.0.0.1", "b_ip": "10.0.0.2", "media": "thunderbolt", "gbps_a2b": 14.5, "gbps_b2a": 15.0}]},
    {"a": "m1", "b": "m3", "error": "unreachable", "routes": []},
    {"a": "m2", "b": "m3", "pinned": None, "routes": [{"a_ip": "x", "b_ip": "y", "media": "wired"}]},
]}


def test_topology_links_draws_only_the_pinned_route_of_working_pairs():
    assert hub.topology_links(LINKS) == [
        {"a": "m1", "b": "m2", "media": "thunderbolt", "gbps_a2b": 14.5, "gbps_b2a": 15.0}]
    assert hub.topology_links(None) == []


def test_topology_endpoint_joins_heartbeat_nodes_and_pinned_links(monkeypatch):
    monkeypatch.setattr(hub.probe, "load_links", lambda: LINKS)
    monkeypatch.setattr(hub.socket, "gethostname", lambda: "m1.local")
    handler = request(b"", path="/v1/topology")
    handler.registry.beat("m1", 3, {"chip": "M1 Ultra"})
    handler.do_GET()
    code, topo = handler._json.call_args.args
    assert code == 200
    assert topo["host"] == "m1"
    assert topo["nodes"]["m1"]["facts"] == {"chip": "M1 Ultra"}
    assert topo["links"][0]["media"] == "thunderbolt"


def test_root_serves_a_page_that_polls_the_topology_endpoint():
    handler = request(b"", path="/")
    handler._send = Mock()
    handler.do_GET()
    code, ctype, body = handler._send.call_args.args
    assert (code, ctype) == (200, "text/html; charset=utf-8")
    assert b"/v1/topology" in body
