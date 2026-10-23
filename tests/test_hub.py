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
