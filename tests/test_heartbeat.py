import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import agent, hub


def test_default_sender_payload_is_accepted_by_hub_without_a_socket(monkeypatch):
    stop = Mock()
    stop.is_set.side_effect = [False, True]
    replies = []
    monkeypatch.setattr(agent.socket, "gethostname", lambda: "worker.example.test")
    monkeypatch.setattr(agent.time, "time", lambda: 123.0)

    def deliver(req, timeout):
        assert req.full_url == "http://unused.invalid/v1/heartbeat"
        assert timeout == 2
        body = json.loads(req.data)
        assert body == {"name": "worker", "seq": 4, "t": 123.0}
        handler = object.__new__(hub.Handler)
        handler.path = "/v1/heartbeat"
        handler.headers = {"Content-Length": str(len(req.data))}
        handler.rfile = io.BytesIO(req.data)
        handler.registry = hub.Registry()
        handler._json = Mock()
        handler.do_POST()
        replies.append(handler._json.call_args.args[0])
        assert handler.registry.nodes()["worker"]["seq"] == 4
        return Mock()

    send = Mock(side_effect=deliver)
    monkeypatch.setattr(agent.urllib.request, "urlopen", send)
    agent.hub_heartbeat("http://unused.invalid/", SimpleNamespace(seq=4), stop)
    send.assert_called_once()
    assert replies == [200]
    stop.wait.assert_called_once_with(1.0)


def test_explicit_sender_name_is_not_replaced_by_hostname(monkeypatch):
    stop = Mock()
    stop.is_set.side_effect = [False, True]
    send = Mock(return_value=Mock())
    monkeypatch.setattr(agent.urllib.request, "urlopen", send)
    monkeypatch.setattr(agent.socket, "gethostname", lambda: "other.example.test")
    agent.hub_heartbeat("http://unused.invalid", SimpleNamespace(seq=8), stop, name="chosen")
    assert json.loads(send.call_args.args[0].data)["name"] == "chosen"


def test_sender_retries_after_transport_error_and_observes_stop(monkeypatch):
    stop = Mock()
    stop.is_set.side_effect = [False, False, True]
    send = Mock(side_effect=[OSError("offline"), Mock()])
    monkeypatch.setattr(agent.urllib.request, "urlopen", send)
    agent.hub_heartbeat("http://unused.invalid", SimpleNamespace(seq=1), stop)
    assert send.call_count == 2
    assert stop.wait.call_count == 2


@pytest.mark.parametrize("name, expected", [(None, "worker"), ("chosen", "chosen")])
def test_startup_uses_same_identity_for_hub_and_advertisement(monkeypatch, name, expected):
    heartbeat = Mock()
    stop = Mock()
    thread = Mock()
    advertise = Mock(return_value=None)
    server = Mock()
    monkeypatch.setattr(agent, "Heartbeat", Mock(return_value=heartbeat))
    monkeypatch.setattr(agent.threading, "Event", Mock(return_value=stop))
    monkeypatch.setattr(agent.threading, "Thread", thread)
    monkeypatch.setattr(agent, "advertise_start", advertise)
    monkeypatch.setattr(agent, "ThreadingHTTPServer", Mock(return_value=server))
    monkeypatch.setattr(agent.socket, "gethostname", lambda: "worker.example.test")
    monkeypatch.setattr(agent.Handler, "agent", None)

    agent.serve(name=name, hub="http://unused.invalid")

    heartbeat_threads = [call.kwargs for call in thread.call_args_list if call.kwargs["target"] is agent.hub_heartbeat]
    assert len(heartbeat_threads) == 1
    assert heartbeat_threads[0]["args"] == ("http://unused.invalid", heartbeat, stop, expected)
    advertise.assert_called_once_with(expected, agent.DEFAULT_HTTP_PORT)
    stop.set.assert_called_once()
