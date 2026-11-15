"""A laptop has Wi-Fi and wired addresses at once, and one of them may not be reachable from the machine running serve. With no probe
pinning a route, the address picked for a node must be one that answers, not just the first one its facts list. Real sockets on
loopback: 127.0.0.2 refuses the connection, 127.0.0.1 has the listener."""
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import cli, probe


@pytest.fixture
def agent_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener.getsockname()[1]


def node(port, *ips, added=None):
    return {"ip": added or ips[0], "port": port, "facts": {"interfaces": [{"ips": [{"ip": ip}]} for ip in ips]}}


def test_a_dead_first_address_loses_to_the_one_that_answers(agent_port, monkeypatch):
    monkeypatch.setattr(probe, "load_links", lambda path=None: None)
    far = node(agent_port, "127.0.0.2", "127.0.0.1")
    assert cli._pick_route_ip(far, [{"node": "here"}, {"node": "far"}], "far") == "127.0.0.1"


def test_three_or_more_ranks_pick_the_live_address_too(agent_port):
    assert cli._first_ip(node(agent_port, "127.0.0.2", "127.0.0.1")) == "127.0.0.1"


def test_the_address_the_node_was_found_at_goes_first_when_it_answers(agent_port):
    assert cli._first_ip(node(agent_port, "127.0.0.1", "127.0.0.3", added="127.0.0.1")) == "127.0.0.1"
    assert cli._first_ip(node(agent_port, "127.0.0.2", "127.0.0.1", added="127.0.0.1")) == "127.0.0.1"


def test_when_nothing_answers_it_exits_naming_every_address_tried(monkeypatch):
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    far = node(port, "127.0.0.2", "127.0.0.3")
    far["name"] = "far"
    with pytest.raises(SystemExit) as stopped:
        cli._first_ip(far)
    assert stopped.value.code == "no address of far answers on port %d (tried: 127.0.0.2, 127.0.0.3)" % port
    monkeypatch.setattr(probe, "load_links", lambda path=None: None)
    with pytest.raises(SystemExit) as picked:  # the same through the route pick
        cli._pick_route_ip({k: v for k, v in far.items() if k != "name"}, [{"node": "here"}, {"node": "far"}], "far")
    assert str(port) in picked.value.code and "far" in picked.value.code


def test_the_ring_says_nothing_was_started():
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    far = node(port, "127.0.0.2")
    far["name"] = "far"
    with pytest.raises(SystemExit) as ring:
        cli._ring_ip(cli._first_ip, far)
    assert ring.value.code.endswith("(tried: 127.0.0.2); nothing was started")


def two_rpc_nodes_second_dead(tmp_path, monkeypatch, cannot_stop=()):
    """alpha's rpc-server is already running when beta turns out to have no address that answers. Plumbing only: the agent calls and
    the rpc handshake are stubbed; the address pick, the state file and the stop calls are the real code."""
    import argparse
    from omarchy_cluster import iosnode

    monkeypatch.setenv("HOME", str(tmp_path))
    nodes = {"alpha": {**node(8025, "10.0.0.1"), "name": "alpha"}, "beta": {**node(8025, "10.0.0.2"), "name": "beta"}}
    started, stopped = [], []

    def agent_post(target, path, payload, timeout=30):
        started.append(target["ip"])
        return {"pid": 4000 + len(started), "cmd": ["rpc-server", "-t", "4"], "env": []}

    def stop(target, pid):
        stopped.append((target["ip"], pid))
        return {"error": "no answer"} if target["ip"] in cannot_stop else {}

    monkeypatch.setattr(cli.discover, "discover_nodes", lambda *a, **k: nodes)
    monkeypatch.setattr(cli, "_agent_post", agent_post)
    monkeypatch.setattr(cli, "_answers", lambda ip, port: ip == "10.0.0.1")
    monkeypatch.setattr(cli, "_stop_rank_via_agent", stop)
    monkeypatch.setattr(iosnode, "wait_rpc", lambda *a, **k: {})
    monkeypatch.setattr(probe, "load_links", lambda path=None: None)
    args = argparse.Namespace(rpc_node=["alpha", "beta"], rpc_binary=[], rpc_env=[], rpc_threads=[], rpc_node_port=50060, rpc_cache=False,
                              links=None, port=8020, engine_port=8031)
    with pytest.raises(SystemExit) as stopping:
        cli._serve_llamacpp_nodes(args, {})
    assert started == ["10.0.0.1", "10.0.0.2"]
    return stopping.value.code, stopped, tmp_path / ".local/state/omarchy-cluster/serve.json"


def test_two_rpc_nodes_where_the_second_has_no_live_address_stop_both_and_say_so(tmp_path, monkeypatch):
    code, stopped, state = two_rpc_nodes_second_dead(tmp_path, monkeypatch)
    assert sorted(stopped) == [("10.0.0.1", "4001"), ("10.0.0.2", "4002")]
    assert code == "no address of beta answers on port 8025 (tried: 10.0.0.2); the 2 rpc-server(s) it had started were stopped"
    assert not state.exists()


def test_a_server_that_cannot_be_stopped_is_named_and_the_state_file_stays_for_omarchy_cluster_stop(tmp_path, monkeypatch):
    code, stopped, state = two_rpc_nodes_second_dead(tmp_path, monkeypatch, cannot_stop={"10.0.0.1"})
    assert code == ("no address of beta answers on port 8025 (tried: 10.0.0.2); "
                    "the rpc-server(s) still running: 10.0.0.1 (pid 4001); run omarchy-cluster stop")
    assert state.exists()
