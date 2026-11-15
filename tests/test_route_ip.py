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


def test_when_nothing_answers_the_first_listed_address_is_kept_so_the_later_check_names_the_failure():
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    assert cli._first_ip(node(port, "127.0.0.2", "127.0.0.3")) == "127.0.0.2"
