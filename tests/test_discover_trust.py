"""#87 review (P1): discovery must never send the cluster token to an advertiser that has not joined."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import bootstrap, discover
from omarchy_cluster import token as cluster_token

SECRET = "review-only-cluster-secret"
FACTS = {"hostname": "peer", "interfaces": []}


class Rogue(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        type(self).seen.append(self.headers.get("X-Cluster-Token"))
        body = json.dumps(FACTS).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def advertiser(tmp_path, monkeypatch):
    """A loopback listener that records the token header it is sent, plus empty trust files."""
    Rogue.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Rogue)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(cluster_token, "read", lambda: SECRET)
    monkeypatch.setattr(discover.platform, "node", lambda: "this-machine")
    monkeypatch.setattr(discover, "HOSTFILE", str(tmp_path / "hosts"))
    monkeypatch.setattr(discover, "JOINED", str(tmp_path / "joined"))
    yield server.server_port, tmp_path
    server.shutdown()
    server.server_close()


def advertise(monkeypatch, port, name="peer", ip="127.0.0.1"):
    monkeypatch.setattr(discover, "browse", lambda timeout: {
        name: {"name": name, "host": name + ".local", "ip": ip, "ips": [ip], "port": port}})


def peer(nodes, name="peer"):
    return nodes[name]


def test_an_unjoined_advertiser_gets_no_token_and_no_request(advertiser, monkeypatch):
    port, _ = advertiser
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == []  # not even an unauthenticated request: nothing it says is believed
    assert node["facts"] is None
    assert "not joined" in node["error"] and "omarchy-cluster join peer" in node["error"]


def test_a_joined_node_at_its_recorded_address_gets_the_token(advertiser, monkeypatch):
    port, _ = advertiser
    discover.remember_joined("peer", ["127.0.0.1"])
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == [SECRET]
    assert node["facts"] == FACTS


def test_a_rogue_that_copies_a_joined_name_from_another_address_gets_nothing(advertiser, monkeypatch):
    port, _ = advertiser
    discover.remember_joined("peer", ["10.9.9.9"])  # joined at another address
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == []
    assert node["facts"] is None


def test_a_hostfile_entry_is_trusted_because_the_operator_wrote_it(advertiser, monkeypatch):
    port, tmp = advertiser
    (tmp / "hosts").write_text("peer=127.0.0.1\n")
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == [SECRET]
    assert node["facts"] == FACTS


def test_remembering_a_joined_node_twice_keeps_one_line_and_merges_addresses(advertiser):
    _, tmp = advertiser
    discover.remember_joined("peer", ["10.0.0.5"])
    discover.remember_joined("peer", ["10.0.0.5", "192.168.1.5"])
    assert discover.read_joined() == {"peer": {"10.0.0.5", "192.168.1.5"}}
    assert len((tmp / "joined").read_text().splitlines()) == 1


def test_bootstrap_records_the_node_it_installed_with_its_addresses(advertiser):
    _, tmp = advertiser
    calls = []

    def ssh(target, command, stdin=None):
        calls.append(command)
        if "sys.version_info" in command:
            return 0, "ok\n", ""
        if command.startswith("hostname -s"):
            return 0, "peer\n", ""
        if command.startswith("uname -s"):
            return 0, "Linux\n", ""
        return 0, "", ""

    src = tmp / "src"
    (src / "omarchy_cluster").mkdir(parents=True)
    (src / "omarchy_cluster" / "__init__.py").write_text("x = 1\n")
    bootstrap.bootstrap("jw@127.0.0.1", ssh=ssh, src_dir=str(src), token="tok", say=lambda *_: None,
                        check=lambda host, port: True)
    assert discover.read_joined()["peer"] >= {"127.0.0.1"}
