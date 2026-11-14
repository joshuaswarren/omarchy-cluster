"""#87 review (P1): discovery must never send the cluster token to an advertiser that has not joined."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import bootstrap, discover, pins, tls
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


def over_tls(server, certdir):
    server.socket = tls.server_context(certdir).wrap_socket(server.socket, server_side=True)
    return server


@pytest.fixture
def advertiser(tmp_path, monkeypatch):
    """A loopback TLS listener that records the token header it is sent, plus empty trust files. Its key is this
    machine's own agent key (loopback is pinned from the local certificate); other addresses have no pin until a test joins them."""
    Rogue.seen = []
    monkeypatch.setattr(pins, "LOCAL_CERT_DIR", str(tmp_path / "cert"))
    monkeypatch.setattr(pins, "PINS", str(tmp_path / "pins"))
    server = over_tls(ThreadingHTTPServer(("127.0.0.1", 0), Rogue), str(tmp_path / "cert"))
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
    assert "not joined" in node["error"] and "omarchy-cluster bootstrap" in node["error"]


def test_a_joined_node_at_its_recorded_address_gets_the_token(advertiser, monkeypatch):
    port, _ = advertiser
    discover.remember_joined("peer", ["127.0.0.1"], port)
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == [SECRET]
    assert node["facts"] == FACTS


def test_a_rogue_that_copies_a_joined_name_from_another_address_gets_nothing(advertiser, monkeypatch):
    port, _ = advertiser
    discover.remember_joined("peer", ["10.9.9.9"], port)  # joined at another address
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == []
    assert node["facts"] is None


def test_a_hostfile_entry_is_trusted_because_the_operator_wrote_it(advertiser, monkeypatch):
    port, tmp = advertiser
    (tmp / "hosts").write_text("peer=127.0.0.1:%d\n" % port)
    advertise(monkeypatch, 1)  # the advertised port is ignored
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == [SECRET]
    assert node["facts"] == FACTS


def test_remembering_a_joined_node_twice_keeps_one_line_and_merges_addresses(advertiser):
    _, tmp = advertiser
    discover.remember_joined("peer", ["10.0.0.5"])
    discover.remember_joined("peer", ["10.0.0.5", "192.168.1.5"])
    assert discover.read_joined() == {"peer": {"10.0.0.5:8025", "192.168.1.5:8025"}}
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
        if "agent.crt" in command:
            return 0, open(tls.ensure_cert(str(tmp / "cert"))[0]).read(), ""
        return 0, "", ""

    src = tmp / "src"
    (src / "omarchy_cluster").mkdir(parents=True)
    (src / "omarchy_cluster" / "__init__.py").write_text("x = 1\n")
    bootstrap.bootstrap("jw@127.0.0.1", ssh=ssh, src_dir=str(src), token="tok", say=lambda *_: None,
                        check=lambda host, port: True)
    assert discover.read_joined()["peer"] >= {"127.0.0.1:8025"}
    assert pins.read()["127.0.0.1:8025"] == tls.fingerprint(str(tmp / "cert" / "agent.crt"))  # the key read over ssh


def test_this_machines_own_advertisement_is_trusted_and_goes_to_loopback(advertiser, monkeypatch):
    port, _ = advertiser
    monkeypatch.setattr(discover, "DEFAULT_PORT", port)
    advertise(monkeypatch, 1, name="this-machine", ip="10.9.9.9")  # nothing joined, nothing in the hosts file
    node = discover.discover_nodes(hostfile=discover.HOSTFILE)["this-machine"]
    assert node["source"] == "local" and node["ip"] == "127.0.0.1"
    assert Rogue.seen == [SECRET] and node["facts"] == FACTS


def test_a_joined_node_is_reached_only_at_the_port_recorded_at_join_time(advertiser, monkeypatch):
    port, _ = advertiser
    discover.remember_joined("peer", ["127.0.0.1"], 1)  # joined on port 1; the advertiser says another port
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == [] and node["facts"] is None


def test_a_hostfile_entry_without_a_port_ignores_the_advertised_port(advertiser, monkeypatch):
    port, tmp = advertiser
    monkeypatch.setattr(discover, "DEFAULT_PORT", 1)
    (tmp / "hosts").write_text("peer=127.0.0.1\n")
    advertise(monkeypatch, port)
    node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    assert Rogue.seen == [] and node["facts"] is None


def test_a_redirect_never_carries_the_token_to_another_listener(advertiser, monkeypatch):
    port, tmp_path = advertiser

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:%d/v1/facts" % port)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    server = over_tls(ThreadingHTTPServer(("127.0.0.1", 0), Redirect), str(tmp_path / "cert"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        discover.remember_joined("peer", ["127.0.0.1"], server.server_port)
        advertise(monkeypatch, server.server_port)
        node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    finally:
        server.shutdown()
        server.server_close()
    assert Rogue.seen == [] and node["facts"] is None and "302" in node["error"]


@pytest.mark.skipif(sys.platform != "linux", reason="needs a second loopback address")
def test_a_machine_that_takes_over_a_joined_address_gets_no_token(advertiser, tmp_path, monkeypatch):
    """Issue 98 risk 2: the node joined at 127.0.0.2 presented another key; whoever answers there now is not it."""
    _, tmp = advertiser
    other_crt, other_key = tls.ensure_cert(str(tmp_path / "joined-node"))
    server = over_tls(ThreadingHTTPServer(("127.0.0.2", 0), Rogue), str(tmp / "cert"))  # the taker-over's own key
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        discover.remember_joined("peer", ["127.0.0.2"], server.server_port)
        pins.remember(["127.0.0.2"], server.server_port, tls.fingerprint(other_crt))
        advertise(monkeypatch, server.server_port, ip="127.0.0.2")
        node = peer(discover.discover_nodes(hostfile=discover.HOSTFILE))
    finally:
        server.shutdown()
        server.server_close()
    assert Rogue.seen == [] and node["facts"] is None and "pinned" in node["error"]
