"""Issue 98: the token travels inside TLS to a node whose key was pinned at join, and to nobody else."""
import json
import os
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from omarchy_cluster import client, tls

TOKEN = "s3cret-token"


class Listener:
    """A server on a loopback port that records every request line and header it is sent."""

    def __init__(self, cert=None, key=None):
        seen = self.seen = []

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(dict(self.headers))
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        if cert:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            self.srv.socket = ctx.wrap_socket(self.srv.socket, server_side=True)
        self.port = self.srv.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


def test_a_pinned_node_answers_and_receives_the_token(tmp_path):
    crt, key = tls.ensure_cert(str(tmp_path / "node"))
    with Listener(crt, key) as node:
        facts = client.fetch_facts("127.0.0.1", node.port, token=TOKEN, pin=tls.fingerprint(crt))
    assert facts == {"ok": True}
    assert node.seen[0]["X-Cluster-Token"] == TOKEN


def test_a_rogue_tls_listener_at_a_recorded_address_gets_no_token(tmp_path):
    real_crt, _ = tls.ensure_cert(str(tmp_path / "real"))
    rogue_crt, rogue_key = tls.ensure_cert(str(tmp_path / "rogue"))
    with Listener(rogue_crt, rogue_key) as rogue:
        with pytest.raises(client.PinMismatch):
            client.fetch_facts("127.0.0.1", rogue.port, token=TOKEN, pin=tls.fingerprint(real_crt))
    assert rogue.seen == []  # the handshake is refused before a single header is written


def test_a_rogue_plain_http_listener_at_a_recorded_address_gets_no_token(tmp_path):
    real_crt, _ = tls.ensure_cert(str(tmp_path / "real"))
    with Listener() as rogue:
        with pytest.raises(OSError):
            client.fetch_facts("127.0.0.1", rogue.port, token=TOKEN, pin=tls.fingerprint(real_crt))
    assert rogue.seen == []


def test_no_pin_means_no_request_and_no_token(tmp_path):
    crt, key = tls.ensure_cert(str(tmp_path / "node"))
    with Listener(crt, key) as node:
        with pytest.raises(client.PinMismatch):
            client.fetch_facts("127.0.0.1", node.port, token=TOKEN, pin=None)
    assert node.seen == []


def test_ensure_cert_keeps_the_same_key_on_every_call(tmp_path):
    first = tls.ensure_cert(str(tmp_path / "n"))
    again = tls.ensure_cert(str(tmp_path / "n"))
    assert first == again and tls.fingerprint(first[0]) == tls.fingerprint(again[0])
    assert (tmp_path / "n" / "agent.key").stat().st_mode & 0o077 == 0


def test_a_plain_http_request_is_refused_even_when_the_endpoint_has_a_pin(tmp_path, monkeypatch):
    """The https handler does not switch off urllib's http one: an http:// URL must never carry the token."""
    import urllib.request
    from omarchy_cluster import pins
    monkeypatch.setattr(pins, "PINS", str(tmp_path / "pins"))
    with Listener() as plain:
        pins.remember(["127.0.0.1"], plain.port, "ab" * 32)
        req = urllib.request.Request("http://127.0.0.1:%d/v1/facts" % plain.port, headers={"X-Cluster-Token": TOKEN})
        with pytest.raises(client.PinMismatch):
            client.urlopen(req, 2)
    assert plain.seen == []


def test_bootstrap_by_hostname_pins_and_checks_the_address_it_resolved(tmp_path, monkeypatch):
    """`bootstrap user@node.local`: the key read over ssh is pinned under the resolved address, so the health check
    and the address list must use that address too, not the spelling the operator typed."""
    from omarchy_cluster import bootstrap, discover, pins
    monkeypatch.setattr(pins, "PINS", str(tmp_path / "pins"))
    monkeypatch.setattr(pins, "LOCAL_CERT_DIR", str(tmp_path / "nolocal"))
    monkeypatch.setattr(discover, "JOINED", str(tmp_path / "joined"))
    monkeypatch.setattr(discover, "_resolve", lambda a: "127.0.0.1" if a == "peer.test" else a)
    crt, key = tls.ensure_cert(str(tmp_path / "node"))
    src = tmp_path / "src" / "omarchy_cluster"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")

    def ssh(target, command, stdin=None):
        if "agent.crt" in command:
            return 0, open(crt).read(), ""
        if command.startswith("uname -s"):
            return 0, "Linux\n", ""
        if command.startswith("hostname -s"):
            return 0, "peer\n", ""
        return 0, "ok\n" if "sys.version_info" in command else "", ""

    with Listener(crt, key) as node:
        name = bootstrap.bootstrap("me@peer.test", ssh=ssh, src_dir=str(tmp_path / "src"), token=TOKEN,
                                   port=node.port, say=lambda *_: None)
    assert name == "peer" and node.seen and f"127.0.0.1:{node.port}" in discover.read_joined()["peer"]


def _fake_run(captured, returncode=0, stderr=b""):
    def run(cmd, **kw):
        captured.append(cmd)
        return type("R", (), {"returncode": returncode, "stdout": b"", "stderr": stderr})()
    return run


def test_ssh_run_checks_the_host_key_whatever_the_operators_ssh_config_says(monkeypatch):
    """The pin is whatever the node at that address presents over ssh: an impostor must fail the host key check."""
    from omarchy_cluster import bootstrap
    captured = []
    monkeypatch.setattr(bootstrap.subprocess, "run", _fake_run(captured))
    bootstrap.ssh_run("me@node", "true")
    cmd = captured[0]
    assert "ControlPath=none" in cmd and "ControlMaster=no" in cmd  # a reused master connection skips the host key check
    assert "StrictHostKeyChecking=yes" in cmd and cmd[cmd.index("StrictHostKeyChecking=yes") - 1] == "-o"
    assert cmd.index("StrictHostKeyChecking=yes") < cmd.index("--")  # an option, not part of the remote command


def test_a_failed_host_key_check_says_to_verify_the_key_once_with_plain_ssh(monkeypatch):
    from omarchy_cluster import bootstrap
    monkeypatch.setattr(bootstrap.subprocess, "run", _fake_run([], 255, b"Host key verification failed.\n"))
    with pytest.raises(bootstrap.BootstrapError, match=r"host key.*run `ssh me@node` once.*verify"):
        bootstrap.ssh_run("me@node", "true")


def test_an_ordinary_ssh_failure_is_returned_not_raised(monkeypatch):
    from omarchy_cluster import bootstrap
    monkeypatch.setattr(bootstrap.subprocess, "run", _fake_run([], 255, b"Connection refused\n"))
    assert bootstrap.ssh_run("me@node", "true") == (255, "", "Connection refused\n")


def test_the_pin_file_is_replaced_atomically_and_never_left_half_written(tmp_path, monkeypatch):
    from omarchy_cluster import pins
    path = str(tmp_path / "pins")
    pins.remember(["10.0.0.1"], 8025, "aa" * 32, path)
    real_replace = pins.os.replace
    seen = []

    def spying_replace(src, dst):
        seen.append((open(src).read(), dst))  # the complete new content is on disk before the swap
        real_replace(src, dst)

    monkeypatch.setattr(pins.os, "replace", spying_replace)
    pins.remember(["10.0.0.2"], 8025, "bb" * 32, path)
    assert seen and seen[0][1] == path and "10.0.0.1:8025" in seen[0][0] and "10.0.0.2:8025" in seen[0][0]
    assert sorted(os.listdir(tmp_path)) == ["pins"]  # no temporary file left behind
