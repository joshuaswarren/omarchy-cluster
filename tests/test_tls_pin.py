"""Issue 98: the token travels inside TLS to a node whose key was pinned at join, and to nobody else."""
import json
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
