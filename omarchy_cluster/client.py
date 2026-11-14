"""Control-plane HTTP helpers for talking to agents."""
from __future__ import annotations

import functools
import http.client
import json
import ssl
import urllib.error
import urllib.request

import hmac

from . import pins, tls
from . import token as cluster_token


class PinMismatch(OSError):
    """The node at this address is not the one whose key was pinned at join (or nothing was pinned): nothing was sent."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # urllib would forward X-Cluster-Token to the Location host


class _PinnedConnection(http.client.HTTPSConnection):
    """Completes the TLS handshake, then compares the peer's key with the pin before any request byte goes out."""

    def __init__(self, *args, pin, **kwargs):
        super().__init__(*args, **kwargs)
        self._pin = pin

    def connect(self):
        super().connect()
        der = self.sock.getpeercert(binary_form=True)
        if not self._pin or not der or not hmac.compare_digest(tls.fingerprint_der(der), self._pin):
            self.sock.close()
            raise PinMismatch("%s:%s does not present the key pinned for it" % (self.host, self.port))


def urlopen(req, timeout, pin=None):
    """urlopen for requests that carry the cluster token: https only, to the pinned key only; a redirect is an error.
    With no pin nothing is sent, not even a connection."""
    if pin is None:
        pin = pins.lookup(req.host.rpartition(":")[0] or req.host, int(req.host.rpartition(":")[2] or 443))
    if not pin:
        raise PinMismatch("no key is pinned for %s: join the node first (`omarchy-cluster bootstrap`)" % req.full_url)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # trust is the pin, checked in _PinnedConnection.connect
    handler = type("_Pinned", (urllib.request.HTTPSHandler,), {
        "https_open": lambda self, r: self.do_open(
            functools.partial(_PinnedConnection, pin=pin), r, context=ctx)})()
    try:
        return urllib.request.build_opener(_NoRedirect, handler).open(req, timeout=timeout)
    except urllib.error.URLError as e:
        if isinstance(e.reason, PinMismatch):
            raise e.reason from None
        raise


def agent_url(ip, port, path=""):
    """The control-plane URL of an agent: always https."""
    return "https://%s:%d%s" % (ip, int(port), path)


def _headers(extra=None, token=None):
    """Request headers; the agent wants the cluster token on every route."""
    headers = dict(extra or {})
    tok = token or cluster_token.read()
    if tok:
        headers["X-Cluster-Token"] = tok
    return headers


def fetch_facts(ip, port, timeout=4, token=None, pin=None):
    req = urllib.request.Request(agent_url(ip, port, "/v1/facts"), headers=_headers(token=token))
    with urlopen(req, timeout, pin) as r:
        return json.loads(r.read())


def post(ip, port, path, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        agent_url(ip, port, path), data=data,
        headers=_headers({"Content-Type": "application/json"}))
    with urlopen(req, timeout) as r:
        return json.loads(r.read())


def chat(url, messages, model="omarchy-cluster", max_tokens=256, token=None, timeout=600):
    """One non-streaming chat completion on a cluster gateway. Returns the parsed body,
    or raises RuntimeError with the gateway's message."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0}).encode()  # greedy only; the gateway refuses the rest (DH-P7-11 pending)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise RuntimeError("gateway %s answered %d %s" % (url, e.code, detail)) from e
    except OSError as e:
        raise RuntimeError("cannot reach the gateway at %s: %s" % (url, e)) from e
