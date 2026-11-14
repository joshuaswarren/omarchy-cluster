"""Which key each recorded agent address must present: {ip:port -> SHA-256 of the node's TLS certificate}.

Pins are written when a node is joined (the key is read over ssh) and nowhere else. This machine's own agent
is pinned by reading its own certificate. An address with no pin is never sent the cluster token."""
from __future__ import annotations

import os

from . import tls

PINS = os.path.expanduser("~/.local/state/omarchy-cluster/pins")
LOCAL_CERT_DIR = os.path.expanduser("~/.config/omarchy-cluster")
LOOPBACK = ("127.0.0.1", "::1", "localhost")


def read(path=None):
    pins = {}
    try:
        with open(path or PINS) as f:
            for line in f:
                endpoint, _, pin = line.strip().partition(" ")
                if endpoint and pin:
                    pins[endpoint] = pin
    except FileNotFoundError:
        pass
    return pins


def remember(addresses, port, pin, path=None):
    """Pin `pin` for every address:port. Re-joining a node replaces its old key."""
    path = path or PINS
    pins = read(path)
    for a in addresses:
        pins["%s:%d" % (a, port)] = pin
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as f:
        for endpoint in sorted(pins):
            f.write("%s %s\n" % (endpoint, pins[endpoint]))
    os.replace(tmp, path)  # a reader sees the old file or the new one, never half of either


def lookup(ip, port, path=None):
    """The pin for this endpoint, or None. Loopback is this machine's own agent: its certificate file."""
    if ip in LOOPBACK:
        crt = os.path.join(LOCAL_CERT_DIR, tls.CERT_NAME)
        if os.path.exists(crt):
            return tls.fingerprint(crt)
    return read(path).get("%s:%d" % (ip, int(port)))
