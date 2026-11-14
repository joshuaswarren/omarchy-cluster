"""The agent's TLS identity: one self-signed key per node, pinned by its SHA-256 fingerprint where it is trusted.

No certificate authority and no hostnames: a client that was given the fingerprint at join (over ssh, which
already authenticates the node) accepts that key and nothing else, so a machine that takes over a recorded
address cannot complete the handshake and never sees the token."""
from __future__ import annotations

import hashlib
import os
import ssl
import subprocess

CERT_NAME, KEY_NAME = "agent.crt", "agent.key"


def ensure_cert(directory):
    """(cert path, key path) in `directory`, generated once with the openssl CLI and kept (key mode 0600)."""
    crt, key = os.path.join(directory, CERT_NAME), os.path.join(directory, KEY_NAME)
    if os.path.exists(crt) and os.path.exists(key):
        return crt, key
    os.makedirs(directory, exist_ok=True)
    old = os.umask(0o077)
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                        "-nodes", "-keyout", key, "-out", crt, "-days", "36500", "-subj", "/CN=omarchy-cluster-agent"],
                       check=True, capture_output=True)
    finally:
        os.umask(old)
    return crt, key


def fingerprint_der(der):
    return hashlib.sha256(der).hexdigest()


def fingerprint(cert_path):
    """The pin for a PEM certificate file."""
    with open(cert_path) as f:
        return fingerprint_der(ssl.PEM_cert_to_DER_cert(f.read()))


def server_context(directory):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(*ensure_cert(directory))
    return ctx
