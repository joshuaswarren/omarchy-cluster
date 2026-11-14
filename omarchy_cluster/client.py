"""Control-plane HTTP helpers for talking to agents."""
from __future__ import annotations

import json
import urllib.request

from . import token as cluster_token


def _headers(extra=None, token=None):
    """Request headers; the agent wants the cluster token on every route."""
    headers = dict(extra or {})
    tok = token or cluster_token.read()
    if tok:
        headers["X-Cluster-Token"] = tok
    return headers


def fetch_facts(ip, port, timeout=4, token=None):
    req = urllib.request.Request("http://%s:%d/v1/facts" % (ip, port), headers=_headers(token=token))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post(ip, port, path, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://%s:%d%s" % (ip, port, path), data=data,
        headers=_headers({"Content-Type": "application/json"}))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
