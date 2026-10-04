"""Control-plane HTTP helpers for talking to agents."""
from __future__ import annotations

import json
import urllib.request


def fetch_facts(ip, port, timeout=4):
    with urllib.request.urlopen("http://%s:%d/v1/facts" % (ip, port), timeout=timeout) as r:
        return json.loads(r.read())


def post(ip, port, path, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://%s:%d%s" % (ip, port, path), data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
