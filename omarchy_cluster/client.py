"""Control-plane HTTP helpers for talking to agents."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import token as cluster_token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # urllib would forward X-Cluster-Token to the Location host


_opener = urllib.request.build_opener(_NoRedirect)


def urlopen(req, timeout):
    """urlopen for requests that carry the cluster token: a redirect is an error, never followed."""
    return _opener.open(req, timeout=timeout)


def _headers(extra=None, token=None):
    """Request headers; the agent wants the cluster token on every route."""
    headers = dict(extra or {})
    tok = token or cluster_token.read()
    if tok:
        headers["X-Cluster-Token"] = tok
    return headers


def fetch_facts(ip, port, timeout=4, token=None):
    req = urllib.request.Request("http://%s:%d/v1/facts" % (ip, port), headers=_headers(token=token))
    with urlopen(req, timeout) as r:
        return json.loads(r.read())


def post(ip, port, path, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://%s:%d%s" % (ip, port, path), data=data,
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
