"""`omarchy-cluster chat`: a small terminal chat over the gateway's /v1/chat/completions."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import token as cluster_token

DEFAULT_GATEWAY = "http://127.0.0.1:8020"


def _turn(url, messages, model, max_tokens, token):
    """One non-streaming chat request. Returns (reply text, tokens per second)."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    rq = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(rq, timeout=600) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise RuntimeError("gateway %s answered %d %s" % (url, e.code, detail)) from e
    except OSError as e:
        raise RuntimeError("cannot reach the gateway at %s: %s" % (url, e)) from e
    return res["choices"][0]["message"]["content"], (res.get("timings") or {}).get("tokens_per_s")


def repl(url=DEFAULT_GATEWAY, model="omarchy-cluster", max_tokens=256, system=None, token=None,
         read=input, write=print):
    """Read a line, send the whole conversation, print the reply. /reset clears the
    history; /exit, /quit or end of input leaves."""
    head = [{"role": "system", "content": system}] if system else []
    history = []
    while True:
        try:
            line = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if line in ("/exit", "/quit"):
            return
        if line == "/reset":
            history = []
            continue
        if not line:
            continue
        try:
            text, tokps = _turn(url, head + history + [{"role": "user", "content": line}],
                                model, max_tokens, token)
        except RuntimeError as e:
            write("error: %s" % e)
            continue
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": text}]
        write(text)
        if tokps is not None:
            write("[%.1f tok/s]" % tokps)


def main(args):
    repl(args.gateway, model=args.model, max_tokens=args.max_tokens, system=args.system,
         token=cluster_token.read())
