"""`omarchy-cluster chat`: a small terminal chat over the gateway's /v1/chat/completions."""
from __future__ import annotations

import os
from datetime import datetime

from . import client
from . import token as cluster_token

DEFAULT_GATEWAY = "http://127.0.0.1:8020"
JOURNAL = "~/.local/state/omarchy-cluster/chat.log"


def _reply(res):
    """(reply text, tokens per second or None) from one gateway body; any other 200 shape is an error.

    The text must be a string. A timing that is not a number is dropped: the reply is still good."""
    try:
        text = res["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = None
    if not isinstance(text, str):
        raise RuntimeError("gateway answered 200 with an unexpected reply shape "
                           "(no choices[0].message.content string)")
    timings = res.get("timings")
    tokps = timings.get("tokens_per_s") if isinstance(timings, dict) else None
    if isinstance(tokps, bool) or not isinstance(tokps, (int, float)):
        tokps = None
    return text, tokps


def repl(url=DEFAULT_GATEWAY, model="omarchy-cluster", max_tokens=256, system=None, token=None,
         read=input, write=print, clock=datetime.now):
    """Read a line, send the whole conversation, print the reply. /reset clears the
    history; /exit, /quit or end of input leaves. No history or memory files are kept;
    the session appends one journal line."""
    head = [{"role": "system", "content": system}] if system else []
    history = []
    turns = 0
    while True:
        try:
            line = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/exit", "/quit"):
            break
        if line == "/reset":
            history = []
            continue
        if not line:
            continue
        try:
            text, tokps = _reply(client.chat(url, head + history + [{"role": "user", "content": line}],
                                             model, max_tokens, token))
        except RuntimeError as e:
            write("error: %s" % e)
            continue
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": text}]
        write(text)
        if tokps is not None:
            write("[%.1f tok/s]" % tokps)
        turns += 1
    path = os.path.expanduser(JOURNAL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write("%s  chat  model=%s gateway=%s turns=%d\n"
                % (clock().strftime("%Y-%m-%d %H:%M:%S"), model, url, turns))


def main(args):
    repl(args.gateway, model=args.model, max_tokens=args.max_tokens, system=args.system,
         token=cluster_token.read())
