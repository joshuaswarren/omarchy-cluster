#!/usr/bin/env python3
"""Measure a loaded omarchy-cluster llama.cpp engine straight at llama-server.

1 cold + N warm greedy chat requests, cache_prompt false (every request evaluates the
whole prompt). Decode tok/s is llama-server's own timings.predicted_per_second; raw
responses go to OUT as JSON lines. Exits nonzero on any failed request. Stops the
cluster at the end unless --keep.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

SERVER = os.environ.get("SERVER", "http://127.0.0.1:8032")
ENGINE = os.environ.get("ENGINE", "http://127.0.0.1:8031")
WARM = int(os.environ.get("WARM", "5"))
OUT = os.environ.get("OUT", os.path.expanduser("~/measure-%d.jsonl" % int(time.time())))
PROMPT = "Write a haiku about shared memory."


def post(path, body, timeout=3600):
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(deadline_s=3600):
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(SERVER + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    sys.exit("llama-server did not become healthy")


def main():
    print("$ waiting for llama-server health", flush=True)
    t0 = time.time()
    wait_health()
    print("healthy after %.0f s more" % (time.time() - t0), flush=True)
    try:
        with urllib.request.urlopen(ENGINE + "/status", timeout=10) as r:
            st = json.loads(r.read())
        print("engine %s, tensor_split %s" % (st.get("engine"), st.get("tensor_split")), flush=True)
    except OSError as e:
        print("engine status unavailable: %s" % e, flush=True)
    rows, shas = [], set()
    with open(OUT, "a") as out:
        for i in range(1 + WARM):
            label = "cold" if i == 0 else "warm %d" % i
            print("\n$ request %d (%s): %r, temperature 0, max_tokens 64, cache_prompt false"
                  % (i + 1, label, PROMPT), flush=True)
            res = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64,
                "temperature": 0, "reasoning_format": "none", "cache_prompt": False})
            out.write(json.dumps({"request": i + 1, "label": label, "response": res}) + "\n")
            out.flush()
            ch = res["choices"][0]
            text = ch["message"]["content"]
            t = res["timings"]
            sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            shas.add(sha)
            if i == 0:
                print(text, flush=True)
            print("prompt %d tokens in %.1f s | decode %d tokens at %.2f tok/s | finish_reason %s | sha256 %s"
                  % (t["prompt_n"], t["prompt_ms"] / 1000, t["predicted_n"], t["predicted_per_second"],
                     ch.get("finish_reason"), sha), flush=True)
            rows.append(t)
    warm = [r["predicted_per_second"] for r in rows[1:]]
    pms = [r["prompt_ms"] / 1000 for r in rows]
    print("\nwarm decode (n=%d): median %.2f tok/s, range %.2f to %.2f" % (
        len(warm), statistics.median(warm), min(warm), max(warm)))
    print("prompt, %d tokens uncached (n=%d): median %.1f s, range %.1f to %.1f" % (
        rows[0]["prompt_n"], len(pms), statistics.median(pms), min(pms), max(pms)))
    print("same text in all %d requests: %s (raw: %s)" % (len(rows), "yes" if len(shas) == 1 else "NO", OUT))
    if "--keep" not in sys.argv:
        print("\n$ omarchy-cluster stop", flush=True)
        subprocess.run([os.path.expanduser("~/.local/bin/omarchy-cluster"), "stop"])
    return 0 if len(shas) == 1 else 3


if __name__ == "__main__":
    sys.exit(main())
