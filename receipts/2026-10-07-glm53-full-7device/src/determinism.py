#!/usr/bin/env python3
"""Per-device greedy determinism, llama.cpp 65840ed, run on the llama-server host.

For each device named on the command line: start llama-server with every layer on that
one device, send RUNS identical greedy chat requests (temperature 0, cache_prompt false,
64 tokens, top-2 logprobs), stop the server. Prints whether the token sequences match,
the first differing token and the top-1 minus top-2 probability margin there, and the
smallest margin seen in run 1. Raw responses: OUT (JSON lines).
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
CPU = HOME + "/src/llama.cpp/build-cpu/bin/llama-server"
VK = HOME + "/src/llama.cpp/build-vulkan/bin/llama-server"
MODEL = os.environ.get("MODEL", HOME + "/models/Qwen3-1.7B-Q4_K_M.gguf")
RUNS = int(os.environ.get("RUNS", "3"))
PORT = 8090
OUT = os.environ.get("OUT", HOME + "/.local/share/omarchy-bench/determinism.jsonl")
ESP = os.environ.get("ESP", "10.10.0.20")
MAC = os.environ.get("MAC", "10.10.0.13:50070")
M2 = os.environ.get("M2", "10.10.0.14:50070")
DEVICES = {
    "x86-cuda": (CPU, ["--rpc", ESP + ":50052", "-dev", "RPC0", "-ngl", "99"]),
    "x86-cpu": (CPU, ["--rpc", ESP + ":50053", "-dev", "RPC0", "-ngl", "99"]),
    "mac-metal": (CPU, ["--rpc", MAC, "-dev", "RPC0", "-ngl", "99"]),
    "m2-vulkan": (CPU, ["--rpc", M2, "-dev", "RPC0", "-ngl", "99"]),
    "m1max-vulkan-local": (VK, ["-dev", "Vulkan0", "-ngl", "99"]),
    "host-cpu": (CPU, ["-dev", "none", "-ngl", "0"]),
}
BODY = {"messages": [{"role": "user", "content": "Write a haiku about shared memory."}],
        "max_tokens": 64, "temperature": 0, "cache_prompt": False,
        "logprobs": True, "top_logprobs": 2}


def post(body):
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def healthy(proc, deadline_s=300):
    end = time.time() + deadline_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def margin(entry):
    tops = sorted((math.exp(t["logprob"]) for t in entry["top_logprobs"]), reverse=True)
    return tops[0] - (tops[1] if len(tops) > 1 else 0.0)


def test(name):
    binary, args = DEVICES[name]
    env = dict(os.environ, HK_SYSMEM=os.environ.get("HK_SYSMEM", "57000000000"))
    log = open("/tmp/det-%s.log" % name, "w")
    proc = subprocess.Popen([binary, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-c", "2048"] + args,
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not healthy(proc):
            print("%-20s FAILED to start (log /tmp/det-%s.log)" % (name, name), flush=True)
            return False
        runs = []
        with open(OUT, "a") as out:
            for i in range(RUNS):
                res = post(BODY)
                out.write(json.dumps({"device": name, "run": i + 1, "response": res}) + "\n")
                runs.append(res["choices"][0]["logprobs"]["content"])
        toks = [[e["token"] for e in r] for r in runs]
        if not all(toks) or not "".join(toks[0]):
            print("%-20s FAILED: empty token list, nothing to compare" % name, flush=True)
            return False
        first = None
        for i in range(1, len(toks)):
            for k, (a, b) in enumerate(zip(toks[0], toks[i])):
                if a != b:
                    first = k if first is None else min(first, k)
                    break
        mins = min(margin(e) for e in runs[0])
        if first is None:
            print("%-20s identical x%d: yes (%d tokens); smallest top-2 margin in run 1: %.4f"
                  % (name, RUNS, len(toks[0]), mins), flush=True)
        else:
            print("%-20s identical x%d: NO; first difference at token %d (%r vs %r), top-2 margin there %.4f"
                  % (name, RUNS, first, toks[0][first], next(t[first] for t in toks[1:] if t[first] != toks[0][first]),
                     margin(runs[0][first])), flush=True)
        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    ok = all([test(n) for n in sys.argv[1:]])
    sys.exit(0 if ok else 1)
