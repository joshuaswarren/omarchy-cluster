#!/usr/bin/env python3
"""What one llama.cpp RPC hop costs per decoded token (run on the llama-server host).

Each config named on the command line: start llama-server, 1 cold + WARM warm greedy chat
requests (temperature 0, cache_prompt false, 64 tokens), stop it. Reports llama-server's
ms per decoded token (median and range over the warm requests) and the TCP connect RTT to
each RPC endpoint (median of 20). Raw responses: OUT (JSON lines).
"""
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
CPU = HOME + "/src/llama.cpp/build-cpu/bin/llama-server"
VK = HOME + "/src/llama.cpp/build-vulkan/bin/llama-server"
MODEL = os.environ.get("MODEL", HOME + "/models/Qwen3-1.7B-Q4_K_M.gguf")
WARM = int(os.environ.get("WARM", "5"))
PORT = 8091
OUT = os.environ.get("OUT", HOME + "/.local/share/omarchy-bench/hopcost.jsonl")
M2 = os.environ.get("M2", "10.10.0.14:50070")
MAC = os.environ.get("MAC", "10.10.0.13:50070")
MAC_TB = os.environ.get("MAC_TB", "10.10.9.1:50070")
LOOP = "127.0.0.1:50071"


def rpc(eps, split=None):
    a = ["--rpc", ",".join(eps), "-dev", ",".join("RPC%d" % i for i in range(len(eps))), "-ngl", "99"]
    return a + (["-ts", split] if split else [])


CONFIGS = {  # name: (binary, args, extra env, endpoints to time)
    "local-vulkan": (VK, ["-dev", "Vulkan0", "-ngl", "99"], {}, []),
    "loopback-rpc-vulkan": (CPU, rpc([LOOP]), {}, [LOOP]),
    "m2-wired": (CPU, rpc([M2]), {}, [M2]),
    "mac-wired": (CPU, rpc([MAC]), {}, [MAC]),
    "mac-thunderbolt": (CPU, rpc([MAC_TB]), {}, [MAC_TB]),
    "mac-wired-no-graph-reuse": (CPU, rpc([MAC]), {"LLAMA_GRAPH_REUSE_DISABLE": "1"}, [MAC]),
    "chain2-m2-mac": (CPU, rpc([M2, MAC], "14,14"), {}, [M2, MAC]),
    "chain3-m2-mac-loop": (CPU, rpc([M2, MAC, LOOP], "10,9,9"), {}, [M2, MAC, LOOP]),
}
BODY = {"messages": [{"role": "user", "content": "Write a haiku about shared memory."}],
        "max_tokens": 64, "temperature": 0, "cache_prompt": False}


def connect_rtt_ms(ep, n=20):
    host, port = ep.rsplit(":", 1)
    out = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            socket.create_connection((host, int(port)), timeout=2).close()
        except OSError:
            return None
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


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


def run(name):
    binary, args, extra, eps = CONFIGS[name]
    rtts = {ep: connect_rtt_ms(ep) for ep in eps}
    env = dict(os.environ, HK_SYSMEM=os.environ.get("HK_SYSMEM", "57000000000"), **extra)
    log = open("/tmp/hop-%s.log" % name, "w")
    proc = subprocess.Popen([binary, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-c", "2048"] + args,
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not healthy(proc):
            print("%-26s FAILED to start (log /tmp/hop-%s.log)" % (name, name), flush=True)
            return False
        ms = []
        with open(OUT, "a") as out:
            for i in range(1 + WARM):
                res = post(BODY)
                m = res["choices"][0]["message"]
                text = (m.get("reasoning_content") or "") + (m.get("content") or "")
                if not text:
                    print("%-26s FAILED: empty answer, nothing to hash" % name, flush=True)
                    return False
                out.write(json.dumps({"config": name, "run": i + 1, "rtt_ms": rtts, "response": res,
                                      "text_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
                ms.append(res["timings"]["predicted_per_token_ms"])
        warm = ms[1:]
        print("%-26s cold %6.2f ms/token | warm median %6.2f ms/token (%.2f to %.2f) = %.1f tok/s | connect RTT %s"
              % (name, ms[0], statistics.median(warm), min(warm), max(warm), 1000 / statistics.median(warm),
                 ", ".join("%s %.3f ms" % (ep, r) if r is not None else "%s unreachable" % ep for ep, r in rtts.items())),
              flush=True)
        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    ok = [run(n) for n in sys.argv[1:]]
    sys.exit(0 if all(ok) else 1)
