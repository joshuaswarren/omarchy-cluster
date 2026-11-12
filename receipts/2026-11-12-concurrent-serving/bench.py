"""Aggregate decode throughput through the gateway at several --max-concurrent values.

Engine: omarchy_cluster.llamacpp_engine on CPU, llama-server with its default 4 slots.
Clients: CLIENTS threads, each sends one greedy completion of MAX_TOKENS tokens.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

REPO = sys.argv[1]
MODEL = sys.argv[2]
LLAMA = sys.argv[3]
CLIENTS = int(os.environ.get("CLIENTS", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
THREADS = os.environ.get("THREADS", "8")
LIMITS = [int(x) for x in os.environ.get("LIMITS", "1,2,4").split(",")]
REPS = int(os.environ.get("REPS", "7"))
ENGINE_PORT, GW_PORT = 8131, 8120
env = dict(os.environ, PYTHONPATH=REPO)

engine = subprocess.Popen(
    [sys.executable, "-m", "omarchy_cluster.llamacpp_engine", "--model", MODEL, "--llama-server", LLAMA,
     "--port", str(ENGINE_PORT), "--server-port", str(ENGINE_PORT + 1), "--ctx", "2048", "--threads", THREADS],
    env=env, stdout=open("/tmp/w7j-bench-engine.log", "w"), stderr=subprocess.STDOUT)


def wait(url, secs=300):
    end = time.time() + secs
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return
        except OSError:
            time.sleep(1)
    raise SystemExit("no answer from " + url)


def one(results, i, port):
    body = json.dumps({"prompt": "Write a long story about a lighthouse keeper number %d." % i,
                       "max_tokens": MAX_TOKENS}).encode()
    rq = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % port, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=600) as r:
        d = json.loads(r.read())
    results[i] = d["usage"]["completion_tokens"]


try:
    wait("http://127.0.0.1:%d/health" % ENGINE_PORT)
    out = []
    gws = {}
    for i, limit in enumerate(LIMITS):
        gws[limit] = subprocess.Popen([sys.executable, "-m", "omarchy_cluster.gateway", "--port", str(GW_PORT + i),
                                       "--engine", "http://127.0.0.1:%d" % ENGINE_PORT, "--max-concurrent", str(limit)],
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for i in range(len(LIMITS)):
            wait("http://127.0.0.1:%d/health" % (GW_PORT + i))
        for rep in range(REPS):
            for i, limit in enumerate(LIMITS):
                results = {}
                ts = [threading.Thread(target=one, args=(results, j, GW_PORT + i)) for j in range(CLIENTS)]
                t0 = time.time()
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                wall = time.time() - t0
                toks = sum(results.values())
                out.append({"max_concurrent": limit, "rep": rep, "clients": CLIENTS, "tokens": toks,
                            "wall_s": round(wall, 2), "agg_tok_s": round(toks / wall, 2),
                            "load1": round(os.getloadavg()[0], 1)})
                print(json.dumps(out[-1]), flush=True)
        import statistics
        for limit in LIMITS:
            v = [o["agg_tok_s"] for o in out if o["max_concurrent"] == limit]
            print("median max_concurrent=%d: %.2f tok/s over %d runs (min %.2f max %.2f)"
                  % (limit, statistics.median(v), len(v), min(v), max(v)), flush=True)
    finally:
        for g in gws.values():
            g.terminate()
            g.wait(10)
finally:
    engine.terminate()
    engine.wait(30)
