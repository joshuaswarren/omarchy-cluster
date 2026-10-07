#!/usr/bin/env python3
"""RPC per-token profile configs on top of hopcost.py (same 1 cold + WARM warm method).
Adds: loopback RPC with the output layer pinned to the host (the cluster's setup), and both
loopback variants routed through rpcproxy.py on :50072 for a per-command log."""
import sys

import hopcost as h

OUT_ON_HOST = ["-ot", r"^output\.weight=CPU"]
PROXY = "127.0.0.1:50072"
h.CONFIGS["loopback-rpc-outhost"] = (h.CPU, h.rpc([h.LOOP]) + OUT_ON_HOST, {}, [h.LOOP])
h.CONFIGS["proxy-rpc"] = (h.CPU, h.rpc([PROXY]), {}, [])
h.CONFIGS["proxy-rpc-outhost"] = (h.CPU, h.rpc([PROXY]) + OUT_ON_HOST, {}, [])
# CPU device over loopback RPC: set_tensor is a memcpy there, so the gap to host-cpu is protocol only
# patched rpc-server (small SET_TENSOR writes async, one sync per command) on :50074
h.CONFIGS["loopback-rpc-async"] = (h.CPU, h.rpc(["127.0.0.1:50074"]), {}, ["127.0.0.1:50074"])
h.CONFIGS["host-cpu"] = (h.CPU, ["-dev", "none", "-ngl", "0"], {}, [])
h.CONFIGS["loopback-rpc-cpu"] = (h.CPU, h.rpc(["127.0.0.1:50073"]), {}, ["127.0.0.1:50073"])

if __name__ == "__main__":
    ok = [h.run(n) for n in sys.argv[1:]]
    sys.exit(0 if all(ok) else 1)
