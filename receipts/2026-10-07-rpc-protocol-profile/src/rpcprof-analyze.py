"""Per-token breakdown of llama.cpp RPC decode from rpcproxy.py's log.

A decode token = the commands from one GRAPH_RECOMPUTE/GRAPH_COMPUTE through the reply to the
GET_TENSOR(s) that follow it. Reports medians over decode tokens (RECOMPUTE ones):
commands before the graph, their bytes, GET_TENSOR count and reply bytes, time from graph send
to first reply byte (server compute + start of transfer), reply streaming time, and the client
gap from the last reply byte to the next token's first command.
"""
import json
import statistics
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1])]
by_conn = defaultdict(list)
for r in rows:
    by_conn[r["c"]].append(r)
for conn, rs in by_conn.items():
    if sum(1 for r in rs if r.get("cmd") == "GRAPH_RECOMPUTE") < 10:
        continue
    rs.sort(key=lambda r: r["t"])
    graphs = [i for i, r in enumerate(rs) if r.get("cmd") in ("GRAPH_RECOMPUTE", "GRAPH_COMPUTE")]
    toks = []
    for gi, g in enumerate(graphs):
        if rs[g]["cmd"] != "GRAPH_RECOMPUTE":
            continue
        prev = graphs[gi - 1] if gi else 0
        nxt = graphs[gi + 1] if gi + 1 < len(graphs) else len(rs)
        # commands between the previous token's last reply and this graph
        pre = [r for r in rs[prev + 1:g] if r["dir"] == "cmd" and r["cmd"] != "GET_TENSOR"]
        post = rs[g + 1:nxt]
        gets = [r for r in post if r.get("cmd") == "GET_TENSOR"]
        replies = [r for r in post if r["dir"] == "reply"]
        trailing = [r for r in post if r["dir"] == "cmd" and r["cmd"] != "GET_TENSOR"]
        if not replies:
            continue
        first_cmd_next = min([r["t"] for r in trailing] + [rs[nxt]["t"] if nxt < len(rs) else replies[-1]["t"]])
        toks.append({
            "pre_cmds": len(pre), "pre_bytes": sum(r["bytes"] for r in pre),
            "gets": len(gets), "reply_bytes": sum(r["bytes"] for r in replies),
            "graph_to_reply_ms": (replies[0]["t"] - rs[g]["t"]) * 1e3,
            "reply_stream_ms": (replies[-1]["t"] - replies[0]["t"]) * 1e3,
            "token_ms": ((rs[nxt]["t"] if nxt < len(rs) else replies[-1]["t"]) - rs[g]["t"]) * 1e3,
            "client_gap_ms": max(0.0, first_cmd_next - replies[-1]["t"]) * 1e3,
        })
    print("connection %s: %d decode tokens" % (conn, len(toks)))
    for k in ("pre_cmds", "pre_bytes", "gets", "reply_bytes", "graph_to_reply_ms", "reply_stream_ms",
              "client_gap_ms", "token_ms"):
        v = [t[k] for t in toks]
        print("   %-18s median %10.3f   min %10.3f   max %10.3f" % (k, statistics.median(v), min(v), max(v)))
