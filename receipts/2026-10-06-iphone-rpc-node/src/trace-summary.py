#!/usr/bin/env python3
"""Per-token breakdown from rpc-trace.py output. A token cycle starts at the first SET_TENSOR after a reply.
Columns (ms): host_gap (reply done -> next SET_TENSOR, i.e. client-side work), phone_wait (GET_TENSOR sent ->
first reply byte: phone compute + RTT), reply_xfer (first -> last reply byte), reply_kb."""
import statistics
import sys

rows = [line.rstrip("\n").split("\t") for line in open(sys.argv[1])]
cycles, cur, last_reply = [], None, None
for t, d, what, size in rows:
    t, size = float(t), int(size)
    if d == "c2s" and what == "SET_TENSOR" and (cur is None or cur.get("reply_end")):
        if cur and cur.get("reply_end"):
            cur["next_start"] = t
            cycles.append(cur)
        cur = {"start": t, "cmds": 0, "c2s_bytes": 0, "reply_bytes": 0}
    if cur is None:
        continue
    if d == "c2s":
        cur["cmds"] += 1
        cur["c2s_bytes"] += size
        if what == "GET_TENSOR":
            cur["get"] = t
    elif "get" in cur:
        cur.setdefault("reply_first", t)
        cur["reply_end"] = t
        cur["reply_bytes"] += size

steady = cycles[3:]
def med(f):
    return statistics.median(f(c) for c in steady)
print(f"tokens={len(steady)} cmds/token={med(lambda c: c['cmds'])} c2s_bytes/token={med(lambda c: c['c2s_bytes'])} "
      f"reply_kb/token={med(lambda c: c['reply_bytes']) / 1024:.1f}")
print(f"median ms: host_gap={med(lambda c: (c['next_start'] - c['reply_end']) * 1e3):.2f} "
      f"send={med(lambda c: (c['get'] - c['start']) * 1e3):.2f} "
      f"phone_wait={med(lambda c: (c['reply_first'] - c['get']) * 1e3):.2f} "
      f"reply_xfer={med(lambda c: (c['reply_end'] - c['reply_first']) * 1e3):.2f} "
      f"cycle={med(lambda c: (c['next_start'] - c['start']) * 1e3):.2f}")
