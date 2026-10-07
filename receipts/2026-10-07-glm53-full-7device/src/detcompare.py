"""Logprob-level determinism from determinism.jsonl.

Within a device: the largest absolute change in any chosen token's logprob (and in the
top-2 logprobs) across its runs; 0.0 means the runs repeated bit for bit as far as the
JSON shows. Across devices: where run 1's tokens first differ, and the largest logprob gap
over the shared prefix.
"""
import json
import sys
from collections import defaultdict

runs = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    runs[r["device"]].append(r["response"]["choices"][0]["logprobs"]["content"])


def seq(run):
    return [e["token"] for e in run]


def lps(e):
    return [e["logprob"]] + [t["logprob"] for t in e["top_logprobs"]]


print("within device (3 runs each):")
for dev, rs in runs.items():
    worst = 0.0
    same_tokens = all(seq(r) == seq(rs[0]) for r in rs)
    for r in rs[1:]:
        for a, b in zip(rs[0], r):
            for x, y in zip(lps(a), lps(b)):
                worst = max(worst, abs(x - y))
    print("  %-20s runs %d, tokens identical %s, max logprob change %.3g" % (dev, len(rs), same_tokens, worst))

print("across devices (run 1 vs run 1):")
devs = list(runs)
for i in range(len(devs)):
    for j in range(i + 1, len(devs)):
        a, b = runs[devs[i]][0], runs[devs[j]][0]
        k = next((n for n, (x, y) in enumerate(zip(seq(a), seq(b))) if x != y), None)
        upto = k if k is not None else min(len(a), len(b))
        gap = max((abs(a[n]["logprob"] - b[n]["logprob"]) for n in range(upto)), default=0.0)
        print("  %-20s vs %-20s first token difference %s, max logprob gap before it %.3g"
              % (devs[i], devs[j], "none" if k is None else k, gap))
