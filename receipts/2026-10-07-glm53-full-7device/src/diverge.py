"""First differing token per pair of greedy outputs (re-tokenized with the GLM-5.3 tokenizer).

Usage: python3 diverge.py measure.jsonl tokenizer.json (zai-org/GLM-5.3 tokenizer.json)
"""
import json
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[2])
rows = [json.loads(line) for line in open(sys.argv[1])]
texts = [r["response"]["choices"][0]["message"]["content"] for r in rows]
ids = [tok.encode(t, add_special_tokens=False).ids for t in texts]
print("re-tokenized lengths:", [len(x) for x in ids], "(server predicted_n:",
      [r["response"]["timings"]["predicted_n"] for r in rows], ")")


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


print("first differing token index (0-based) per pair:")
for i in range(len(ids)):
    print(" req%d" % (i + 1), [first_diff(ids[i], ids[j]) if i != j else "-" for j in range(len(ids))])
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        k = first_diff(ids[i], ids[j])
        print("req%d vs req%d at token %2d: after %r -> %r vs %r" % (
            i + 1, j + 1, k, tok.decode(ids[i][max(0, k - 5):k]),
            tok.decode(ids[i][k:k + 1]), tok.decode(ids[j][k:k + 1])))
