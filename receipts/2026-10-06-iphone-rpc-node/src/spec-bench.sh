#!/usr/bin/env bash
# Speculative decoding: target Qwen3-8B on m1max-linux CPU; draft Qwen3-0.6B on m1max-linux CPU vs on the iPhone (RPC0).
# The draft lm_head stays on m1max-linux (-otd) so 8 KB, not 593 KB of logits, crosses USB per draft token.
# Usage: spec-bench.sh <outdir>
set -euo pipefail
B=~/src/llama.cpp/build-linux-aligned/bin
T=~/models/Qwen3-8B-Q4_K_M.gguf
D=~/models/Qwen3-0.6B-Q8_0.gguf
OUT=$1
mkdir -p "$OUT"
P="Write a Python function that parses an ISO 8601 date string and returns a datetime. Include docstring and tests."
common=(-p "$P" -n 128 -c 4096 --temp 0 --seed 42 -t 8)
spec=(--spec-type draft-simple -md "$D")
sha256sum "$T" "$D" > "$OUT/models.sha256"
"$B/llama-completion" -m "$T" "${common[@]}" -no-cnv > "$OUT/target.txt" 2> "$OUT/target.log"
echo "== target alone"; grep -E " eval time" "$OUT/target.log"
"$B/llama-speculative-simple" -m "$T" "${spec[@]}" "${common[@]}" > "$OUT/spec-local.txt" 2> "$OUT/spec-local.log"
"$B/llama-speculative-simple" -m "$T" "${spec[@]}" "${common[@]}" --rpc 127.0.0.1:50052 -dev none \
  -devd RPC0 -ngld 99 -otd "^output\.weight=CPU" > "$OUT/spec-phone.txt" 2> "$OUT/spec-phone.log"
for f in spec-local spec-phone; do
  echo "== $f"; grep -aE "decoded|accept|n_draft|n_accept" "$OUT/$f.log" | tail -5
done
