# Seeded sampling on a split: lab receipt for DH-P7-11 (PR #82)

Date: 2026-11-13 evening (US Central), three runs; the third is the result.

## Setup

- Node A: 16-inch M1 Max, 62 GB. Node B: 14-inch M2 Max, 94 GB. Both on the lab Linux build, wired link between them (measured 1.8 Gb/s, 0.24 ms rtt).
- Model: Qwen2.5-1.5B-Instruct-4bit (28 layers, tied embeddings). A full copy is on both nodes.
- Rank code: this PR at `e1a5df71d61dc21faffee8204fdf1185deb0fbf3`, `rank.py` sha256 `95fbe615997ad49627dbfae52a7a558499bebb9226afdd60300669eb5867ea71` on both nodes. The installed copy was not edited; a wrapper puts the PR's `src` first on the Python path.
- Confounders: the two nodes run different MLX wheel builds (node A `0.32.4.dev202611101859+7ec53476`, node B `0.32.4.dev202611110914+17a9718a`), mlx-lm 0.31.3 with a 0.32.2 overlay on both.
- Command: `data/run.sh` (scrubbed copy). Arms: `whole` (the planner placed the whole model on node B) and `split` (`--stages 2`, measured split 27 layers on node A, 1 layer on node B). Per arm, five requests: greedy, then temperature 0.8, top_p 0.95 with seed 7, seed 7 again, seed 8. Prompt "Write a short story opening about a lighthouse keeper.", 48 new tokens.

## Result (run 3, `data/final/`)

| Request | whole | split |
|---|---|---|
| greedy | `In the misty dawn, a lone lighthouse keeper stood on the rocky cliff, ...` | same text as whole |
| seed 7 (first) | `In the dim, misty glow of the early morning, William Strider peered out ...` | `In the dim, misty glow of the early morning, William McCalla took his wooden crutch ...` |
| seed 7 (second) | identical to the first | identical to the first |
| seed 8 | `The lighthouse keeper's job was to keep his light shining brightly ...` | `The lighthouse keeper's job was to keep his light shining brightly ... prepare himself, and start the oil lamps ...` |

- No request was rejected.
- Same seed, same placement: identical text, in both arms.
- Another seed: different text, in both arms.
- Sampled text differs from greedy text, in both arms.
- Split greedy equals whole greedy over 48 tokens.
- Whole and split give different text for the same seed. They share the opening "In the dim, misty glow of the early morning, William" and then differ. The nodes use different chips, so the logits differ in the last bits and the seeded draw lands on another token. A seed fixes the text for one placement, not across placements.

## Two defects the lab runs found

1. Run 1 (`data/defect-run1/`): split sampled text was garbled. Cause: the code made every rank sample, so ranks other than 0 fed their layers a token that rank 0 did not pick. Fix: followers take rank 0's token.
2. Run 2 (not kept; same garbled text): this model ties its embeddings, so there was no `lm_head` to swap and no rank shared a token. Each rank sampled for itself. Fix: a follower swaps `embed_tokens.as_linear` for the token share.

Both fixes have tests (`tests/test_rank_sampling.py`), including one against a real tied Qwen2 module.

## Not measured

- Speed. Sampled decode ran at 9.6 to 11.2 tokens/s against 81 to 150 for greedy (issue 97). This receipt makes no speed claim.
- A split of three or more ranks.
- A model with an untied `lm_head` on the lab nodes (the unit test covers it; the lab run did not).
- Memory use per rank beyond the `rank0.log` lines.

## Files

`data/final/*.json` are the raw gateway replies; `*.rank0.log` and `*.serve.txt` are the rank and serve output. Hostnames, home paths and addresses are replaced by `node-a`, `node-b`, `ring-ip-a`, `ring-ip-b`, `ifc-a`, `ifc-b`. `SHA256SUMS` covers every file under `data/`.
