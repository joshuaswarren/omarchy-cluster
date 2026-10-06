# ClusterRun5 — live stop proof, 2-node decode speed, layer-12 numerics (2026-10-06)

Pair: macstudio (rank 0, M1 Ultra, stock MLX 0.32.2, `/tmp/od-mac-venv313`) +
jw14m2-linux (rank 1, M2 Max, Honeykrisp Vulkan, mlx-omarchy
0.32.4.dev202610042317+b8af62c, `/var/tmp/OmarchyDistributed-venv`), LAN
en0 192.168.10.15 <-> enu1 192.168.10.218 (2.16 Gb/s, rtt 0.41 ms).
Model `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit` (27 layers, bf16
activations). Prompt "Write a haiku about shared memory.", greedy, max_tokens 64.
Full record: lab notebook `entries/ClusterRun5/`, artifacts `artifacts/ClusterRun5/`.

## 1. Stop proof (d9f1187, unchanged)
Runner with `ROUTE_PROFILE=/tmp/links-m2-lan.json PEER_NAME=jw14m2-linux
PEER_IP=192.168.10.218 PEER_KEY=~/.ssh/id_app
LINUX_PY=/var/tmp/OmarchyDistributed-venv/bin/python ENGINE_WAIT_S=10800`,
2026-10-06T00:20:49Z, receipt macstudio `~/.local/share/omarchy-cluster/runs/20261006T002049Z`:
HTTP 200 in 3.447 s, 17 tokens, 9.45 tok/s, text sha `f9bc076f…` on both ranks.
`stopped rank pid 258808 on 192.168.10.218 … 'escalated': False`;
`peer-stop-proof.txt` 0 bytes. Then `omarchy-cluster stop` ("no serve state");
no `omarchy_cluster.rank`/gateway process and no :8020/:8031/:52100 listener on
either host (00:22:50Z).

## 2. Speed
### Where a decode step goes (before: even split 14,13)
A request with `"timing": true` evaluates and timestamps every ring op of the
pipeline forward (both ranks). Mean per decode step, M2 sessions 01:00-01:04Z:

| | ms |
|---|---|
| rank 1 compute on M2 (head + embed + 13 layers) | 101-103 |
| rank 0 compute on Metal (14 layers) | 9.9-22.7 (varies with other mac GPU load) |
| rank 0 head + sample | 1.6-5.7 |
| ring: rank 1 send + all_gather exchange | 0.25-0.33 |
| step | 114-127 |

Biggest cost: the M2's layers, about 8.2 ms per MoE layer (sweep fit:
13 layers 102 ms, 7 52.6, 3 20.7, 1 3.9) against 0.7-1.6 ms per layer on the
M1 Ultra. The ring costs under 0.5 ms per step. HTTP/poll is per request, not
per step: rank 1's 0.5 s poll sleep cost 503-791 ms on a request that arrives
while it sleeps.

### Changes (commit below)
- `serve --split N0,N1` -> `rank --layers N0,N1`: per-rank decoder layer counts
  (rank 0 runs the last layers). Previously `--layers` was ignored and the split
  was always even.
- Only rank 0 samples. It all_gathers its token and rank 1 replaces lm_head with
  that all_gather. Found during the sweep: at split 26,1 rank 1's own Vulkan
  argmax flipped a near-tie, rank 1 fed a different token to layer 0, and the
  ranks' texts diverged (r0 `d8f53137…` vs r1 `a5e70689…`). With the shared
  token every session agrees on both ranks.
- rank 1 long-polls `/prompt/wait` (Condition, 30 s window) instead of sleeping
  0.5 s; `poll_ms` reports publish -> rank 1 fetch.
- `"timing": true` per request; `prefill_ms` is now real (was hard-coded 0.0).

### Before/after, interleaved on the same hosts (01:06:57-01:08:43Z)
`m2-session.sh` sessions, 3 rounds of before/after; per-session jobs 1-3 plain,
job 4 timing. Round 1 ran during heavy macstudio GPU load (after's timing job:
rank 0 compute 198 ms instead of 16), so both arms were slow (before 4.68-7.66,
after 4.37-4.87 tok/s). Rounds 2-3:

| | before: d9f1187, even | after: this commit, `--split 24,3` |
|---|---|---|
| decode tok/s (plain jobs) | 5.56, 9.33, 9.53, 9.56, 9.50, 9.48 | 30.99, 29.68, 28.55, 30.35, 30.81, 30.65 |
| request wall (jobs 2-3) | 3.05-3.12 s | 1.01-1.03 s |
| text sha, rank 0 / rank 1 | `f9bc076f…` / `f9bc076f…` | `f9bc076f…` / `f9bc076f…` |
| step (timing job) | 124-127 ms (d9f1187 cannot time; timing build, same split, 01:00Z) | 37.0 ms: M2 compute 20.0-20.1, Metal 15.9-16.0, head 1.6, ring 0.3 |

Production runner with the final tree, `SPLIT=24,3 GPU_TURN=0` (M2 lock held by
a gpu-turn ticket): 19.92 tok/s / 1.896 s and 30.36 tok/s / 1.157 s, text
`f9bc076f…`, peer-stop-proof empty (`runs/20261006T011303Z`, `…011311Z`).
Six earlier runner pairs at 01:05-01:11Z gave 4.25-9.52 tok/s for both arms;
macstudio was saturated then (load average 6-8, MTLCompilerService 99 %,
oMLX 58 % CPU at 01:06:16Z). Treat runner single shots as noise-limited.

Other splits, final tree: 26,1 52.2-55.4 tok/s (text `fc080ce9…`, both ranks);
20,7 15.7-16.1 tok/s (`81773132…`). 24,3 is the fastest split tested that
keeps the even-split text.

## 3. Numerics: hidden state after layer 12, same 15 prompt ids
Single host each, full model, bf16: M2 Vulkan vs macstudio Metal, plus a
macstudio fp32-activation reference (same 4-bit weights). Each backend is
bitwise deterministic run to run.

| layer 12 | Vulkan vs Metal | Vulkan vs fp32 | Metal vs fp32 |
|---|---|---|---|
| max abs diff | 4.0 (600 vs 596, 1 bf16 ulp) | 9.63 | 9.63 |
| max abs / max abs ref (2880) | 1.39e-3 | 3.34e-3 | 3.34e-3 |
| relative L2, whole tensor | 2.86e-3 | 3.57e-3 | 4.14e-3 |
| relative L2 per token | 0.26-6.0 % (median 1.6 %) | median 1.2 %, max 6.4 % | median 2.0 %, max 6.7 % |
| elements within fp16 tol (rtol 1e-3, atol 1e-5) | 8.4 % | 3.9 % | 2.2 % |

Not within fp16 tolerance: the model computes in bf16 (unit roundoff 3.9e-3),
so neither backend is within fp16 tolerance of fp32 either. The Vulkan-Metal
gap is smaller than each backend's own distance to the fp32 reference: it is
bf16 rounding-order noise, not a Vulkan defect. Both give first token 22258.

## Commits
- `f133975` per-rank layer split, rank 0 sole sampler, long-poll, step timing;
  runner SPLIT/GPU_TURN. `python3 -m pytest -q`: 24 passed. The measured
  "after" rank.py (md5 82617139) differs from `f133975` only in the module
  docstring.
