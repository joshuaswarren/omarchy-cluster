Muse Spark Open 2-bit, 62 of 62 layers, 5 ranks on 4 Apple Silicon machines; text NOT CHECKED for coherence; quality NOT MEASURED

Muse Spark Open (2-bit mixed build: experts 2-bit g128, attention and embedding 4-bit g64, lm_head 6-bit g64), all 62 layers, real weights, no pruning of
experts, no 3-bit, nothing substituted. Split in rank order 24,12,12,12,2: rank 0 Studio (the last layers + lm_head), ranks 1 and 2 two processes on the M2,
rank 3 the 16-inch, rank 4 the 13-inch (the first layers + the embedding). Every layer is on an Apple Silicon GPU through MLX; no layer is on a CPU.
Pipeline prune (PR 199) on, MLX_DISABLE_COMPILE=1, default kernels, raw completion route, greedy (the gateway accepts temperature 0 only).
TTFT is the gateway's prefill_ms and decode tokens per second is the gateway's tokens_per_s. One throw-away request runs before the captures.
Memory lines come from this run's rank logs only (files newer than the start marker). Files: studio-run.log, limits-vs-weights.txt, hk-sysmem.txt,
gateway-status.json, capture/, capture.log, rank-load-lines.txt. A killed driver leaves the prune flag file on the nodes; it is harmless.
Guard for this run (attempt 2): Studio per-minute swap line 14,000 MB (approved 07:20 CDT); total swap line 32,000 MB; +14,000 MB over the baseline the watch read at its start; internal floor 20 GB; NVMe floor 25 GB. Attempt 1 (line 8,000) was stopped at 12:03:47Z at +10,173 MB in 20 s; the ladder at 24 layers jumped +6,969 MB. The 22-layer Studio share was the last one that ran inside the original 3,072 line; the 24-layer share ran only under a raised line.
Studio swap at launch: total = 11264.00M  used = 10351.88M  free = 912.12M  (encrypted)
NEXT after reviewer 1 coherence read: CAPTURE_LENGTHS='128 256' capture-full62.sh http://<studio-ip>:18020 $RUN_ARTIFACT/capture local/muse-2bit 5 4 '<placement>'
RANKS LEFT RUNNING for the coherence check; stop with pN-stop.sh
