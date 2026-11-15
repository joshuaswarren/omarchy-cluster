# P7 exit run: packet106 (isolated TLS agents, per-rank disk gate, 48-layer pipeline over two Macs)

Date: 2026-11-14, 20:59:06 to 21:01:37 CST (151 s), run by lane dhp7. Third attempt; the first two are listed below.
Machines: rank 0 on jw16 (M1 Max, 62 GB), rank 1 on jwm2 (M2 Max, 94 GB), the controller and gateway on jwm1 (M1). Sizes are as `omarchy-cluster status` prints them. Addresses, host names and home paths are removed from the raw files and replaced by these three names.
Packet: `packet106.py`, sha256 `f0056144da7c5acebc489e364820076037762be3bb267cf41479eff69c4522b8`, prepared owner `ab2982ba`.
Pinned inputs (`raw/packet-facts.txt`): omarchy-cluster head `ade3075e72a7291052983a765ca01fa71fd45ab3`, darkhorse head `52403b5144964a9e9a689911ed79b98cc23c97dc`, model `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` revision `6e302ea604ad9ab206367e2c501d1571023e7b6d`.

## Result
The run exited 0. Both ranks loaded the model from a private cache that had only the pinned metadata, the three greedy completions are identical, the sampled completion ran, and all three hosts ended CLEAN.
One criterion is **not met**: the split was the explicit `--split 24,24`, not a measured one (`layer_ms` is `null` in the gateway status). That is issue #108.
jw16 **downloaded 17.18 GB fresh** for this run (17,181,071,994 B in four shards).

## Per-criterion result
"Expected" is what the packet and its task asked for, before the run.

| criterion | expected | observed | evidence |
|---|---|---|---|
| greedy completions identical | 3 of 3 the same text | PASS: text sha256 `ab20ecc794a60cfe4286943d5e3d03f3454f6749d36ebeedcd26bb6324bd9248` for jobs 1, 2, 3 (ids `chatcmpl-omarchy-1794711685`, `-1794711688`, `-1794711690`); 19.59, 17.67, 16.40 tok/s. Request: temperature 0, max_tokens 48 | `raw/controller/greedy-{1,2,3}.json`, rank 1 job lines in `rank-1.json`, request body in `raw/packet-facts.txt` |
| sampled completion | one answer | PASS: "Red and blue.", sha256 `4f94ad1b8d4e496dfb57fda884dea2b2b9ef84e5edf0b91c3fa66929e48e72f7`, id `chatcmpl-omarchy-1794711691`, 22.44 tok/s. Request: temperature 0.8, top_p 0.95, seed 7, max_tokens 32 | `raw/controller/sampled.json`, `raw/packet-facts.txt` |
| split | measured from json-only sizing | **UNMET**: explicit `[24, 24]`, `layer_ms: null`; jw16 holds layers 24-47, jwm2 layers 0-23 | `raw/controller/gateway-status.json`, rank logs; issue #108 |
| jwm2 downloads at most 11.90 GB | files 1, 2 and 4 only | PASS: 3 files, 11,904,184,575 B (5,321,473,414 + 5,366,644,780 + 1,216,066,381) | `rank-1.json` weights list; blob sizes in `raw/blob-mtimes.txt` |
| jw16 downloads all four shards | 17.18 GB | PASS: 4 files, 17,181,071,994 B, fetched in 1 min 40 s | `rank-0.json` log and weights list; `raw/blob-mtimes.txt` |
| jwm2 download time | none set | 1 min 34 s for 3 files | `rank-1.json` log |
| terminal state | CLEAN on all three | PASS: jw16, jwm2 and jwm1 all `CLEAN` (no owned process, no port holder, ports closed, unit inactive, rank leases resolved) | `raw/supervisor/terminal-{jw16,jwm2,jwm1}.json` (the files were named by ssh alias; renamed here), `cleanup-phase.json` has no errors |
| wall time | inside the work deadline of 1750 s | PASS: 151 s outer unit (service runtime 2 min 30.7 s), controller unit 2 min 11.6 s | `raw/supervisor/budgets.json` (elapsed 15.9 + remaining 1734.1 = 1750.0; controller_runtime 1200; cleanup_s 300), `remote.log`; 1750 = 1770 - 20 in `raw/packet-facts.txt` |
| per-rank disk gate | refuse below shard bytes + 5 GB (jw16 22.18 GB, jwm2 16.90 GB) | NOT SHOWN BY A LOG LINE: the gate prints only when it refuses, and no rank log has such a line; the fetches completed, so neither rank refused | thresholds in `raw/packet-facts.txt` |

Two warnings in the raw logs, neither fatal: rank 1 (jwm2) prints that its GPU heap (101,202,313,216 B) is above the kernel's 75,901,734,912 B per-process limit; and rank 1 polled rank 0's engine three times with "connection refused" before it was up.

## Cold cache
- The packet's `cold_cache()` runs inside the `_qualify` step on each rank before the run, and refuses any private cache that holds more than the pinned metadata files and `refs/main`. The supervisor does not keep that step's output, so **there is no pass line in this receipt**. What shows it: the run continued past `_qualify` on both ranks (it stops there when the check fails, which is how attempt 2 ended), and the files below.
- `raw/blob-mtimes.txt` lists the modification times of the blobs, read at 21:02 CST before the caches were deleted. The four jw16 shard blobs were written between 21:00:57 and 21:01:13, the three jwm2 shard blobs between 21:00:51 and 21:01:07, all after the run began at 20:59:06. The metadata blobs on both ranks are stamped 20:59:32. The rank logs show the downloads happening inside the run (jw16 "Fetching 4 files" 1:40, jwm2 "Fetching 3 files" 1:34).
- jw16: my Nov 13 prefetch of the four shards in the real HF cache was deleted at 20:49 CST with the coordinator's approval, so that jw16 fetched them fresh (free space 12,881,215,488 B before, 28,940,001,280 B after; shard blobs freed 17,181,071,994 B; lane notes, see `raw/free-space.txt`). A hard-link seed of the private cache was tried first and `cold_cache()` rejected it at 20:47 CST (attempt 2).
- After the run both private caches were deleted: jwm2 at 21:03 and jw16 at 21:08 CST. Sizes and free space before and after are in `raw/free-space.txt` and the two `rm-*.out` files. The last number in `rm-jwm2-private-2.out` (15,386,226,688 B free) was read right after the delete, before the filesystem settled; 25 s later free space was 19,940,089,856 B.

## Earlier attempts (all failed before the run began)
1. 20:27 to 20:33 CST: jw16's disk gate refused (needs 17.18 GB + 5.00 GB; 13.37 GB free). All hosts CLEAN.
2. 20:47:35 to 20:47:40 CST (`raw/p106-launch3.log` shows the launch and rc=1 only): `cold_cache()` on jw16 refused the hard-linked seed. The refusal text ("cold cache contains unaccounted files or weights") is in lane report 127, not in this receipt. All hosts CLEAN.
Older failures: 15:49 (`layer_ms`), 15:58 (ssh `_rank-receipt`), 16:05 (shard set, XDG environment); each fixed in the packet before this run.

## Not checked
- The automated rows of `scripts/p7-exit-test.sh` have not been run for this receipt.
- No manager readback is claimed.
- The measured split (issue #108) was not exercised.
- Throughput was not benchmarked beyond the four requests above.
- The pass line of `cold_cache()` was not kept (see Cold cache).

## Raw evidence
`raw/controller/` (engine log, plan, completions with headers, serve.json, rank receipts, stop.txt), `raw/supervisor/` (budgets.json, remote.log, terminal-*.json, cleanup-phase.json), `raw/p106run-console.log`, `raw/p106-launch3.log`, `raw/blob-mtimes.txt`, `raw/free-space.txt`, `raw/packet-facts.txt`, `raw/rm-*.out`.
The earlier `status-after.txt` (a link table of the whole lab) was dropped from the receipt.
