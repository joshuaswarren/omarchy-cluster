# P7 exit run: packet106 (isolated TLS agents, per-rank disk gate, 48-layer pipeline over two Macs)

Date: 2026-11-14, 20:59:06 to 21:01:37 CST, lane dhp7. Run on the third attempt; the first two are listed below.
Packet: `packet106.py` sha256 prefix `f0056144da7c5ace`, prepared owner `ab2982ba` (re-prep 23 at 20:48:31 CST).
Pinned inputs: omarchy-cluster head `ade3075e72a7291052983a765ca01fa71fd45ab3`, darkhorse head `52403b5144964a9e9a689911ed79b98cc23c97dc`, model `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` revision `6e302ea604ad9ab206367e2c501d1571023e7b6d`.

## Result
The run exited 0. Both ranks loaded the model from a cold private cache, the three greedy completions are identical, the sampled completion ran, and all three hosts ended CLEAN.
One criterion is **not met**: the split was the explicit `--split 24,24`, not a measured one (`layer_ms` is `null` in the gateway status). That is omarchy-cluster issue #108.
jw16 **downloaded 17.18 GB fresh** for this run (17,181,071,994 B in four shards). Its earlier prefetch in the real cache was deleted first (see Cold cache).

## Per-criterion result against what was predicted
| criterion | predicted | observed | machine and command |
|---|---|---|---|
| greedy completions identical | 3 of 3 the same text | PASS: text sha256 `ab20ecc794a60cfe4286943d5e3d03f3454f6749d36ebeedcd26bb6324bd9248` for jobs 1, 2, 3 (ids `chatcmpl-omarchy-1794711685`, `-1794711688`, `-1794711690`); 19.59, 17.67, 16.40 tok/s | jwm1 controller, `raw/jwm1-controller/greedy-{1,2,3}.json`, rank 1 log in `rank-1.json` |
| sampled completion | one answer, top_p path | PASS: "Red and blue.", text sha256 `4f94ad1b8d4e496dfb57fda884dea2b2b9ef84e5edf0b91c3fa66929e48e72f7`, id `chatcmpl-omarchy-1794711691`, 22.44 tok/s | `raw/jwm1-controller/sampled.json` |
| split | measured from json-only sizing | **UNMET**: explicit `[24, 24]`, `layer_ms: null` (`raw/jwm1-controller/gateway-status.json`); jw16 holds layers 24-47, jwm2 layers 0-23 | issue #108 |
| jwm2 downloads at most 11.90 GB | files 1, 2, 4 only | PASS: 3 files, 11,904,184,575 B (5,321,473,414 + 5,366,644,780 + 1,216,066,381); `rank-1.json` weights list has exactly those three | `du`/`stat` on jwm2 blobs at 21:02 CST |
| jw16 downloads all four shards | 17.18 GB | PASS: 4 files, 17,181,071,994 B, fetched in 1 min 40 s (`rank-0.json` log) | jw16 private hub blob sizes |
| jwm2 download time | | 1 min 34 s for 3 files | `rank-1.json` log |
| terminal state | CLEAN on all three | PASS: jw16, jwm2, jwm1 all `CLEAN` (no owned process, no port holder, ports closed, unit inactive, rank leases resolved) | `raw/ct230-supervisor/terminal-*.json`, `cleanup-phase.json` (no errors) |
| wall time within the budget | under 1770 s | PASS: 151 s outer unit (service runtime 2 min 30.7 s), controller unit 2 min 11.6 s; budgets `controller_runtime` 1200, `cleanup_s` 300 | `raw/ct230-supervisor/budgets.json`, `remote.log` |
| per-rank disk gate | jw16 22.18 GB (17.18 + 5.00), jwm2 16.90 GB | PASS: neither rank refused (jw16 free 28,940,001,280 B after the deletion, before launch; jwm2 21,834,788,864 B just before launch); the gate runs before the fetch, and the fetch completed | rank logs |

Two warnings in the raw logs, neither fatal: rank 1 (jwm2) prints that its GPU heap (101,202,313,216 B) is above the kernel's 75,901,734,912 B per-process limit; and rank 1 polled jw16's engine three times with "connection refused" before rank 0's engine was up.

## Cold cache
- The packet's `cold_cache()` refuses any private cache that holds more than the pinned metadata files and `refs/main`. It ran on both ranks at the start of the run and passed (the second attempt failed on it).
- jw16: my Nov 13 prefetch of the four shards (and their snapshot links) in the real HF cache was deleted at 20:49 CST with the coordinator's approval. Before: df avail 12,881,215,488 B; after: 28,940,001,280 B; shard blobs freed 17,181,071,994 B (`rm -rf` script output kept by lane dhp7; not in this receipt). A hard-link seed of the private cache was tried and rejected by `cold_cache()` at 20:47 CST.
- In the run, the four jw16 shard blobs were written between 21:00:57 and 21:01:13 CST, after the run began at 20:59:06; the metadata files were written at 20:59:32. The same holds for jwm2 (21:00:51 to 21:01:07).
- After the run, jwm2's private hub and xet directories were deleted by name at 21:03 CST (`raw/rm-jwm2-private-2.out`): `du -sb` of the private home before 11,923,201,411 B; `df -B1` free 8,294,621,184 B before and 19,940,089,856 B at 21:03:32 CST. jw16's private cache (17,200,242,504 B at 21:02 CST) is still in place.

## Earlier attempts (all failed before the run began)
1. 20:27 to 20:33 CST: jw16's disk gate refused (needs 17.18 GB + 5.00 GB; 13.37 GB free). All hosts CLEAN.
2. 20:47 CST: failed 5 s in; `cold_cache()` on jw16 refused the hard-linked seed. All hosts CLEAN.
Older failures: 15:49 (`layer_ms`), 15:58 (ssh `_rank-receipt`), 16:05 (shard set, XDG environment); each fixed in the packet before this run.

## Not checked
- The automated rows of `scripts/p7-exit-test.sh` have not been run for this receipt.
- No manager readback is claimed.
- The measured split (issue #108) was not exercised.
- Throughput was not benchmarked beyond the four requests above.

## Raw evidence
`raw/jwm1-controller/` (engine log, plan, completions with headers, serve.json, rank receipts, stop.txt, status-after.txt) and `raw/ct230-supervisor/` (budgets.json, remote.log, terminal-*.json, cleanup-phase.json), `raw/p106run-console.log`, `raw/p106-launch3.log`.
Run on jw16 (M1 Max, 62 GB), jwm2 (M2 Max, 94 GB; sizes as `omarchy-cluster status` prints them), controller on jwm1 (M1); orchestrated from CT230.
