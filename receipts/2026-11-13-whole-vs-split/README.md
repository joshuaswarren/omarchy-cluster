# Whole on one machine against a split over two, one model that fits one machine

2026-11-13. Issue #67 step 4 (placement: whole model on one node before any split). The run was made on
two lab laptops. The data below is the raw gateway reply of every request.

## Setup

- Model: `mlx-community/Qwen2.5-7B-Instruct-4bit`, 28 layers, 4.1 GB of weights. It was chosen because
  plain `qwen2` models pipeline in mlx-lm 0.32.2 (Qwen3 4B and the `qwen3_5` models did not, with
  "does not support pipelining"). It is a measurement subject, not a recommended model. The weights
  hashes were not recorded.
- Machines: MacBook Pro 16-inch M1 Max (62 GB) and MacBook Pro 14-inch M2 Max (94 GB), Omarchy Linux,
  Vulkan through omarchy-mlx. Wired Ethernet between them, 1.8 Gb/s measured by `probe`, 0.24 ms RTT,
  no RDMA.
- Software, same on both: omarchy-mlx wheel `0.32.4.dev202611040253+a39f0251`, mlx-lm 0.32.2 as a
  pure-Python overlay (`serve --rank-pythonpath`), system Mesa 26.2.2 ICD. The M2 Max build of the main
  Mesa tree aborted on this model (`lower_muladd` assertion), so both ran the system ICD.
  omarchy-cluster was the tree at PR #75's head `fca6d018d360044305bff9ae15f4611a55d03931`.
- `place` picked the arm itself: without `--stages` it printed `whole on` the 14-inch M2 Max (the node with
  the most free memory that holds all 28 layers). With `--stages 2` it printed `split across 2 nodes`,
  rank 0 on the 16-inch M1 Max (engine, samples, last layer), the split `[1,27]` layers chosen at rank
  start from measured ms per layer.

## Commands

Run on the 16-inch M1 Max, which holds the cluster token and runs the gateway. `NODE_M1` and `NODE_M2` are the
two nodes' names as `discover` prints them; `ADDR_M1` and `ADDR_M2` are their wired addresses. `MODEL` is
the same local directory on both machines. The plan lines `serve` printed are in
[place-whole.txt](data/place-whole.txt) and [place-split.txt](data/place-split.txt) (interface names and
addresses removed).

```
probe:   omarchy-cluster probe --out links.json
whole:   omarchy-cluster serve MODEL --links links.json --nodes NODE_M1,NODE_M2 \
           --rank-pythonpath mlx-lm-0.32.2 --gpu-turn 30
split:   omarchy-cluster serve MODEL --links links.json --nodes NODE_M1,NODE_M2 --stages 2 \
           --ring-ip NODE_M1=ADDR_M1 --ring-ip NODE_M2=ADDR_M2 --rank-pythonpath mlx-lm-0.32.2 --gpu-turn 30
request: curl -s http://127.0.0.1:8020/v1/completions -H 'Content-Type: application/json' \
           -d '{"prompt":"Write a short paragraph about why the sky is blue.","max_tokens":64,"temperature":0,"timing":true}'
stop:    omarchy-cluster stop
```

Three requests per arm after the load: 40 prompt tokens, `max_tokens` 64, temperature 0, `timing` true.

Order of the run. Whole arm: `serve` at 21:54:24Z, gateway ready 22:00:58Z, three requests 22:00:59Z to
22:01:02Z, `stop`. Split arm: waited until neither machine's GPU queue had a holder, `serve` at 22:08:37Z,
ready 22:10:30Z, three requests 22:10:32Z to 22:10:36Z, `stop` at 22:10:37Z. Logs:
[whole-log.txt](data/whole-log.txt), [split-log.txt](data/split-log.txt). Each request's raw reply is
`data/whole.N.json` and `data/split.N.json`.

## Result

| arm | request | decode tok/s | prefill ms | decode ms | ms per token step |
|---|---|---|---|---|---|
| whole, 14-inch M2 Max | 1 | 60.48 | 322.7 | 1071.5 | 16.80 |
| whole, 14-inch M2 Max | 2 | 60.28 | 202.4 | 1075.3 | 16.85 |
| whole, 14-inch M2 Max | 3 | 60.34 | 213.1 | 1075.1 | 16.84 |
| split, M1 Max + M2 Max | 1 | 36.99 | 268.1 | 1734.2 | 27.46 |
| split, M1 Max + M2 Max | 2 | 37.65 | 224.6 | 1702.2 | 26.98 |
| split, M1 Max + M2 Max | 3 | 38.34 | 267.6 | 1672.7 | 26.49 |

Median decode: whole 60.34 tok/s, split 37.65 tok/s, so whole is 1.60 times faster. All six replies
carry the same text (SHA-256 prefix `f83f6fd2b3af3c97`), so the speeds compare like for like. The first split
request spent 21.18 ms of its 27.46 ms step in the ring receive (`step_ms` in
[split.1.json](data/split.1.json)).

Load time. Whole: `serve` returned at 21:54:32Z and the gateway was ready at 22:00:58Z, 6 min 26 s, of
which the wait for the node's turn on the shared GPU queue is not separated out. Split: 22:08:45Z to
22:10:30Z, 105 s, started when both GPU queues were idle. The two load times are not comparable.

## Limits

- The arms differ in hardware: whole ran on the M2 Max alone, the split put the last layer on the M1
  Max. A whole run on the M1 Max was not made, so whole on the slower node is unmeasured.
- Three requests of 64 tokens on one prompt, short context. Variation was under 2 percent whole and
  under 4 percent split.
- The result says nothing about a model too large for one machine.
- A second split attempt (22:01Z) was stopped before any request; its data is not here.

## Headroom

`plan_placement` lets a node hold the model whole when weights plus the KV cache for the planned context
fit in 90 percent of the node's free memory (`planner.max_layers`). The margin is 10 percent of free
memory, pinned by `test_a_node_keeps_ten_percent_of_its_free_memory_before_it_holds_the_model_whole`.
This receipt does not measure it: the 4.1 GB model sits far below the limit on a 94 GB node.

Files: [data](data), checksums in [data/SHA256SUMS](data/SHA256SUMS).
