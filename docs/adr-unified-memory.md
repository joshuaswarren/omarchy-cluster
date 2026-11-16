# ADR: the cluster shards layers across nodes; it does not pool unified memory

Status: decided (issue 65). Date: 2026-11-16.

## Context

Every node in this cluster is an Apple-silicon or other machine whose memory is
unified *within the machine*: CPU and GPU see one address space on that node.
Across nodes there is no shared addressable memory - ranks exchange data only
through the network path the pipeline already uses (rank 0 publishes jobs,
`all_gather` returns every rank's token). A reader could hear "unified memory"
and expect the cluster to allocate one tensor across several machines. It
cannot, and nothing in the code tries.

## Decision

The supported model is **distributed layer sharding**. Each rank owns a block
of layers; its weights and its KV cache live in that rank's own memory and
never cross nodes. "Unified memory" in these docs always means one node's own
UMA, never a pooled cluster-wide space. `place` and `serve` choose layer
blocks per node (docs/placement.md); no code path claims or attempts pooled
allocation.

## Evidence

Measured across four machines (receipt `receipts/2026-10-16-muse-spark-mlx-full62`,
Muse Spark Open 2-bit, 62 of 62 layers, 5 ranks): the model is larger than any
single node, and each rank's peak stays inside its own node's limit -

| node | rank layers | measured peak | node limit |
|---|---|---|---|
| ST | 38-61 | 92.372 GB | 130.57 GB |
| M2 | 26-37, 14-25 | 45.514 GB | 57.98 GB |
| JW | 2-13 | 45.514 GB | 53.1 GB |
| M1 | 0-1 | 8.567 GB | 9.9 GB |

Peaks equal per-rank weight+activation footprint measured at load
(`sharded_load done` lines); per-node limits are the planner's measured heaps.
Nothing in the run shared an allocation across machines.

## Consequences

- Capacity grows node by node: to serve a bigger model, add a rank with its own
  memory, not a bigger pool.
- Per-node memory rules (placement headroom, #67) apply per rank, per node.
- If pooled memory ever becomes possible (e.g. TB5 memory fabric), that is a
  new decision with its own receipt; this ADR stands until then.
