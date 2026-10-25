# Placement: how layers are spread

How `place` and `serve` decide which node runs which layers, for the MLX engine and
the llama.cpp engine.

## MLX engine

### Automatic placement

`place MODEL` / `serve MODEL` (no `--nodes`) run the planner in
`omarchy_cluster/planner.py`:

1. Every unified-memory node is a candidate rank; only the final rank must be
   decode-eligible (`--no-decode` excludes a node from the tail).
2. The planner tries the largest stage count first, then fewer, and stops at the
   first feasible plan. Feasible means: every stage holds at least one layer in
   90 percent of its free memory (`max_layers`), all stages together hold every
   layer, and every boundary has a measured pinned route from `probe`.
3. Boundary cost is the bf16 activation (2 bytes x hidden x 2) over the pinned
   route's measured Gb/s plus its TCP RTT, printed as ms/token.
4. `serve` without an explicit `--nodes`/`--split` launches the planned stages:
   two ranks on a two-node plan. If the plan has more than two stages,
   `serve` without `--nodes` and `--split` stops with an error. For three or
   more ranks, give both options. With `--nodes N1,...` and `--split N0,N1,...`
   the ranks are exactly what you listed, any number; rank 0 samples and runs
   the last layers (`--nodes studio,m2,m1max --split 25,21,16` puts layers
   37-61 on `studio`, 16-36 on `m2`, 0-15 on `m1max`). More than two ranks has
   been run: the Muse Spark Open full-62 run used five MLX ranks on four
   machines ([receipt](../receipts/2026-10-16-muse-spark-mlx-full62)).
5. A node must hold its layers in its free memory or `serve` stops before it
   starts any rank. `--no-fit-check` skips that check (the full-62 run needed
   it: the Studio share was above the 90 percent rule).

### The split chooser

At rank start, each rank times one-token decode steps through 1 and 4 of the
model's layers in a child process and keeps the fastest ms per layer it has seen
on that node (`~/.local/state/omarchy-cluster/layer-ms.json`). Every rank runs the
same chooser in `planner.choose_split`: one layer per rank, the rest on the
cheapest ranks up to their memory caps. A stored measurement is only ever replaced
by a faster one, so a burst of other GPU work cannot steal the split. Delete
`layer-ms.json` after a real slowdown (a driver change, a config change).

### KV reservations

A layer's share is its weights plus its KV cache for the planned context.

- Plain models: every layer is charged `kv_bytes_per_token x ctx`.
- Hybrid models (Qwen3.5, Qwen3.8, Qwen3-Next): only full-attention layers grow a
  per-token cache, so the planner reserves KV for those layers only: 16 of 64
  layers on Qwen3.8-27B, 64 KiB per token instead of 256 KiB. Measured planner
  math: [receipts/2026-10-15-hybrid-kv-reservation](../receipts/2026-10-15-hybrid-kv-reservation).
- Sliding-window models (Muse Spark Open: 46 of 62 layers): the planner charges a
  window layer `min(context, window + 2047)` tokens, because mlx-lm's rotating
  cache peaks at window + one prefill chunk (2048) - 1.

### Multi-rank on one node

A node may appear more than once in `--nodes`: each of its ranks is its own
process on its own ring port (52100, 52101, ...), and all the node's layers must
fit its free memory together. Use this for a laptop with more memory than one
Vulkan process can map (about 60 GiB). The full-62 run ran two rank processes on
the 14-inch M2 Max this way.

### Ring addressing

The ring needs every rank to reach its neighbours at one address:
`--ring-ip NAME=IP` sets a node's address when its first interface is not the one
the others reach (a Wi-Fi address for a laptop with no Ethernet, for example).
The ring carries one 16 KB activation per token per hop, so a slow link costs
milliseconds, not seconds.

### Model families

The MLX engine needs a model mlx-lm can pipeline: the `deepseek_v2`/`v3`,
`glm4_moe`, `glm4_moe_lite` and `ministral3` families (mlx-lm 0.31.3's
PipelineMixin). `qwen2`/`qwen3` are not pipeline models upstream; run those on
one node (`--stages 1`).

Muse Spark Open checkpoints declare `model_type` `muse_spark`, which mlx-lm main
does not load. omarchy-mlx carries a loader in its `serve/mlx_omarchy_muse`
package (the class of mlx-lm pull request 1975, plain `mx` operations, so it runs
on Metal and Vulkan alike). Point every rank at it with
`OMARCHY_MLX_SERVE=<omarchy-mlx checkout>/serve` in the agent's environment; each
rank registers the adapter before it loads the model. Nothing changes when the
variable is unset or the package is absent.

## llama.cpp engine

`serve --engine llamacpp` (see `omarchy_cluster/llamacpp_engine.py`):

1. It starts or reaches every rpc-server first and asks each one over llama.cpp's
   RPC protocol how much memory it has (`GET_DEVICE_MEMORY`), the heap llama.cpp
   will allocate from. `=GB` after a node overrides that budget by hand.
2. A Vulkan node's agent measures how much one process can really allocate
   (`/v1/gpu-cap`, `omarchy_cluster/vkprobe.py`): 1 GiB blocks until the driver
   refuses. Cached per boot; a heap that reports 80 GiB may stop at 63 GiB
   ([receipt](../receipts/2026-10-07-measured-placement)).
3. `place` picks the node order and how many layers stay on the gateway machine so
   the fewest layers are read from disk, then uses as few nodes as it can. Whole
   layers go in order after the first `--host-layers` (or fit-chosen) host layers;
   the last count includes llama.cpp's output-layer slot, and lm_head itself is
   pinned to the host with `-ot "^output\.weight=CPU"`.
4. The split becomes `--rpc` endpoints, `--tensor-split` weights and `-ngl`.

KV sizing charges each GGUF layer its context; a sliding-window layer (GGUF
`sliding_window_pattern`) is charged `window x 4 server slots + n_ubatch (512)`
instead, because llama-server auto mode runs 4 slots on one unified KV cache and
`serve` passes no `--parallel`. On Muse Spark Open that is 3.79 GB of KV at 32768
tokens instead of 8.32 GB
([receipt](../receipts/2026-10-15-swa-kv-reservation)).

Only the gateway machine needs the GGUF file; each RPC server receives its layers
over the network at load (`--rpc-cache` keeps them on its disk for the next
load).
