# GB10 Spark: CUDA against Vulkan, matched pair, Qwen3.8-27B Q4_0

2026-11-09 (w7G lane). Answers the backend-choice question in
omarchy-cluster#43 and gives issue #41 its CUDA-tier data point: same card, same
llama.cpp commit, same model, both backends, on the fleet's NVIDIA node.

## Setup

- Machine: a DGX Spark (NVIDIA GB10, ARM64, 121 GiB unified memory, driver
  580.178.04, CUDA 13.0.88). nvidia-smi reports no per-card memory on this part;
  `free -g` is the memory record.
- Software: llama.cpp at `c7fc509`, two build trees from the same dirty source
  tree (`build-w7g-cuda` with GGML_CUDA=ON, `build-vulkan` with GGML_VULKAN=ON,
  CUDA off and Vulkan off respectively). At bench time the working tree carried
  four uncommitted files from other lanes (`src/llama-memory-hybrid-idx.cpp`,
  `src/models/qwen4exp.cpp`, `tools/server/server-context.cpp`,
  `tools/server/server-queue.cpp`); both backends compiled the same source, so
  the pair stays matched.
- Model: unsloth `Qwen3.8-27B-Q4_0.gguf`, 16,056,478,688 bytes (re-measured here
  with `stat`; llama-bench's "14.94 GiB" size column is the loaded model
  footprint, not the file), sha256
  `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`, verified by
  the bench runner before every window. The same file is on record in the eGPU
  corpus, where it is identified by the same bytes and sha as the Hugging Face
  LFS object: [B60 M1 Max record](https://github.com/joshuaswarren/omarchy-egpu/blob/main/benchmarks/2026-10-23-b60-m1max-llamacpp-vulkan.md).
  The RTX 5060 Ti record used this model and quant:
  [5060 Ti record](https://github.com/joshuaswarren/omarchy-egpu/blob/main/benchmarks/2026-10-23-5060ti-m1max-nvidia.md).
- Run: `llama-bench -m MODEL -p 512 -n 128 -fa 1 -t 8`, greedy, per backend,
  CUDA first then Vulkan back to back inside each window. Repetitions per window:
  `bench.out` n=3, `cuda.out`/`vulkan27.out` n=5, `cuda9.out`/`vulkan9.out` n=9.
- Vulkan path: NVIDIA GB10, UMA, matrix cores KHR_coopmat (glslc on this stack
  exposes GL_KHR_cooperative_matrix, not NV_cooperative_matrix2, so the shader
  path is the KHR one — the same path class as the third-party 3090 numbers).
- Host conditions, recorded for honesty: one co-tenant server
  (`build-short-preempt-c7fc509`, 86,761 MiB held through nvidia-smi, mostly
  idle) was resident through all runs. Numbers are contended-host numbers; the
  CUDA-Vulkan comparison inside a window is matched because both sides ran under
  the same conditions. Raw outputs: [data](data).

## Results

Two matched windows plus one CUDA-only first run, `llama-bench -p 512 -n 128
-fa 1 -t 8` (mean ± std as llama-bench prints it):

| window (UTC) | backend | n | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|---|
| 13:44 | CUDA only | 3 | 548.61 ± 199.70 | 6.60 ± 0.96 |
| 13:47 | CUDA | 5 | 458.58 ± 19.88 | 5.38 ± 0.54 |
| 13:47 | Vulkan | 5 | 347.20 ± 50.04 | 7.34 ± 1.68 |
| 14:05 | CUDA | 9 | 623.54 ± 227.82 | 10.33 ± 1.60 |
| 14:05 | Vulkan | 9 | 287.46 ± 21.25 | 4.95 ± 0.48 |

Vulkan attempt history: the 13:44 Vulkan bench never ran — its binary had not
finished linking (`bench.out` records "No such file or directory"). The next
attempt (`vulkan.out`) failed with `vk::PhysicalDevice::createDevice:
ErrorInitializationFailed` while the CUDA bench of that window held the model in
memory; a 1.7B model loaded fine on the same build at the same time
([vk17b.out](data/vk17b.out)), so the failure was memory pressure, not a device
bug. The third attempt passed (`vulkan27.out`) and every later window ran.

Prompt processing: CUDA is faster in both matched windows, by 1.32x and 2.17x.
The direction is stable across all three windows; the size is not, because the
co-tenant's memory traffic moves both sides between windows.

Generation: the ranking changed sign between the two matched windows. At 13:47
Vulkan was nominally ahead (7.34 ± 1.68 against 5.38 ± 0.54, bands touching —
not significant). At 14:05 CUDA was clearly ahead (10.33 ± 1.60 against
4.95 ± 0.48, disjoint bands). A contended unified-memory host does not give a
stable tg128 ranking, and this record does not claim one; the quiet-window rerun
below pins it.

## Correctness probe

Both backends answered the raw `/completion` probe ("The capital of France is",
temperature 0, 8 tokens, seed 0) with the identical greedy text
`' Paris.\nThe capital of Germany is'` and 8 predicted tokens:
[probe2.out](data/probe2.out).

## Decision

CUDA is the backend for an NVIDIA rank in a split. Prompt processing decides it
on this card: CUDA led every window. For generation, the one window with
disjoint bands (14:05, the tighter of the two) has CUDA ahead; the 13:47 window
was a non-significant Vulkan edge. Nothing measured here argues against CUDA,
and the enclosure cards on record agree on both workloads: RTX 5060 Ti 16 GB
1.40x prompt and 1.09x generation
([record](https://github.com/joshuaswarren/omarchy-egpu/blob/main/benchmarks/2026-10-23-5060ti-m1max-nvidia.md));
for Qwen3.8-27B the third-party 3090 pair reads 1.242x prompt and 1.115x
generation and the 3090 Ti 1.290x and 1.133x
([comment](https://github.com/joshuaswarren/omarchy-cluster/issues/43#issuecomment-6723474060),
audited from the retained archive).

Named follow-up: one quiet-window pair (the resident 86 GiB server drained) to
pin the GB10 decode ranking before the planner hard-codes per-card tg128 values.
Until then the planner takes CUDA as the NVIDIA default with pp-weighted sizing.

## Co-tenant

One resident server held most of the memory through every run:
`build-short-preempt-c7fc509` (Remnic core model, 262144 context, 86,761 MiB
held per nvidia-smi, mostly idle). Absolute numbers are contended-host numbers.
Both backends ran under the same conditions inside each window, which is what
keeps the pair matched.
