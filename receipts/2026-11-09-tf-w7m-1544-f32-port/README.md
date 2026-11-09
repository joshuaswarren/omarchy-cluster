# w7m/1544-f32-port — the Qwen3.5 Vulkan f32 layers 0-3 port, end to end

Date: 2026-11-09. Lane: w7m. Source fix: w7p `w7p/1544-gdn-f32` (tip `3e3c9d21a`, issue #1544).
Delivered branch: `w7m/1544-f32-port`, pushed to https://github.com/joshuaswarren/TensorFold (fork).
Worktree: `~/scratch/w7m-tf-1544`, tip `7db9b04f5b56ba75a9fcea05c4fefd52d0f40b8b`, clean.

## What was ported

The eight-commit f32 series from the merge-base `4875da90e` (cherry-picked in order, conflicts
resolved at picks 1, 2, 3 and 8):

| w7p commit | Content |
|---|---|
| 2fb2e52df | shaders: f32 activation variants (ACT_F32 / ACT_IN_F32 / ACT_OUT_F32 / ADD_WIDE_F32 macro families) for the wide layers 0-3 stores |
| 348857b08 | vulkan ops: qmm_f32 / add_bf16_f32 pipelines, f32-aware ops.zig plumbing |
| ba4791ca0 | qwen35 engine: per-layer activation kind, layers 0-3 f32 (engine.zig, native/vulkan.zig) |
| abb25ec07 | backend: lanes max_rows step bound |
| 3e907c582 | tests: stage2_fixture_f32, the wide-layer sweep |
| 8b9735877 | style: one-line comments at the layer-kind call sites |
| d94bc5459 | style: lean check at base parity |
| 3e3c9d21a | the end-to-end fix: norm_f32 gains ACT_OUT_F32, conv_dw1d_f32 becomes -DUSE_BF16 -DACT_F32 |

## Conflict resolution (the actual port work)

The w7m line (`w7m/1990-oracle-rename`) carries two things the w7p line lacks: the
qmm_q4_vec4 projection (alignment-forced 16-byte-load q4) and the ash→oracle rename
(`ASH_QWEN35`→`ORACLE_QWEN35`, `ash_gate`→`oracle_gate`,
`gated_delta_prefill_ash.comp`→`gated_delta_prefill_oracle.comp`, corrected copyright headers).
Every conflict was resolved keeping both concerns:

- `zig/build/vulkan.zig`: oracle defines on all bf16 entries + all f32 entries renamed to
  `-DORACLE_QWEN35`; the final fix's two semantic deltas applied (`norm_f32` → ACT_OUT_F32,
  `conv_dw1d_f32` → `-DUSE_BF16 -DACT_F32`); q4_vec4 build/pipeline wiring kept.
- `zig/src/vulkan/ops_q4.zig`: merged selection —
  `qmmPipeline(o, force, act_f32, x, w)`: an f32 output kind returns `qmm_f32`; otherwise
  w7m's scalar/vec4 alignment logic (force or fit, Misaligned refusal) returns
  `qmm_q4_vec4` or `qmm_bf16`. Both the rows==0 and chunked paths route through it; the
  scatter picks copy_u32/copy_u16 by output kind.
- `ops.zig`, `native/vulkan.zig`, five probe/test files: union of `qmm_q4_vec4` + `qmm_f32`
  fields/pipelines (no duplicate `copy_f32_bf16`).
- `gdn_split.comp` resolved to the w7p post-fix shader with `oracle_gate` naming.
- One stale rename leftover on the w7m base fixed: `attention.zig` header comment now names
  `-DORACLE_QWEN35`; `gdn.zig` doc comment now names `gated_delta_prefill_oracle.comp`.

## Verification that the port is faithful

`git diff 3e3c9d21a HEAD -- zig/`: 36 files, +243/−59, and every hunk is one of (a) ASH→ORACLE
naming, (b) copyright header text, (c) qmm_q4_vec4 machinery and its contract fixtures.
No other drift.

## Gates run on omp-studio-local (Debian 12 PVE container, EPYC 7443P, zig 0.17.0, glslang 15)

- `zig fmt --check zig/` — clean.
- Host steps (all exit 0): `vulkan-host-test`, `stage1..stage5-host-test`, `model-host-test`,
  `contract-pointwise-host-test`. These compile every ported f32 SPIR-V variant and run the
  stage-2 f32 fixture / wide-layer sweep host-side. Logs: full build summaries captured.
- Device *build* steps (all exit 0): `vulkan-gpu-build`, `stage1..stage5-gpu-build`,
  `contract-pointwise-build`, `tf-vulkan-kernel-contract` — every device artifact compiles.
- `vulkan-gpu-test` exit 1 **by design**: this host has only llvmpipe (software device), and
  device selection refuses software drivers (`llvmpipe: SoftwareDevice`), as does the gdn
  contract ("a CPU (software) Vulkan device is never a GPU result").
- One transient: the first cold-build combined host run exited 1 in the stage4_fixture binary
  under `--listen` coordination; the same binary with the same seed passes standalone, and
  three repeat combined runs plus the full summary battery passed. Not reproducible; not
  port-attributable.
- Note: the eight picked commits were gated at the branch tip, not individually built.

## Omarchy M+ track touches

- omarchy-mlx / Mesa Honeykrisp: nothing changed; the port only consumes the Vulkan device at
  runtime. Device receipts for this branch will come from the w7m M1 lane where the ICD is the
  Honeykrisp stack built from joshuaswarren/mesa-1 (per the mesa-repo-migration rule; no Mesa
  work happened here and none references joshuaswarren/mesa).
- omarchy-cluster: this receipt. ANE: none. macOS parity: none (the Metal path is untouched).

## Next step (lane owner)

Run the device contract battery on the w7m 13-inch M1 (`stage4-gpu-run`, gdn contract,
pointwise/attention/greedy/qmm contracts) and record receipts; compare the f32 layers 0-3
cases against the w7p device findings (argmax V==A 64/64, pooled upper95 bound). The private
notebook entry for this run: `~/.local/share/apple-silicon-lab/entries/w7m/20261109T0400Z-omp-studio-local-tf-1544-f32-port.md`
(transcripts under `artifacts/w7m/20261109-run1/`).
