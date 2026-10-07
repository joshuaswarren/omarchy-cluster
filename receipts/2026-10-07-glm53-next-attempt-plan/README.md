# Pre-registration: the next GLM-5.3 UD-IQ1_S run (written 2026-10-07, before the run)

The run waits for the x86 laptop on wired Ethernet and a window on all nodes. Everything below is
fixed before it starts; the receipt for the run will report against it.

## What changed since the first run

The first full run (receipt [2026-10-07-glm53-full-7device](../2026-10-07-glm53-full-7device)) is the
reference: 0.39 tok/s warm median, 24.8 GB of host layers on a 16 GB host. The attempt that followed
([2026-10-07-measured-placement](../2026-10-07-measured-placement)) was stopped before it finished
loading. Since then:

- placement comes from measured budgets and pages the fewest host bytes, then uses the fewest nodes;
- the macOS budget counts free, purgeable and file-backed pages, validated on the Mac Studio;
- llama-server uses the performance cores only;
- rpc-servers can keep received weights on their own disk (`--rpc-cache`), which makes a second load
  of the same model fast enough to test one change in the same session.

## Preconditions

- The x86 laptop on wired Ethernet (Wi-Fi made most of the first run's 2724 s load).
- The host on omarchy-cluster main at or after this commit's parent (`install-agent`).
- The M1 Max's scratch space cleared (its budget assumes 60.3 GB available).
- rpc-servers with `-c` (`--rpc-cache`): the Mac Studio's cache on the external volume
  (`--rpc-env mac-ultra=LLAMA_CACHE=...`), never the system disk; the M2 Max CPU server started by hand
  with `-c -t 8` (its 8 performance cores); the x86 laptop's servers with `-c`.

## Session (one window, about 70 minutes)

| arm | change | measured |
|---|---|---|
| A | new baseline: measured placement, macOS budget, performance-core threads, cache on, x86 wired | cold load time, the host paging check, 1 cold + 5 warm requests |
| B | A plus `--rpc-env mac-ultra=GGML_METAL_SHARED_BUFFERS_DISABLE=1`, reloaded from the caches | warm reload time, 1 cold + 5 warm requests |

Arm A changes several things at once against the first run and is reported as a new baseline, not as
the effect of any one change. Arm B against arm A is a single-variable test of the Metal setting, in the
same session, on the same loaded weights.

Launch (arm A), started detached on the host so a dropped ssh session cannot stop it, recorded by
following its log:

```sh
omarchy-cluster serve ~/models/GLM-5.3-UD-IQ1_S/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf --engine llamacpp \
  --llama-server ~/src/llama.cpp/build-cpu/bin/llama-server --rpc-cache \
  --rpc-node x86-laptop:50052 --rpc-node x86-laptop:50053 --rpc-node omarchy-m1 --rpc-node mac-ultra \
  --rpc-node omarchy-m2 --rpc-node omarchy-m2:50061 \
  --rpc-binary omarchy-m1=~/src/llama.cpp/build-vulkan/bin/ggml-rpc-server \
  --rpc-binary omarchy-m2=~/src/llama.cpp/build-vulkan/bin/ggml-rpc-server \
  --rpc-env omarchy-m1=HK_SYSMEM=60000000000 --rpc-env omarchy-m2=HK_SYSMEM=86000000000 \
  --rpc-env mac-ultra=LLAMA_CACHE=/Volumes/ext/rpc-cache --ctx 2048
```

No `=GB` and no `--host-layers`: the budgets are measured and the placement is chosen. No wrapper
scripts: the heap sizes and the cache folder go to each node's RPC server with `--rpc-env` (amended
before the run, when that flag landed; nothing measured changed).

## Expected (before the run)

- Host layers beyond the host's page cache: about 6.65 GB (dry split at the validated Mac Studio
  budget), against about 16 GB in the first run.
- Load: about 17 minutes cold with the laptop wired (estimate; hashing the weights on the host takes
  about 275 s of it); about 6 to 7 minutes warm from the caches.
- Decode: 0.4 to 0.6 tok/s if host paging accounts for the time the way the first run's breakdown
  suggests. That is an inference; the run measures it.

## Method and guards

- Requests straight to llama-server: temperature 0, `cache_prompt` false, 64 tokens, top-2 logprobs;
  answers hashed as reasoning text plus content, and an empty answer stops the run.
- Host paging check before the first timed request: major faults and disk reads during one request.
- Stop rules: Mac Studio memory pressure below 15 percent free or its Data volume below 20 GB; any
  Linux node below 2 GB available and falling; any node the memory watcher cannot read, twice in a row.
  The watcher runs on a machine that is not part of the run. Using 90 percent or more of each node's
  RAM is the target, not a limit.
