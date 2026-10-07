# GLM-5.3-Flash (UD-IQ4_XS, 156.8 GB) on four Apple Silicon machines

2026-10-07. One OpenAI-compatible endpoint serving a model no single machine here can hold, split
over a Mac Studio on macOS and three Macs running Omarchy (Arch Linux ARM), through llama.cpp RPC.

## Setup

| node | hardware | OS | backend | layers | GB on node |
|---|---|---|---|---|---|
| mac-ultra | Mac Studio M1 Ultra, 128 GB | macOS 26.6.2 | Metal | 0-13 | 41.7 |
| omarchy-m2 | MacBook Pro 14-inch M2 Max (2023), 94.9 GiB visible to Linux | Omarchy on Arch Linux ARM | Vulkan (Honeykrisp, Mesa 26.2.3) | 14-30 | 60.1 |
| omarchy-m1 | MacBook Pro 16-inch M1 Max (2021), 62.8 GiB visible to Linux | Omarchy | Vulkan (Honeykrisp, Mesa 26.2.2) | 31-42 | 42.4 |
| omarchy-host | MacBook Pro 13-inch M1 (2020), 15.3 GiB visible to Linux | Omarchy on Arch Linux ARM | CPU, plus the llama-server host | 43-45 | 7.8 |

- Model: unsloth GLM-5.3-Flash UD-IQ4_XS, 5 shards, 156.81 GB, 46 layers (layer 45 is the MTP
  layer, which llama.cpp does not load).
- llama.cpp `65840ed` on every node. omarchy-cluster `54ef8b1`.
- Network: wired LAN. The host's 2.5GbE USB adapter measured 2.35 Gb/s to every node; RTT 0.2 to
  0.5 ms.
- Command: [src/headline.sh](src/headline.sh), with `MACGB=42 M2GB=62 M1GB=45 HOSTGB=8`.

## Results

| | request 1 | request 2 |
|---|---|---|
| decode | 1.39 tok/s | 1.57 tok/s |
| prompt | 13.6 s | 2.7 s (prompt cache) |
| completion | 64 tokens | 64 tokens |
| text sha256 | `3ad4239e76aabcf2` | `3ad4239e76aabcf2` |

- Same prompt, temperature 0, both times: identical text.
- Load: 1055 s. The host pushes about 149 GB over its one 2.35 Gb/s link; that alone takes about
  507 s.
- Recording: [data/glm53-flash-4node.cast](data/glm53-flash-4node.cast) (`asciinema play`). Engine
  log: [data/llamacpp-engine.log](data/llamacpp-engine.log).

## What it took

Four earlier attempts did not finish. What they turned up, each fixed in omarchy-cluster or worked
around in this run:

1. The M2 could not allocate its share. On this M2, Honeykrisp's user GPU address space stops at
   63 GiB (67.6 GB) while Vulkan reports an 82015 MiB heap. Measured with
   [src/vkalloc.c](src/vkalloc.c): 1 GiB blocks until `Failed to allocate BO VMA`. The M2 budget
   here is 62 GB.
2. omarchy-cluster passed GB budgets to llama.cpp as `--tensor-split` weights. llama.cpp gives each
   device one contiguous block by layer count, and layers here run from 0.3 to 4.7 GB, so a node
   could get more bytes than its budget. The split now fills nodes in order with each layer's real
   size and passes whole layer counts (`01a64cf`). Skipped MTP layers are not charged (`a71165e`).
3. `rpc-server -c` caches every large tensor it receives on local disk. A 42 GB share would have
   filled a nearly full system disk, so that attempt was stopped before any weights moved. The run
   used wrappers that drop `-c` ([src/rpc-vulkan](src/rpc-vulkan)). On macOS the launch agent could
   not run the wrapper from an external volume ("Operation not permitted"); it runs from the home
   directory.
4. `serve` returned before the gateway listened, so the first request got connection refused after
   an 18 minute load. `serve` now waits for the gateway (`54ef8b1`).
