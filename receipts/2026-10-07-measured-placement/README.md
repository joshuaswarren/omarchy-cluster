# Measured budgets and placement for `serve --engine llamacpp`

2026-10-07. Until now each `--rpc-node NAME=GB` was a number someone typed, and the node order
was the order typed. Tonight both went wrong in ways the hand numbers hid: an M2 Max heap that
reported 80 GiB stopped allocating at 63 GiB, and the 753B run kept 24.8 GB of layers on a 16 GB
host, which may have been read from disk on every token.

## What serve does now

1. It starts or reaches every rpc-server first and asks each one, over llama.cpp's own RPC
   protocol, how much memory it has (`GET_DEVICE_MEMORY`). That is the heap llama.cpp will
   allocate from: a Vulkan heap, Metal's working set, or system RAM for a CPU server.
2. A Vulkan node's agent measures how much one process can allocate (`/v1/gpu-cap`,
   `omarchy_cluster/vkprobe.py`): 1 GiB blocks until the driver refuses, with the heap set to the
   machine's RAM so the address space is what stops it. Cached per boot.
3. Without `=GB`, a device's budget is the least of: what llama.cpp reports free, the machine's
   available RAM minus 6 GB, and the allocation cap; less 1 GB for compute buffers on a GPU or
   0.5 GB on a CPU. On macOS, available RAM is free, purgeable and file-backed pages (they come
   back without swapping) less 8 GB for the Mac's own services; anonymous inactive pages do not
   count, they come back only by swapping. Zero when the Data volume has under 20 GB free. Two
   servers on one machine share its RAM; the CPU server gives way first.
4. The host keeps layers it has room for in its page cache: available RAM minus 2 GB, the model's
   non-layer tensors and 1 GB for llama-server.
5. Placement searches the node order and the host layer count. It minimizes the host bytes beyond
   that page cache first, then the devices used, then the host layers. `--host-layers N` keeps
   N and the typed order, as before.
6. rpc-servers no longer write their share to local disk unless `--rpc-cache` is given.

## Dry split: GLM-5.3 UD-IQ1_S on the 2026-10-07 nodes

[src/dry_split.py](src/dry_split.py) runs the code above on inputs recorded that night. Two are
assumed, not recorded: the x86 laptop's CUDA free memory (3.9 GB of a 4 GB card) and the Mac
Studio's Metal working set (103 GB; its RAM binds first). The Mac Studio's RAM comes from the 07:37Z
sample, taken after run P, which gives the 32.86 GB budget validated below. Output:
[dry-split.txt](dry-split.txt).

| | paged from host disk | devices used | host layers |
|---|---|---|---|
| hand-built (run P), Mac Studio 41.2 GB | 0 | 6 | 0-4, 6.67 GB |
| measured, M1 Max scratch cleared (60.3 GB available) | 6.65 GB | 6 | 0-7, 15.48 GB |
| measured, M1 Max as it was at 02:27Z (51.5 GB available) | 13.29 GB | 6 | 0-9, 22.12 GB |

Run P's table paged nothing because it gave the Mac Studio 41.2 GB, sized from free + inactive +
speculative pages (60.6 GB); macOS took that partly by swapping (17.1 to 25.9 GB). At a Mac budget
that holds without swapping, GLM-5.3 at UD-IQ1_S pages 6.65 GB from the host's disk at best on these
machines, and the measured placement says so before any load starts.

A first fix that counted free pages only left the Mac Studio 1.3 GB, smaller than one layer, and
paged about 40 GB from the host. That rule threw away a node that held 40 GB earlier the same night;
the budget now counts free, purgeable and file-backed pages instead.

The measured budgets also run the M2 Max GPU at 66.47 of 66.65 GB, 1.2 GB below its 63 GiB cap.
The 1 GB GPU reserve is an estimate; the only measured bound that night is 1.3 GB spare on the
4 GB card.

## Validating the macOS budget

07:37Z, Mac Studio, oMLX running: the budget computed live from the rule above was 32.86 GB (free
31.79 + purgeable 0.11 + file-backed 15.95, less 8 + 6 + 1). One Metal rpc-server allocated
32.86 GB in one buffer, every page written ([src/macfill.py](src/macfill.py)), held 60 s, then
freed it.

| | before | during (min) | after |
|---|---|---|---|
| swap used | 25600 MB | 25592 MB | 25584 MB |
| memory_pressure free | 63% | 38% | 65% |
| Data volume free | 59 GiB | 59 GiB | 59 GiB |

Swap did not grow (-16 MB) with the whole budget resident. The pass mark was under 2 GB.

The measured budgets also run the M2 Max GPU at 66.47 of 66.65 GB, 1.2 GB below its 63 GiB cap.
The 1 GB GPU reserve is an estimate; the only measured bound that night is 1.3 GB spare on the
4 GB card.

## Tests

`tests/test_placement.py`: the tightest-limit budget, RAM shared by two servers on one machine,
placement that prefers no paging and then fewer devices, a fixed order and host count, and the
hand-built GLM-5.3 run-P table reproduced exactly from its budgets. `tests/test_llamacpp.py`:
the device-memory query over RPC, and the cache flag; macOS free pages, the agent's probe
limit. 83 tests pass.

## The probe on hardware

05:26Z, MacBook Pro 16-inch M1 Max (62.8 GiB visible to Linux), heap set to its RAM: 60 GiB
(64.4 GB) allocated, then the driver refused with `DRM_IOCTL_ASAHI_VM_BIND failed`, not the
address-space error the M2 Max hit at 63 GiB. Page cache fell from 18 to 1 GiB during the probe:
blocks past free memory take real memory. The agent now caps the probe at available RAM minus
the 6 GB headroom and reports `stopped_at_limit` when the driver never refused. The budget
already takes the least of RAM and the cap, so the limit costs nothing.

For run P this settles the M1 Max share: 55.79 GB under a measured 64.4 GB.

Not yet run on hardware: `serve` with measured budgets end to end.
