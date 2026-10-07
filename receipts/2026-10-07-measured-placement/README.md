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
   0.5 GB on a CPU. Two servers on one machine share its RAM; the CPU server gives way first.
4. The host keeps layers it has room for in its page cache: available RAM minus 2 GB, the model's
   non-layer tensors and 1 GB for llama-server.
5. Placement searches the node order and the host layer count. It minimizes the host bytes beyond
   that page cache first, then the devices used, then the host layers. `--host-layers N` keeps
   N and the typed order, as before.
6. rpc-servers no longer write their share to local disk unless `--rpc-cache` is given.

## Dry split: GLM-5.3 UD-IQ1_S on the 2026-10-07 nodes

[src/dry_split.py](src/dry_split.py) runs the code above on inputs recorded that night. Two are
assumed, not recorded: the x86 laptop's CUDA free memory (3.9 GB of a 4 GB card) and the Mac
Studio's Metal working set (103 GB; its available RAM binds first either way). Output:
[dry-split.txt](dry-split.txt).

| | paged from host disk | devices used | host layers |
|---|---|---|---|
| hand-built (run P) | 0 | 6 | 0-4, 6.67 GB |
| measured, M1 Max scratch cleared (60.3 GB available) | 0 | 5 | 0-4, 6.67 GB |
| measured, M1 Max as it was at 02:27Z (51.5 GB available) | 3.58 GB | 6 | 0-6, 12.41 GB |

With the same inputs as run P, the measured placement pages nothing and uses one device fewer: it
leaves the 4 GB CUDA card out. Without the scratch cleanup it pages 3.58 GB from the host's disk
and says so.

It also runs two devices closer to their limits than the hand table: the M2 Max GPU at 66.47 GB
under its 66.65 GB budget (1.2 GB below the 63 GiB cap, where the hand table left 5.9 GB), and the
Mac Studio at 42.82 of 42.90 GB. The 1 GB GPU reserve is an estimate; the only measured bound
tonight is 1.3 GB spare on the 4 GB card. The receipt for the first run that uses these budgets
will show whether it holds.

## Tests

`tests/test_placement.py`: the tightest-limit budget, RAM shared by two servers on one machine,
placement that prefers no paging and then fewer devices, a fixed order and host count, and the
hand-built GLM-5.3 run-P table reproduced exactly from its budgets. `tests/test_llamacpp.py`:
the device-memory query over RPC, and the cache flag. 80 tests pass.

## The probe on hardware

05:26Z, MacBook Pro 16-inch M1 Max (62.8 GiB visible to Linux), heap set to its RAM: 60 GiB
(64.4 GB) allocated, then the driver refused with `DRM_IOCTL_ASAHI_VM_BIND failed`, not the
address-space error the M2 Max hit at 63 GiB. Page cache fell from 18 to 1 GiB during the probe:
blocks past free memory take real memory. The agent now caps the probe at available RAM minus
the 6 GB headroom and reports `stopped_at_limit` when the driver never refused. The budget
already takes the least of RAM and the cap, so the limit costs nothing.

For run P this settles the M1 Max share: 55.79 GB under a measured 64.4 GB.

Not yet run on hardware: `serve` with measured budgets end to end.
