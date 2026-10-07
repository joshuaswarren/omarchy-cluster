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
   0.5 GB on a CPU. On macOS, available RAM is free pages only (inactive pages come back only by
   swapping), and zero when the Data volume has under 20 GB free. Two servers on one machine
   share its RAM; the CPU server gives way first.
4. The host keeps layers it has room for in its page cache: available RAM minus 2 GB, the model's
   non-layer tensors and 1 GB for llama-server.
5. Placement searches the node order and the host layer count. It minimizes the host bytes beyond
   that page cache first, then the devices used, then the host layers. `--host-layers N` keeps
   N and the typed order, as before.
6. rpc-servers no longer write their share to local disk unless `--rpc-cache` is given.

## Dry split: GLM-5.3 UD-IQ1_S on the 2026-10-07 nodes

[src/dry_split.py](src/dry_split.py) runs the code above on inputs recorded that night. Two are
assumed, not recorded: the x86 laptop's CUDA free memory (3.9 GB of a 4 GB card) and the Mac
Studio's Metal working set (103 GB). Output: [dry-split.txt](dry-split.txt).

On macOS the budget counts free pages only. Run P sized the Mac Studio from free + inactive +
speculative pages (60.6 GB) and gave it 41.2 GB; macOS took that by swapping (17.1 to 25.9 GB) and
the run was stopped before it finished loading. With free pages (8.3 GB, sampled that night) the
Mac Studio's budget is 1.3 GB, smaller than one layer, so it gets none:

| | paged from host disk | devices used | host layers |
|---|---|---|---|
| hand-built (run P), Mac Studio 41.2 GB | 0 | 6 | 0-4, 6.67 GB |
| measured, M1 Max scratch cleared (60.3 GB available) | 39.98 GB | 5 | 0-19, 48.81 GB |
| measured, M1 Max as it was at 02:27Z (51.5 GB available) | 47.99 GB | 5 | 0-22, 56.82 GB |

So the hand-built table fit only by counting memory the Mac Studio did not have free. Without that,
GLM-5.3 at UD-IQ1_S does not fit these machines without paging about 40 GB from the host's disk,
and the measured placement says so before any load starts.

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
