"""Measured-budget placement for GLM-5.3 UD-IQ1_S on the 2026-10-07 nodes, against the
hand-built run-P table. Runs omarchy-cluster's own _plan_devices on inputs recorded that
night (marked "rec") or stated where not recorded ("assumed"). No host is touched.

    python3 receipts/2026-10-07-measured-placement/src/dry_split.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests"))

from omarchy_cluster import cli, llamacpp_engine as lce
from test_placement import glm53_iq1s_info

GB, GIB = 10 ** 9, 2 ** 30


def devices(m1_ram_avail):
    # free/total: what the rpc-server reports over RPC; ram_*: the machine (agent facts)
    return [
        # x86 laptop, CUDA 4 GB card: free assumed 3.9 GB (not recorded); no agent facts
        {"name": "x86-gpu", "ep": "x86:50052", "machine": "x86", "budget": None, "free_bytes": 3.9 * GB,
         "total_bytes": 4.29 * GB, "ram_avail": None, "ram_total": None, "va_cap": None},
        # x86 laptop CPU server: 24 GB usable (x86 laptop receipt), assumed reported as free
        {"name": "x86-cpu", "ep": "x86:50053", "machine": "x86", "budget": None, "free_bytes": 24 * GB,
         "total_bytes": 24 * GB, "ram_avail": None, "ram_total": None, "va_cap": None},
        # M1 Max: heap 60 GB (HK_SYSMEM wrapper, rec); RAM total 66.97 GB (rec); avail varies
        {"name": "m1max-vulkan", "ep": "m1max", "machine": "m1max", "budget": None, "free_bytes": 60 * GB,
         "total_bytes": 60 * GB, "ram_avail": m1_ram_avail, "ram_total": 66.97 * GB, "va_cap": None},
        # Mac Studio Metal: working set assumed 103 GB (not recorded; RAM binds anyway);
        # free+inactive+speculative 49.9 GB at 02:27Z (rec)
        {"name": "mac-metal", "ep": "mac", "machine": "mac", "budget": None, "free_bytes": 103 * GB,
         "total_bytes": 103 * GB, "ram_avail": 49.9 * GB, "ram_total": 137.4 * GB, "va_cap": None},
        # M2 Max CPU server: system RAM (rec: 89 GiB available at 02:27Z, total 101.2 GB)
        {"name": "m2-cpu", "ep": "m2:50061", "machine": "m2", "budget": None, "free_bytes": 89 * GIB,
         "total_bytes": 101.2 * GB, "ram_avail": 89 * GIB, "ram_total": 101.2 * GB, "va_cap": None},
        # M2 Max Vulkan: heap 82015 MiB (rec), allocation cap 63 GiB (vkalloc, rec)
        {"name": "m2-vulkan", "ep": "m2", "machine": "m2", "budget": None, "free_bytes": 82015 * 2 ** 20,
         "total_bytes": 82015 * 2 ** 20, "ram_avail": 89 * GIB, "ram_total": 101.2 * GB, "va_cap": 63 * GIB},
    ]


def show(label, ordered, h, host_bytes, paged, info):
    print("%s: host layers 0-%d %.2f GB, paged from disk %.2f GB, devices used %d" % (
        label, h - 1, host_bytes / GB, paged / GB, len(ordered)))
    first = h
    for d in ordered:
        n = min(d["weight"], info["n_layer"] - first)
        gb = sum(info["per_layer"][first:first + n]) / GB
        print("   %-12s layers %2d-%2d  %6.2f of %6.2f GB" % (d["name"], first, first + n - 1, gb, d["budget"] / GB))
        first += n


def main():
    info = glm53_iq1s_info()
    host_avail = 12.9 * GB  # host MemAvailable 12 GiB idle (rec 04:26Z)
    print("host page-cache budget %.2f GB (host 12.9 GB available, headroom %.1f, non-layer %.2f, compute %.1f)\n" % (
        lce.host_budget(host_avail, info) / GB, lce.HOST_HEADROOM / GB, info["other_bytes"] / GB, lce.HOST_COMPUTE / GB))
    hand = [2.75 * GB, 22 * GB, 56 * GB, 42 * GB, 24 * GB, 62 * GB]
    names = ["x86-gpu", "x86-cpu", "m1max-vulkan", "mac-metal", "m2-cpu", "m2-vulkan"]
    hp = [{"name": names[i], "weight": w, "budget": hand[i]} for i, w in zip([0, 1, 5, 2, 3, 4], [1, 7, 23, 20, 15, 9])]
    show("hand-built P (run-P table)", hp, 5, sum(info["per_layer"][:5]), 0, info)
    for label, jw in (("measured, M1 Max scratch cleared (avail 60.3 GB)", 60.3 * GB),
                      ("measured, M1 Max as at 02:27Z (avail 51.5 GB)", 51.5 * GB)):
        print()
        show(label, *cli._plan_devices(devices(jw), info, 2048, host_avail), info)


if __name__ == "__main__":
    main()
