import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import cli, llamacpp_engine as lce, vkprobe

GB = 10 ** 9


def glm53_iq1s_info():
    """GLM-5.3 UD-IQ1_S per-layer bytes from its GGUF headers (79 layers; the last is the
    MTP layer llama.cpp skips, charged 0 as gguf_info does)."""
    D, A, B, C, E = 309201920, 2668793856, 3071447040, 3571027968, 3549597696
    per = [A] * 79
    per[0:3] = [D] * 3
    for i in (4, 6, 7, 9, 29, 40, 43, 47, 48, 49, 50, 68, 69, 70, 71, 72, 73, 74):
        per[i] = B
    per[8] = C
    per[75:78] = [E] * 3
    per[78] = 0
    return {"n_layer": 79, "per_layer": per, "layer_bytes": max(per), "other_bytes": 1070555136,
            "total_bytes": 216_710_000_000, "kv_bytes_per_token_layer": 2176}


def small_info(sizes):
    return {"n_layer": len(sizes), "per_layer": [int(s * GB) for s in sizes], "layer_bytes": int(max(sizes) * GB),
            "other_bytes": GB // 2, "total_bytes": int(sum(sizes) * GB) + GB // 2, "kv_bytes_per_token_layer": 0}


def test_device_budget_takes_the_tightest_limit_less_the_reserve():
    # an M2 Max: heap 86 GB, 95.6 GB of RAM free, Honeykrisp stops at 63 GiB
    assert lce.device_budget(86 * GB, 95.6 * GB, 63 * 2 ** 30) == 63 * 2 ** 30 - lce.GPU_RESERVE
    # Metal reports a big working set; RAM minus headroom is what binds
    assert lce.device_budget(103 * GB, 49.9 * GB) == int(49.9 * GB - lce.NODE_HEADROOM - lce.GPU_RESERVE)
    assert lce.device_budget(20 * GB, None, gpu=False) == 20 * GB - lce.CPU_RESERVE
    assert lce.device_budget(GB // 2, None) == 0


def test_two_servers_on_one_machine_share_its_ram_cpu_first():
    gpu = {"name": "m2-gpu", "machine": "m2", "budget": 65 * GB, "ram_avail": 95 * GB, "gpu": True}
    cpu = {"name": "m2-cpu", "machine": "m2", "budget": 80 * GB, "ram_avail": 95 * GB, "gpu": False}
    other = {"name": "mac", "machine": "mac", "budget": 40 * GB, "ram_avail": 50 * GB, "gpu": True}
    lce.pool_machines([gpu, cpu, other])
    room = 95 * GB - lce.NODE_HEADROOM - lce.GPU_RESERVE - lce.CPU_RESERVE
    assert gpu["budget"] == 65 * GB and cpu["budget"] == room - 65 * GB
    assert other["budget"] == 40 * GB


def test_place_pages_nothing_first_then_uses_fewest_devices():
    info = small_info([1, 1, 1, 1, 1, 1])
    # room for 2 resident host layers: two devices suffice (4 + 2 on host), the third stays unused
    r = lce.place(info, [2 * GB, 2 * GB, 1 * GB], 2 * GB, 0)
    assert (r["host_layers"], r["paged_bytes"]) == (2, 0)
    assert sum(1 for w in r["weights"] if w) == 2
    # room for only 1: all three devices rather than page a second host layer from disk
    r = lce.place(info, [2 * GB, 2 * GB, 1 * GB], 1 * GB, 0)
    assert (r["host_layers"], r["paged_bytes"]) == (1, 0)
    assert sum(1 for w in r["weights"] if w) == 3


def test_place_pages_the_least_when_it_must():
    info = small_info([1, 1, 1, 1, 1, 1])
    r = lce.place(info, [2 * GB, 1 * GB], 0, 0)
    assert (r["host_layers"], r["paged_bytes"]) == (3, 3 * GB)


def test_place_can_keep_the_given_order_and_host_layers():
    info = small_info([1, 1, 1, 1, 1, 1])
    r = lce.place(info, [1 * GB, 4 * GB], 10 * GB, 0, host_layers=1, keep_order=True)
    assert (r["order"], r["host_layers"], r["weights"]) == ([0, 1], 1, [1, 5])
    with pytest.raises(ValueError):
        lce.place(info, [1 * GB, 1 * GB], 0, 0, host_layers=0, keep_order=True)


def test_place_reproduces_the_hand_built_glm53_placement():
    """Run P (2026-10-07): host layers 0-4 resident on a 16 GB host; x86 CUDA 5, x86 CPU
    6-12, M2 Vulkan 13-35, M1 Max Vulkan 36-55, Mac Metal 56-70, M2 CPU 71-78."""
    budgets = [2.75 * GB, 22 * GB, 56 * GB, 42 * GB, 24 * GB, 62 * GB]  # x86 gpu, x86 cpu, m1 max, mac, m2 cpu, m2 gpu
    r = lce.place(glm53_iq1s_info(), budgets, 9 * GB, 2048)
    assert (r["host_layers"], r["paged_bytes"]) == (5, 0)
    assert r["order"] == [0, 1, 5, 2, 3, 4]
    assert r["weights"] == [1, 7, 23, 20, 15, 9]


def dev(name, machine, free, total, ram_avail, ram_total, budget=None, va=None):
    return {"name": name, "ep": name, "machine": machine, "budget": budget, "free_bytes": free,
            "total_bytes": total, "ram_avail": ram_avail, "ram_total": ram_total, "va_cap": va}


def test_plan_devices_measures_budgets_and_drops_unused_devices():
    info = small_info([1, 1, 1, 1])
    devs = [dev("big-gpu", "a", 50 * GB, 60 * GB, 40 * GB, 64 * GB),
            dev("spare-cpu", "b", 30 * GB, 32 * GB, 30 * GB, 32 * GB)]
    ordered, h, _, paged = cli._plan_devices(devs, info, 0, 4 * GB)
    assert [d["name"] for d in ordered] == ["big-gpu"]  # one device holds all four layers
    assert ordered[0]["gpu"] is True and ordered[0]["measured"] is True
    assert devs[1]["gpu"] is False  # llama.cpp total equal to the machine's RAM: a CPU device
    assert (h, paged) == (0, 0)


def test_plan_devices_keeps_a_given_budget_and_order_with_host_layers():
    info = small_info([1, 1, 1, 1])
    devs = [dev("a", "a", 50 * GB, 60 * GB, 40 * GB, 64 * GB, budget=1 * GB),
            dev("b", "b", 50 * GB, 60 * GB, 40 * GB, 64 * GB, budget=2 * GB)]
    ordered, h, _, _ = cli._plan_devices(devs, info, 0, 0, host_layers=1)
    assert [(d["name"], d["weight"], d["budget"]) for d in ordered] == [("a", 1, 1 * GB), ("b", 3, 2 * GB)]
    assert h == 1


def test_plan_devices_refuses_an_unmeasured_device_without_memory():
    with pytest.raises(ValueError):
        cli._plan_devices([dev("x", "x", None, None, None, None)], small_info([1]), 0, 0)


def test_vkprobe_counts_blocks_until_refused_and_frees_every_one():
    freed = []
    left = iter([1, 2, 3, None])
    assert vkprobe.count_blocks(lambda: next(left), freed.append, 10) == 3
    assert freed == [1, 2, 3]
    freed.clear()
    assert vkprobe.count_blocks(iter(range(100)).__next__, freed.append, 4) == 4
    assert len(freed) == 4


def test_agent_gpu_cap_probes_within_spare_ram_and_caches_per_boot(tmp_path, monkeypatch):
    """Probe blocks past free RAM take real memory (an M1 Max evicted 17 GiB of page cache),
    so the agent caps the probe at available RAM minus the 6 GB headroom."""
    import json as _json
    from omarchy_cluster import agent
    seen = []

    class Done:
        returncode, stderr = 0, ""
        stdout = _json.dumps({"device": "gpu", "heap_bytes": 70 * GB, "alloc_cap_bytes": 40 * 2 ** 30,
                              "stopped_at_limit": True})

    def fake_run(cmd, env=None, **kw):
        seen.append((cmd, env))
        return Done()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent.sys, "platform", "linux")
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    monkeypatch.setattr(agent.facts_mod, "boot_id", lambda: "boot-1")
    monkeypatch.setattr(agent.facts_mod, "memory_total_bytes", lambda: 67 * GB)
    monkeypatch.setattr(agent.facts_mod, "memory_free_bytes", lambda: 50 * GB)
    res = agent.gpu_cap()
    assert res["alloc_cap_bytes"] == 40 * 2 ** 30 and res["boot_id"] == "boot-1"
    cmd, env = seen[0]
    assert cmd[-1] == str((50 * GB - 6 * GB) // 2 ** 30) and env["HK_SYSMEM"] == str(67 * GB)
    agent.gpu_cap()
    assert len(seen) == 1  # cached for this boot
