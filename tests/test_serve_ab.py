"""Pure-part tests for the serve A/B arm runner (no network, no server)."""

from __future__ import annotations

import hashlib
import json
import os

from omarchy_cluster import serve_ab


def test_spec_pins_are_complete():
    assert len(serve_ab.SPEC["sha256"]) == 64
    assert serve_ab.SPEC["bytes"] > 0
    assert serve_ab.SPEC["file"].endswith(".gguf")
    assert serve_ab.SPEC["revision"] and len(serve_ab.SPEC["revision"]) == 40


def test_prompts_are_fixed_and_named():
    assert [pid for pid, _ in serve_ab.PROMPTS] == ["p1-short", "p2-switch", "p3-long"]
    assert all(isinstance(text, str) and text for _, text in serve_ab.PROMPTS)


def test_parse_devices_handles_empty_and_gpu_output():
    empty = serve_ab.parse_devices("Available devices:\n  (none)\n")
    assert empty == []
    gpu = serve_ab.parse_devices(
        "Available devices:\n  CUDA0: NVIDIA GeForce RTX 5060 Ti\n"
        "  Vulkan0: AMD Radeon B60\n")
    assert gpu == ["CUDA0: NVIDIA GeForce RTX 5060 Ti", "Vulkan0: AMD Radeon B60"]


def test_medians_per_prompt():
    runs = [
        {"prompt": "p1-short", "prefill_ms": 10.0, "tokps": 40.0},
        {"prompt": "p1-short", "prefill_ms": 20.0, "tokps": 60.0},
        {"prompt": "p1-short", "prefill_ms": 30.0, "tokps": 80.0},
        {"prompt": "p2-switch", "prefill_ms": 100.0, "tokps": 10.0},
    ]
    out = serve_ab.medians(runs)
    assert out["p1-short"] == {"ttft_ms_median": 20.0, "tokps_median": 60.0}
    assert out["p2-switch"] == {"ttft_ms_median": 100.0, "tokps_median": 10.0}


def test_run_digest_is_order_sensitive():
    a = hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest()
    b = hashlib.sha256(json.dumps([2, 1]).encode()).hexdigest()
    assert a != b
    combined = hashlib.sha256((a + b).encode()).hexdigest()
    assert len(combined) == 64


def test_server_command_pins_device_when_given():
    base = serve_ab.server_command("/bin/llama-server", serve_ab.Path("/m.gguf"), 8129, 8, 4096, None)
    assert "-dev" not in base
    pinned = serve_ab.server_command("/bin/llama-server", serve_ab.Path("/m.gguf"), 8129, 8, 4096, "Vulkan0")
    assert pinned[-2:] == ["-dev", "Vulkan0"]
    assert "-ngl" in pinned and "99" in pinned


def test_device_names_extract_tokens():
    assert serve_ab.device_names([
        "Vulkan0: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)",
        "Vulkan1: NVIDIA GeForce RTX 5060 Ti"]) == ["Vulkan0", "Vulkan1"]


def test_scrub_drops_home_prefix_and_keeps_other_paths():
    import os
    home = os.path.expanduser("~")
    assert serve_ab.scrub(home + "/scratch/serve-ab/m.gguf") == "~/scratch/serve-ab/m.gguf"
    assert serve_ab.scrub("/usr/bin/llama-server") == "/usr/bin/llama-server"


def test_uname_string_carries_no_nodename():
    assert os.uname().nodename not in serve_ab.uname_string()
    assert serve_ab.uname_string().startswith(os.uname().sysname)


def test_machine_info_has_uname():
    info = serve_ab.machine_info()
    assert "uname" in info and isinstance(info["uname"], str)
