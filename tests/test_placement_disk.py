"""The planner counts the disk a rank must download into, not only memory (issue 81): a 17 GB model that fits a 94 GB
Mac's memory is not placed whole on it when its model volume has 16 GB free."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import planner

GB = 10**9
INFO = {"ref": "m", "path": None, "layers": 48, "hidden": 2048, "kv_heads": 4, "head_dim": 128,
        "weights_bytes": int(17.2 * GB)}


def node(free_gb, total_gb, disk_gb=None):
    facts = {"memory_free_bytes": int(free_gb * GB), "memory_total_bytes": int(total_gb * GB), "unified_memory": True}
    if disk_gb is not None:
        facts["model_volume_free_bytes"] = int(disk_gb * GB)
    return facts


def links(*names):
    pairs = []
    for a, b in zip(names, names[1:]):
        pairs.append({"a": a, "b": b, "routes": [], "pinned": {
            "a_node": a, "b_node": b, "a_iface": "en0", "a_ip": "10.0.0.1", "b_iface": "en1", "b_ip": "10.0.0.2",
            "gbps_a2b": 2.0, "gbps_b2a": 2.0, "rtt_tcp_ms": 0.3, "eligible_for_decode": True}})
    return {"pairs": pairs}


def test_a_whole_placement_skips_the_node_whose_model_volume_cannot_hold_the_weights():
    nodes = {"roomy-ram": node(90, 94, disk_gb=16), "roomy-disk": node(60, 62, disk_gb=200)}
    plan = planner.plan_placement(nodes, links("roomy-ram", "roomy-disk"), INFO)
    assert (plan["mode"], plan["stages"][0]["node"]) == ("replica", "roomy-disk")


def test_when_no_node_has_the_disk_the_reason_says_disk_not_memory():
    nodes = {"a": node(90, 94, disk_gb=8), "b": node(60, 62, disk_gb=5)}
    plan = planner.plan_placement(nodes, links("a", "b"), INFO)
    assert plan["mode"] == "none" and "disk" in plan["reason"] and "memory" not in plan["reason"]


def test_a_stage_is_capped_by_the_disk_for_its_own_layers():
    nodes = {"a": node(90, 94, disk_gb=9), "b": node(60, 62, disk_gb=200)}
    plan = planner.plan_placement(nodes, links("a", "b"), INFO, max_stages=2)
    by_node = {s["node"]: s["max_layers"] for s in plan["stages"]}
    assert by_node["a"] <= int(9 * GB / (INFO["weights_bytes"] / INFO["layers"]))  # about 25 of 48 layers
    assert by_node["b"] >= 47


def test_a_node_that_does_not_report_its_disk_is_not_limited():
    nodes = {"a": node(90, 94), "b": node(60, 62)}
    assert planner.plan_placement(nodes, links("a", "b"), INFO)["mode"] == "replica"


def test_facts_report_the_free_bytes_of_the_model_volume(tmp_path, monkeypatch):
    from omarchy_cluster import facts
    monkeypatch.setenv("HOME", str(tmp_path))
    free = facts.model_volume_free()
    assert isinstance(free, int) and free > 0
    assert facts.collect_facts()["model_volume_free_bytes"] > 0
