import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import planner
from omarchy_cluster.hub import Registry


def facts(free_gb, total_gb, unified=True):
    return {"memory_free_bytes": int(free_gb * 1e9), "memory_total_bytes": int(total_gb * 1e9),
            "unified_memory": unified}


def links_with(a, b, gbps=2.2, rtt=0.3):
    return {"pairs": [{"a": a, "b": b, "routes": [], "pinned": {
        "a_node": a, "b_node": b, "a_iface": "en0", "a_ip": "10.0.0.1",
        "b_iface": "en1", "b_ip": "10.0.0.2", "gbps_a2b": gbps, "gbps_b2a": gbps,
        "rtt_tcp_ms": rtt, "eligible_for_decode": True}}]}


INFO = {"ref": "m", "path": "/x", "layers": 28, "hidden": 2048, "kv_heads": 8,
        "head_dim": 128, "weights_bytes": int(1.2e9)}


def test_pipeline_split_fits_and_orders_decode_tail():
    nodes = {"mac-a": facts(100, 128), "linux-b": facts(80, 94)}
    plan = planner.plan_placement(nodes, links_with("mac-a", "linux-b"), INFO)
    assert plan["mode"] == "pipeline"
    assert plan["decode_tail"] == "mac-a"  # biggest memory takes the tail
    assert plan["stages"][0]["node"] == "linux-b"
    a0, b0 = plan["stages"][0]["layers"]
    a1, b1 = plan["stages"][1]["layers"]
    assert (a0, b1) == (0, 28) and b0 == a1  # contiguous cover
    for s in plan["stages"]:
        assert s["weights_bytes"] + s["kv_bytes"] < facts(*[0, 0])["memory_free_bytes"] + 1e18
        assert s["weights_bytes"] + s["kv_bytes"] <= 0.9 * (
            nodes[s["node"]]["memory_free_bytes"])
    assert plan["boundaries"][0]["gbps"] == 2.2
    assert plan["boundaries"][0]["per_token_ms"] > 0


def test_no_decode_excluded():
    nodes = {"linux-d": facts(8, 8), "mac-a": facts(100, 128)}
    plan = planner.plan_placement(nodes, links_with("linux-d", "mac-a"), INFO,
                                  no_decode=["linux-d"])
    assert [s["node"] for s in plan["stages"]] == ["mac-a"]
    assert plan["mode"] == "replica"


def test_infeasible_when_model_does_not_fit():
    nodes = {"tiny": facts(0.1, 1)}
    plan = planner.plan_placement(nodes, links_with("tiny", "tiny"), INFO)
    assert plan["mode"] == "none"


def test_kv_bytes():
    assert planner.kv_bytes_per_token(INFO) == 2 * 28 * 8 * 128 * 2


def test_hub_registry_marks_out():
    reg = Registry()
    reg.beat("a", 5)
    nodes = reg.nodes()
    assert nodes["a"]["up"] and nodes["a"]["seq"] == 5
    # simulate stale heartbeat
    import time as t
    reg.last["a"] = (t.monotonic() - 10, 5)
    assert reg.nodes()["a"]["up"] is False
    assert reg.up_names() == set()
