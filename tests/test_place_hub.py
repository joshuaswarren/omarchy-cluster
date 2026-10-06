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


def test_pipeline_plan_orders_decode_tail_and_caps_stages_by_free_memory():
    nodes = {"mac-a": facts(100, 128), "linux-b": facts(80, 94)}
    plan = planner.plan_placement(nodes, links_with("mac-a", "linux-b"), INFO)
    assert plan["mode"] == "pipeline"
    assert plan["decode_tail"] == "mac-a"  # biggest memory takes the tail
    assert [s["node"] for s in plan["stages"]] == ["linux-b", "mac-a"]
    # each stage may take every layer but one (the other rank keeps one)
    assert [s["max_layers"] for s in plan["stages"]] == [27, 27]
    assert plan["boundaries"][0]["gbps"] == 2.2
    assert plan["boundaries"][0]["per_token_ms"] > 0


def test_plan_infeasible_when_stages_together_cannot_hold_the_model():
    per_layer_gb = (INFO["weights_bytes"] + planner.kv_bytes_per_token(INFO) * 2048) / 28 / 1e9
    small = per_layer_gb * 10 / 0.9  # free memory for 10 layers per node
    nodes = {"a": facts(small, 1), "b": facts(small, 2)}
    assert planner.plan_placement(nodes, links_with("a", "b"), INFO, max_stages=2)["mode"] == "none"
    nodes = {"a": facts(small * 2, 1), "b": facts(small, 2)}
    assert planner.plan_placement(nodes, links_with("a", "b"), INFO, max_stages=2)["mode"] == "pipeline"


def test_choose_split_puts_layers_on_the_cheapest_rank_up_to_its_cap():
    # measured: mac-a 1.0 ms/layer, linux-b 8.2 ms/layer (ClusterRun5)
    assert planner.choose_split([1.0, 8.2], [120, 200], 27) == [26, 1]
    assert planner.choose_split([8.2, 1.0], [120, 200], 27) == [1, 26]
    assert planner.choose_split([1.0, 8.2], [20, 200], 27) == [20, 7]  # memory cap binds
    assert planner.choose_split([2.0, 1.0, 3.0], [5, 10, 30], 27) == [5, 10, 12]
    assert planner.choose_split([1.0, 1.0], [30, 30], 27) == [26, 1]  # tie: lower rank first
    assert planner.choose_split([1.0, 8.2], [10, 10], 27) is None
    assert planner.choose_split([1.0, 8.2], [27, 0], 27) is None


def test_max_layers_counts_weights_and_kv_per_layer():
    per_layer = (INFO["weights_bytes"] + planner.kv_bytes_per_token(INFO) * 2048) / 28
    assert planner.max_layers(per_layer * 10 / 0.9 + 1, INFO) == 10
    assert planner.max_layers(per_layer * 10 / 0.9 - 1, INFO) == 9


def test_no_decode_node_can_prefill_but_cannot_be_decode_tail():
    nodes = {"mac-a": facts(100, 128), "linux-c": facts(80, 94)}
    plan = planner.plan_placement(nodes, links_with("mac-a", "linux-c"), INFO,
                                  no_decode=["mac-a"], max_stages=2)
    assert plan["mode"] == "pipeline"
    assert [stage["node"] for stage in plan["stages"]] == ["mac-a", "linux-c"]
    assert plan["decode_tail"] == "linux-c"


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
