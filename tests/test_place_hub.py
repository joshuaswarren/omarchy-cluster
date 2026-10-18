import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import cli, planner
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


@pytest.mark.parametrize("main_revision", ["999-current", "missing"])
def test_model_info_uses_cached_main_instead_of_an_old_snapshot(tmp_path, monkeypatch, main_revision):
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache/huggingface/hub/models--example--model"
    for revision, layers, weights in [("000-old", 2, 32), ("999-current", 8, 128)]:
        snapshot = cache / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"num_hidden_layers": layers}))
        (snapshot / "model.safetensors").write_bytes(b"0" * weights)
    (cache / "refs").mkdir()
    (cache / "refs/main").write_text(main_revision + "\n")

    info = planner.model_info("example/model")
    if main_revision == "missing":
        assert info is None
    else:
        assert info is not None
        assert info["path"] == str(cache / "snapshots" / "999-current")
        assert (info["layers"], info["weights_bytes"]) == (8, 128)


def test_pipeline_plan_orders_decode_tail_and_caps_stages_by_free_memory():
    nodes = {"big-desktop": facts(100, 128), "laptop": facts(80, 94)}
    plan = planner.plan_placement(nodes, links_with("big-desktop", "laptop"), INFO)
    assert plan["mode"] == "pipeline"
    assert plan["decode_tail"] == "big-desktop"  # biggest memory takes the tail
    assert [s["node"] for s in plan["stages"]] == ["laptop", "big-desktop"]
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
    # measured: desktop 1.0 ms/layer, laptop 8.2 ms/layer
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
    nodes = {"big-desktop": facts(100, 128), "laptop": facts(80, 94)}
    plan = planner.plan_placement(nodes, links_with("big-desktop", "laptop"), INFO,
                                  no_decode=["big-desktop"], max_stages=2)
    assert plan["mode"] == "pipeline"
    assert [stage["node"] for stage in plan["stages"]] == ["big-desktop", "laptop"]
    assert plan["decode_tail"] == "laptop"


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


def stored_route(a_iface, a_ip, a2b, b2a, rtt):
    return {"a_node": "a", "b_node": "b", "a_iface": a_iface, "a_ip": a_ip, "b_iface": "en1", "b_ip": "10.0.0.2",
            "gbps_a2b": a2b, "gbps_b2a": b2a, "rtt_tcp_ms": rtt, "eligible_for_decode": True}


def stale_links(with_wired=True):
    """A links.json written before probe.measured_gbps required both directions: the pin is the
    idle-bridge route whose reverse blast failed (the route set recorded on two laptops)."""
    bridge = stored_route("docker0", "172.17.0.1", 2.017, None, 0.39)
    routes = [bridge] + ([stored_route("en0", "10.0.0.1", 2.017, 1.992, 0.27)] if with_wired else [])
    return {"pairs": [{"a": "a", "b": "b", "routes": routes, "pinned": bridge}]}


def test_stored_one_way_pin_is_replaced_by_the_best_two_way_route():
    nodes = {"a": facts(100, 128), "b": facts(80, 94)}
    plan = planner.plan_placement(nodes, stale_links(), INFO, max_stages=2)
    assert plan["mode"] == "pipeline"
    assert plan["boundaries"][0]["route"].startswith("en0:10.0.0.1")
    assert plan["boundaries"][0]["gbps"] == 1.992
    route = planner.pinned_route(stale_links(), "a", "b")
    assert (route["a_iface"], route["a_ip"]) == ("en0", "10.0.0.1")


def test_stored_one_way_pin_with_no_two_way_route_fails_closed(monkeypatch):
    nodes = {"a": facts(100, 128), "b": facts(80, 94)}
    assert planner.plan_placement(nodes, stale_links(with_wired=False), INFO, max_stages=2)["mode"] == "none"
    assert planner.pinned_route(stale_links(with_wired=False), "a", "b") is None
    monkeypatch.setattr(cli.probe_mod, "load_links", lambda *a, **k: stale_links(with_wired=False))
    with pytest.raises(SystemExit, match="both directions"):
        cli._pick_route_ip({"facts": {"interfaces": []}}, [{"node": "a"}, {"node": "b"}], "a")


def test_a_two_way_route_on_a_bridge_is_still_pinned(monkeypatch):
    # The rule is both directions measured, not an interface name list.
    bridge = stored_route("docker0", "172.17.0.1", 2.0, 1.9, 0.2)
    links = {"pairs": [{"a": "a", "b": "b", "routes": [bridge], "pinned": bridge}]}
    assert planner.pinned_route(links, "a", "b") is bridge
    monkeypatch.setattr(cli.probe_mod, "load_links", lambda *a, **k: links)
    assert cli._pick_route_ip({"facts": {"interfaces": []}}, [{"node": "a"}, {"node": "b"}], "a") == "172.17.0.1"


def test_ring_address_follows_the_two_way_route_for_a_stale_stored_pin(monkeypatch):
    monkeypatch.setattr(cli.probe_mod, "load_links", lambda *a, **k: stale_links())
    stages = [{"node": "a"}, {"node": "b"}]
    assert cli._pick_route_ip({"facts": {"interfaces": []}}, stages, "a") == "10.0.0.1"
    assert cli._pick_route_ip({"facts": {"interfaces": []}}, stages, "b") == "10.0.0.2"
