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


def test_a_node_that_already_holds_the_model_is_not_charged_a_second_copy():
    # 16 GB free, but the complete 17.2 GB model is already in its cache: serving it downloads nothing
    holder = node(90, 94, disk_gb=16)
    holder["cached_models"] = {INFO["ref"]: INFO["weights_bytes"]}
    plan = planner.plan_placement({"holder": holder}, {"pairs": []}, INFO)
    assert (plan["mode"], plan["stages"][0]["node"]) == ("replica", "holder")


def test_a_partly_cached_model_is_charged_only_its_missing_bytes():
    half = node(90, 94, disk_gb=9)
    half["cached_models"] = {INFO["ref"]: INFO["weights_bytes"] // 2}  # 8.6 GB present + 9 GB free > 17.2 GB
    assert planner.plan_placement({"half": half}, {"pairs": []}, INFO)["mode"] == "replica"
    empty = node(90, 94, disk_gb=9)
    assert planner.plan_placement({"empty": empty}, {"pairs": []}, INFO)["mode"] == "none"


def test_a_cached_copy_of_another_model_buys_nothing():
    other = node(90, 94, disk_gb=16)
    other["cached_models"] = {"someone/else": INFO["weights_bytes"]}
    assert planner.plan_placement({"other": other}, {"pairs": []}, INFO)["mode"] == "none"


def test_a_node_with_no_disk_does_not_hide_a_placement_on_the_others():
    # ranked by total memory a(62) < b(94) < c(128); b has no disk, a and c each fit a part and share a measured route
    nodes = {"a": node(60, 62, disk_gb=10), "b": node(90, 94, disk_gb=0), "c": node(120, 128, disk_gb=10)}
    plan = planner.plan_placement(nodes, links("a", "c"), INFO)
    assert plan["mode"] == "pipeline" and sorted(s["node"] for s in plan["stages"]) == ["a", "c"]


def test_facts_list_the_complete_cached_models_by_weight_bytes(tmp_path, monkeypatch):
    from omarchy_cluster import facts
    from omarchy_cluster import models
    monkeypatch.setattr(models, "cache_dir", lambda: str(tmp_path))
    snap = tmp_path / "models--org--good/snapshots/abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x" * 1234)
    partial = tmp_path / "models--org--partial/snapshots/abc"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")  # no weights: not a cached model
    assert facts.collect_facts()["cached_models"] == {"org/good": 1234}


def _cached_repo(cache, revisions, main):
    """org/m in a HF-shaped cache: {revision: weight bytes} (sparse files), refs/main -> main."""
    repo = cache / "models--org--m"
    for rev, size in revisions.items():
        snap = repo / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        with open(snap / "model.safetensors", "wb") as f:
            f.truncate(size)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(main)
    return repo


def test_the_cache_credit_is_the_revision_serve_loads_not_the_first_by_name(tmp_path, monkeypatch):
    from omarchy_cluster import facts, models
    monkeypatch.setattr(models, "cache_dir", lambda: str(tmp_path))
    _cached_repo(tmp_path, {"a-old": GB, "z-main": int(17.2 * GB)}, main="z-main")
    assert facts.collect_facts()["cached_models"] == {"org/m": int(17.2 * GB)}


def test_a_local_snapshot_path_gets_the_same_credit_as_its_repo_id():
    ref = "/Users/someone/.cache/huggingface/hub/models--org--m/snapshots/z-main"  # need not exist on the node planning
    held = node(90, 94, disk_gb=16)
    held["cached_models"] = {"org/m": INFO["weights_bytes"]}
    plan = planner.plan_placement({"holder": held}, {"pairs": []}, {**INFO, "ref": ref})
    assert plan["mode"] == "replica"


def test_a_node_with_no_disk_at_all_still_reports_disk_not_missing_decode_nodes():
    plan = planner.plan_placement({"only": node(90, 94, disk_gb=0)}, {"pairs": []}, INFO)
    assert plan["mode"] == "none" and "disk" in plan["reason"]


def test_prefill_only_leftovers_after_the_disk_filter_report_disk():
    nodes = {"a": node(90, 94, disk_gb=0), "b": node(60, 62, disk_gb=200)}
    plan = planner.plan_placement(nodes, links("a", "b"), INFO, no_decode=("b",))
    assert plan["mode"] == "none" and "disk" in plan["reason"]
