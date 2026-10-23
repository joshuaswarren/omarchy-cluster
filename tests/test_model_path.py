import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omarchy_cluster import planner


@pytest.mark.parametrize("state", ["absent", "no-snapshots", "snapshots-file", "empty"])
def test_incomplete_cache_is_a_model_miss(tmp_path, monkeypatch, state):
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache/huggingface/hub/models--example--model"
    if state != "absent":
        cache.mkdir(parents=True)
    if state == "snapshots-file":
        (cache / "snapshots").write_text("incomplete cache")
    elif state == "empty":
        (cache / "snapshots").mkdir()

    assert planner.find_model_path("example/model") is None
    assert planner.model_info("example/model") is None


def test_no_main_ref_chooses_first_complete_snapshot_in_sorted_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    snapshots = tmp_path / ".cache/huggingface/hub/models--example--model/snapshots"
    for revision in ("zzz", "aaa", "000-incomplete"):
        snapshot = snapshots / revision
        snapshot.mkdir(parents=True)
        if revision != "000-incomplete":
            (snapshot / "config.json").write_text("{}")
    # Directory enumeration order must not choose the newest-created or first-listed revision.
    assert planner.find_model_path("example/model") == str(snapshots / "aaa")


def test_explicit_local_directory_takes_precedence_over_cached_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "example/model"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}")
    cached = tmp_path / ".cache/huggingface/hub/models--example--model/snapshots/rev"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")

    assert planner.find_model_path("example/model") == "example/model"
    assert planner.model_info("example/model")["path"] == "example/model"
