import os

from omarchy_cluster import rank


def _reset(monkeypatch, path):
    monkeypatch.setattr(rank, "_LIVE_ENV_FILE", str(path))
    monkeypatch.setattr(rank, "_live_applied", {})


def test_live_env_sets_changes_and_unsets(tmp_path, monkeypatch):
    f = tmp_path / "live-env"
    _reset(monkeypatch, f)
    monkeypatch.delenv("W7G_LIVE_A", raising=False)
    monkeypatch.delenv("W7G_LIVE_B", raising=False)
    f.write_text("# comment\nW7G_LIVE_A=1\nW7G_LIVE_B = x y\n")
    assert rank._apply_live_env() == {"W7G_LIVE_A": "1", "W7G_LIVE_B": "x y"}
    assert os.environ["W7G_LIVE_A"] == "1" and os.environ["W7G_LIVE_B"] == "x y"
    f.write_text("W7G_LIVE_A=0\n")
    rank._apply_live_env()
    assert os.environ["W7G_LIVE_A"] == "0"
    assert "W7G_LIVE_B" not in os.environ
    f.unlink()
    rank._apply_live_env()
    assert "W7G_LIVE_A" not in os.environ


def test_live_env_leaves_variables_it_never_set(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path / "missing")
    monkeypatch.setenv("W7G_LIVE_KEEP", "stay")
    rank._apply_live_env()
    assert os.environ["W7G_LIVE_KEEP"] == "stay"
