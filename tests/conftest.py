import pytest

from omarchy_cluster import discover


@pytest.fixture(autouse=True)
def _isolated_trust_record(tmp_path_factory, monkeypatch):
    """No test may write this machine's real record of joined nodes."""
    monkeypatch.setattr(discover, "JOINED", str(tmp_path_factory.mktemp("trust") / "joined"))
