import pytest

from omarchy_cluster import discover, pins


@pytest.fixture(autouse=True)
def _isolated_trust_record(tmp_path_factory, monkeypatch):
    """No test may write this machine's real record of joined nodes."""
    trust = tmp_path_factory.mktemp("trust")
    monkeypatch.setattr(discover, "JOINED", str(trust / "joined"))
    monkeypatch.setattr(pins, "PINS", str(trust / "pins"))  # nor its pins, nor read this machine's own agent key
    monkeypatch.setattr(pins, "LOCAL_CERT_DIR", str(trust / "no-local-agent"))
