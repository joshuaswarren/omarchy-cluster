from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "omarchy-spark-runbook.md"


def _text():
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_exists_and_covers_cl14_sections():
    text = _text().lower()
    for needle in (
        "cold boot",
        "cuda",
        "rpc",
        "mdns",
        "ufw",
        "pacman -syu",
        "rollback",
        "snapper",
        "efibootmgr",
        "<remnic_core_port>",
        "<reranker_port>",
    ):
        assert needle in text, f"runbook missing CL14 element: {needle}"


def test_runbook_references_existing_repo_paths():
    text = _text()
    for rel in (
        "spark-omarchy/PHYSICAL-INSTALL.md",
        "spark-omarchy/UPDATE-AND-ROLLBACK.md",
        "docs/benchmarks.md",
    ):
        assert rel in text, f"runbook does not reference {rel}"
        assert (ROOT / rel).exists(), f"referenced path {rel} does not exist"


def test_rollback_proof_precedes_service_restore():
    text = _text()
    rollback = text.index("## 5. Rollback proof")
    restore = text.index("## 7. Memory-service restore")
    assert rollback < restore, "service restore must not precede the rollback proof"


def test_no_lane_ports_or_hostnames():
    text = _text()
    for leaked in (":8095", ":8098", "10.77.", "192.168."):
        assert leaked not in text, f"runbook leaks a service literal: {leaked}"
