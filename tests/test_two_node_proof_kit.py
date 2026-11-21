import hashlib
import importlib.util
import json
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "receipts" / "two-node-proof-kit"


def load_verifier():
    spec = importlib.util.spec_from_file_location("two_node_check_proof", KIT / "check-proof.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kit_hashes_match_kit_sha256():
    for line in (KIT / "kit.sha256").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((KIT / name).read_bytes()).hexdigest() == digest, name


def test_reference_receipt_matches_verifier_expectations():
    verifier = load_verifier()
    receipt = json.loads((KIT / "reference-combine.json").read_text())
    assert receipt["exact_token_parity"] is True
    assert receipt["firewall_restored"] is True
    digests = set()
    for arm in ("mac_only", "spark_only", "shared"):
        data = receipt[arm]
        assert data["tokens"] == 64
        assert data["ids_sha256"] == verifier.ID_SHA
        assert data["decode_tokens_per_s"] > 0
        digests.add(data["ids_sha256"])
    assert len(digests) == 1
    assert receipt["mac_only"]["vulkan"]["jobs"] > 0
    shared = receipt["shared"]
    assert shared["vulkan"]["jobs"] > 0
    assert shared["cuda"]["model_buffer_mib"] > 0
    assert all(mib > 0 for mib in shared["cuda"]["pid_memory_mib"])
    assert shared["rpc_reply_bytes"] > 0
