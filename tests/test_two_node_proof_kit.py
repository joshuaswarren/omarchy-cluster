import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "receipts" / "two-node-proof-kit"
# Real `llama-server --version` stderr from an upstream llama.cpp build at commit 19bc41d.
REAL_LLAMA_VERSION = "version: 1.0.0-dev (build 1, commit 19bc41d)\nbuilt with GNU 12.2.0 for Linux x86_64\n"


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


def write_arm(root, name, ids):
    """Write one arm's token log and response so parse_arm can read it."""
    root.mkdir(parents=True, exist_ok=True)
    lines = [f"n_gen = {index}, n_remaining = -1, next token: {token}" for index, token in enumerate(ids, start=1)]
    (root / f"{name}-tokens.log").write_text("\n".join(lines) + "\n")
    (root / f"{name}-response.json").write_text(
        json.dumps(
            {
                "usage": {"prompt_tokens": 43, "completion_tokens": 64},
                "timings": {"tokens_per_s": 1.0},
            }
        )
    )


def test_parse_arm_accepts_token_ids_that_differ_from_the_reference():
    """An outside tester's tokens need not equal our reference digest."""
    verifier = load_verifier()
    root = KIT / "tmp-parity-check"
    ids = [1000 + index for index in range(64)]
    write_arm(root, "mac", ids)
    try:
        arm = verifier.parse_arm(root, "mac")
    finally:
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
    assert arm["ids_sha256"] != verifier.ID_SHA
    assert arm["matches_reference"] is False


def test_check_parity_accepts_one_shared_digest_and_rejects_mismatched_arms():
    verifier = load_verifier()
    digest = "a" * 64
    other = "b" * 64
    assert verifier.check_parity([digest, digest, digest]) is True
    try:
        verifier.check_parity([digest, other, digest])
    except ValueError as error:
        assert "differ" in str(error)
    else:
        raise AssertionError("mismatched arm digests must fail the parity check")


def test_cross_host_llama_versions_must_match():
    """The two hosts must run the same llama.cpp build for one shared request."""
    verifier = load_verifier()
    mac = "version: 0.7.0-dev (build 1, commit c7fc509)\n"
    spark = "version: 0.7.0-dev (build 4, commit c7fc509)\n"
    other = "version: 0.9.0-dev (build 1, commit 9f0e509)\n"
    assert verifier.check_llama_versions(mac, spark) is True
    try:
        verifier.check_llama_versions(mac, other)
    except ValueError as error:
        assert "llama-server" in str(error)
    else:
        raise AssertionError("mismatched llama-server builds must fail")


def test_help_names_the_pinned_source_commit():
    """A stranger must be told which commit SOURCE must contain."""
    text = (KIT / "two-node-proof.sh").read_text()
    assert "10a47b69df53f60b1e9b3f67aa5092fdff2c4562" in text.split("commit=")[1].split("\n")[0]
    assert "merged PR #117" not in text


def test_check_reports_all_prerequisites_without_running_the_proof(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not the pinned model")
    tls = tmp_path / "tls"
    tls.mkdir(mode=0o700)
    out = tmp_path / "out"
    result = subprocess.run(
        [
            "bash",
            str(KIT / "two-node-proof.sh"),
            "--check",
            "spark",
            str(source),
            str(model),
            str(out),
            "/bin/false",
            str(tls),
            "192.0.2.10",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2
    assert "FAIL source commit present" in result.stdout
    assert "FAIL model sha256 matches" in result.stdout
    assert "FAIL llama-server version readable" in result.stdout
    assert "FAIL TLS material and modes" in result.stdout
    assert "CHECKS " in result.stdout
    assert not out.exists()


def test_check_reports_missing_parent_paths_instead_of_aborting(tmp_path):
    """--check must print FAIL lines, not crash, when a path parent does not exist."""
    tls = tmp_path / "tls"
    tls.mkdir(mode=0o700)
    missing = tmp_path / "no-such-dir"
    result = subprocess.run(
        [
            "bash",
            str(KIT / "two-node-proof.sh"),
            "--check",
            "spark",
            str(missing / "repo"),
            str(missing / "model.gguf"),
            str(missing / "out"),
            str(missing / "llama-server"),
            str(tls),
            "192.0.2.10",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "FAIL source commit present" in result.stdout
    assert "FAIL model file present" in result.stdout
    assert "FAIL disk space" in result.stdout
    assert "CHECKS " in result.stdout


def expect_value_error(call, *args, match):
    try:
        call(*args)
    except ValueError as error:
        assert match in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {match!r}")


def test_llama_build_reads_the_commit_from_real_llama_server_output():
    verifier = load_verifier()
    assert verifier.llama_build(REAL_LLAMA_VERSION) == {"version": "1.0.0-dev", "commit": "19bc41d"}
    assert verifier.llama_build("version: 5123 (aaaaaaa)\nbuilt with cc for x86_64\n")["commit"] == "aaaaaaa"
    expect_value_error(verifier.llama_build, "version: 1.0.0-dev\n", match="cannot read")


def test_cross_host_llama_commit_mismatch_fails_loudly():
    """Same version string, different commits: the old regex accepted this pair."""
    verifier = load_verifier()
    other = REAL_LLAMA_VERSION.replace("19bc41d", "deadbee")
    assert verifier.check_llama_versions(REAL_LLAMA_VERSION, REAL_LLAMA_VERSION) is True
    expect_value_error(verifier.check_llama_versions, REAL_LLAMA_VERSION, other, match="llama-server")
    expect_value_error(verifier.check_llama_versions, "version: 5123 (aaaaaaa)\n", "version: 5123 (bbbbbbb)\n", match="llama-server")
    expect_value_error(verifier.check_llama_versions, REAL_LLAMA_VERSION, "", match="cannot read")
    # One clone can abbreviate the same commit with more hex digits than another.
    assert verifier.check_llama_versions(REAL_LLAMA_VERSION, REAL_LLAMA_VERSION.replace("19bc41d", "19bc41db")) is True


def test_check_reads_llama_server_version_from_stderr(tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\nprintf '%s' '" + REAL_LLAMA_VERSION + "' >&2\n")
    fake.chmod(0o755)
    tls = tmp_path / "tls"
    tls.mkdir(mode=0o700)
    result = subprocess.run(
        ["bash", str(KIT / "two-node-proof.sh"), "--check", "spark", str(tmp_path / "src"), str(tmp_path / "m.gguf"),
         str(tmp_path / "out"), str(fake), str(tls), "192.0.2.10"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert "PASS llama-server version readable" in result.stdout, result.stdout + result.stderr


UFW_BEFORE = """Status: active

     To                         Action      From
     --                         ------      ----
[ 1] 22/tcp                     ALLOW IN    Anywhere
"""
UFW_DURING = UFW_BEFORE + "[ 2] 24861/tcp                  ALLOW IN    192.0.2.11                 # cluster-proof-0123456789abcdef\n"
ENGINE_LOG = """llama_prepare_model_devices: using device CUDA0 (NVIDIA GB10)
llama_prepare_model_devices: using device RPC0 (127.0.0.1:24864)
load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size =   512.00 MiB
"""


def write_receipts(tmp_path):
    """Write a complete, passing Mac and Spark receipt pair for verify()."""
    mac, spark = tmp_path / "mac", tmp_path / "spark"
    ids = list(range(64))
    write_arm(mac, "mac", ids)
    write_arm(spark, "spark", ids)
    write_arm(spark, "shared", ids)
    for root in (mac, spark):
        (root / "exit-code").write_text("0\n")
        (root / "run-ca.sha256").write_text("c" * 64 + "\n")
        (root / "llama-version.txt").write_text(REAL_LLAMA_VERSION)
    (mac / "ufw-before.txt").write_text(UFW_BEFORE)
    (mac / "ufw-during.txt").write_text(UFW_DURING)
    (mac / "ufw-after.txt").write_text(UFW_BEFORE)
    (mac / "mac-start.utc").write_text("2026-11-21T10:00:00.000000Z\n")
    (mac / "mac-end.utc").write_text("2026-11-21T10:00:10.000000Z\n")
    (spark / "shared-start.utc").write_text("2026-11-21T10:01:00.000000Z\n")
    (spark / "shared-end.utc").write_text("2026-11-21T10:01:10.000000Z\n")
    (mac / "gpu.tsv").write_text(
        "2026-11-21T10:00:01.000000Z 100 1\n2026-11-21T10:00:05.000000Z 900 9\n"
        "2026-11-21T10:01:01.000000Z 1000 10\n2026-11-21T10:01:05.000000Z 2000 20\n"
    )
    (spark / "shared-engine.log").write_text(ENGINE_LOG)
    (spark / "shared.pid").write_text("4242\n")
    (spark / "gpu-mem.csv").write_text("2026/11/21 10:01:02.000, 4242, 700\n")
    (spark / "shared-rpc-before.txt").write_text("cubic bytes_received:100 segs_out:3\n")
    (spark / "shared-rpc-after.txt").write_text("cubic bytes_received:900 segs_out:9\n")
    return mac, spark


def test_verify_passes_a_complete_receipt_pair(tmp_path):
    result = load_verifier().verify(*write_receipts(tmp_path))
    assert result["exact_token_parity"] is True
    assert result["firewall_restored"] is True


def test_verify_rejects_firewall_evidence_that_proves_nothing(tmp_path):
    verifier = load_verifier()
    mac, spark = write_receipts(tmp_path)
    leaked = UFW_BEFORE + "[ 2] 24861/tcp                  ALLOW IN    Anywhere\n"
    cases = [
        ({"before": "", "after": ""}, "not active"),
        ({"before": "Status: inactive\n", "after": "Status: inactive\n"}, "not active"),
        ({"before": leaked, "after": leaked}, "24861 existed"),
        ({"during": UFW_BEFORE}, "proof rule"),
        ({"during": UFW_DURING.replace("192.0.2.11", "Anywhere")}, "proof rule"),
        ({"after": UFW_DURING}, "not restored"),
    ]
    for files, message in cases:
        for stage, text in {"before": UFW_BEFORE, "during": UFW_DURING, "after": UFW_BEFORE, **files}.items():
            (mac / f"ufw-{stage}.txt").write_text(text)
        expect_value_error(verifier.verify, mac, spark, match=message)
    (mac / "ufw-during.txt").unlink()
    expect_value_error(verifier.verify, mac, spark, match="missing ufw-during.txt")
