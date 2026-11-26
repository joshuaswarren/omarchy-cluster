"""Unit and loopback dry-run tests for the Milestone 1 EXO pair runner (CL03)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from omarchy_cluster import exo_pair


def test_deterministic_prompt_reproducible_and_sized() -> None:
    text_a, sha_a = exo_pair.build_deterministic_prompt(8192, seed=42)
    text_b, sha_b = exo_pair.build_deterministic_prompt(8192, seed=42)
    assert text_a == text_b
    assert sha_a == sha_b
    assert exo_pair.HEX16_RE.fullmatch(sha_a)
    assert len(text_a.split()) >= 8190

    _, sha_other = exo_pair.build_deterministic_prompt(8192, seed=7)
    assert sha_other != sha_a


def test_compute_token_digest_requires_int_ids() -> None:
    d1 = exo_pair.compute_token_digest([10, 20, 30])
    d2 = exo_pair.compute_token_digest([10, 20, 30])
    d3 = exo_pair.compute_token_digest([10, 30, 20])
    assert d1 == d2
    assert d1 != d3
    assert exo_pair.HEX16_RE.fullmatch(d1)
    with pytest.raises(TypeError, match="integers"):
        exo_pair.compute_token_digest(["not", "ints"])  # type: ignore[list-item]


def test_select_pair_placement_requires_min_nodes() -> None:
    single_node = [
        {
            "sharding": "Pipeline",
            "instance_meta": "MlxRing",
            "error": None,
            "instance": {
                "MlxRingInstance": {
                    "instanceId": "solo-1",
                    "shardAssignments": {"nodeToRunner": {"node-a": "r-a"}},
                }
            },
        }
    ]
    with pytest.raises(exo_pair.ExoRunnerError, match=">= 2 nodes"):
        exo_pair.select_pair_placement(single_node, min_nodes=2)

    two_nodes = [
        *single_node,
        {
            "sharding": "Pipeline",
            "instance_meta": "MlxRing",
            "error": None,
            "instance": {
                "MlxRingInstance": {
                    "instanceId": "pair-1",
                    "shardAssignments": {
                        "nodeToRunner": {"node-a": "r-a", "node-b": "r-b"}
                    },
                }
            },
        },
        {
            "sharding": "Tensor",
            "instance_meta": "MlxJaccl",
            "error": None,
            "instance": {
                "MlxJacclInstance": {
                    "instanceId": "pair-2",
                    "shardAssignments": {
                        "nodeToRunner": {"node-a": "r-a", "node-b": "r-b"}
                    },
                }
            },
        },
    ]
    inst, nodes, sharding, meta = exo_pair.select_pair_placement(
        two_nodes, min_nodes=2, sharding="Pipeline", instance_meta="MlxRing"
    )
    assert nodes == ["node-a", "node-b"]
    assert sharding == "Pipeline"
    assert meta == "MlxRing"
    assert exo_pair._extract_instance_id(inst) == "pair-1"


def test_verify_exo_checkout_enforces_pin_and_clean_tree(tmp_path: Path) -> None:
    repo = tmp_path / "exo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    email = (
        subprocess.run(
            ["git", "config", "--get", "user.email"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "816217+joshuaswarren@users.noreply.github.com"
    )
    name = (
        subprocess.run(
            ["git", "config", "--get", "user.name"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "Joshua Warren"
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.email", email], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", name], check=True)
    (repo / "README.md").write_text("exo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    assert exo_pair.verify_exo_checkout(repo, expected_commit=sha) == sha

    with pytest.raises(exo_pair.ExoRunnerError, match="commit mismatch"):
        exo_pair.verify_exo_checkout(repo, expected_commit=exo_pair.EXO_PINNED_COMMIT)

    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(exo_pair.ExoRunnerError, match="uncommitted changes"):
        exo_pair.verify_exo_checkout(repo, expected_commit=sha)

    assert exo_pair.verify_exo_checkout(repo, expected_commit=sha, allow_dirty=True) == sha


def test_integrate_power_samples_trapezoidal() -> None:
    energy_j, avg_w = exo_pair.integrate_power_samples(
        [(0.0, 50.0), (1.0, 70.0), (2.0, 70.0)], t0=0.0, t1=2.0
    )
    assert energy_j == pytest.approx(130.0, rel=1e-4)
    assert avg_w == pytest.approx(65.0, rel=1e-4)


def test_compute_lan_mbps() -> None:
    assert exo_pair.compute_lan_mbps(1000, 1000 + 12_500_000, 1.0) == pytest.approx(100.0)
    assert exo_pair.compute_lan_mbps(None, 100, 1.0) is None
    assert exo_pair.compute_lan_mbps(200, 100, 1.0) is None


def test_validate_cl02_record_rejects_bad_fields() -> None:
    good = {
        "schema": "cl02-baseline/1",
        "run_id": "20261122T043000Z-mac+spark",
        "host": {
            "name": "mac+spark",
            "os": "macOS 26.6.2 + Ubuntu 24.04",
            "chip": "Apple M2 Max + NVIDIA GB10",
            "gpu": "Apple M2 Max GPU + NVIDIA GB10 CUDA",
        },
        "engine": {"name": "exo", "version": exo_pair.EXO_ENGINE_VERSION},
        "model": {
            "id": "Llama-3.1-8B-Q4",
            "revision": exo_pair.PINNED_MODELS["Llama-3.1-8B-Q4"].revision,
            "quant": "4bit",
        },
        "prompt": {"tokens": 8192, "sha256_16": "0123456789abcdef"},
        "sampling": {"temperature": 0, "seed": 42, "max_new_tokens": 32},
        "metrics": {
            "token_digest": "fedcba9876543210",
            "generated_tokens": 32,
            "ttft_s": 1.47,
            "decode_tok_s": 36.47,
            "total_s": 2.32,
            "ram_peak_mb": 4820.0,
            "vram_peak_mb": None,
            "errors": [],
            "lan_mbps": 1200.5,
            "energy_j": None,
            "power_avg_w": None,
            "power_source": "none",
        },
        "dry_run": True,
    }
    assert exo_pair.validate_cl02_record(good) == []

    bad_extra = dict(good, extra_key=1)
    assert any("extra=" in e for e in exo_pair.validate_cl02_record(bad_extra))

    bad_temp = dict(good, sampling={"temperature": 0.7, "seed": 42, "max_new_tokens": 32})
    assert any("temperature" in e for e in exo_pair.validate_cl02_record(bad_temp))

    bad_power = dict(good, metrics=dict(good["metrics"], power_source="invalid"))
    assert any("power_source" in e for e in exo_pair.validate_cl02_record(bad_power))

    bad_none_power = dict(
        good, metrics=dict(good["metrics"], power_source="none", energy_j=10.0, power_avg_w=5.0)
    )
    assert any("power_source is none" in e for e in exo_pair.validate_cl02_record(bad_none_power))

    bad_bool = dict(good, metrics=dict(good["metrics"], generated_tokens=True))
    assert any("generated_tokens" in e for e in exo_pair.validate_cl02_record(bad_bool))


def test_end_to_end_loopback_mock_exo_pair_and_cli(tmp_path: Path) -> None:
    out_dir = tmp_path / "dry-run"
    rc = exo_pair.main(
        [
            "--dry-run",
            "--model",
            "Llama-3.1-8B-Q4",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    run_json = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    details_json = json.loads((out_dir / "exo-pair-details.json").read_text(encoding="utf-8"))

    assert exo_pair.validate_cl02_record(run_json) == []
    assert run_json["schema"] == "cl02-baseline/1"
    assert run_json["dry_run"] is True
    assert run_json["engine"] == {"name": "exo", "version": exo_pair.EXO_ENGINE_VERSION}
    assert run_json["model"]["id"] == "Llama-3.1-8B-Q4"
    assert run_json["prompt"]["tokens"] == 8192
    assert run_json["sampling"] == {"temperature": 0, "seed": 42, "max_new_tokens": 32}

    m = run_json["metrics"]
    assert m["errors"] == []
    assert m["generated_tokens"] == 32
    assert m["token_digest"] == exo_pair.compute_token_digest(list(range(1000, 1032)))
    assert m["ttft_s"] > 0.0
    assert m["decode_tok_s"] > 0.0
    assert m["total_s"] >= m["ttft_s"]
    # Blocker 2: ram_peak_mb must reflect generation_stats.peak_memory_usage (4820 MiB),
    # NOT host-wide aggregate nodeMemory (7168 MiB), and vram_peak_mb must be null without
    # in-request device peak sampling (NOT the post-run 4200 MiB nodeMemory snapshot).
    assert m["ram_peak_mb"] == pytest.approx(4820.0)
    assert m["vram_peak_mb"] is None
    assert m["lan_mbps"] is not None and m["lan_mbps"] > 0.0
    assert m["energy_j"] is None
    assert m["power_avg_w"] is None
    assert m["power_source"] == "none"

    assert details_json["exo_pin"]["commit"] == exo_pair.EXO_PINNED_COMMIT
    assert details_json["placement"]["node_count"] == 2
    assert details_json["placement"]["sharding"] == "Pipeline"
    assert details_json["placement"]["instance_meta"] == "MlxRing"
    assert details_json["token_digest_basis"] == "stream_token_ids"
    assert details_json["exo_node_system_power"]["source"] == "exo:/state/nodeSystem:sysPower"


def test_refuses_text_fallback_when_exo_stream_omits_token_ids() -> None:
    """Blocker 1: stock EXO emits text without token_id; runner must refuse text substitution."""
    with exo_pair.MockExoServer(prompt_tokens=8192, emit_token_ids=False) as mock:
        record, details = exo_pair.run_exo_pair(
            base_url=mock.base_url,
            model_key="Llama-3.1-8B-Q4",
            dry_run=False,
        )
        assert mock.deleted_instances == ["dryrun-pair-inst-001"]

    assert exo_pair.validate_cl02_record(record) == []
    assert record["metrics"]["generated_tokens"] == 32
    assert record["metrics"]["token_digest"] == ""
    assert any("token_id" in err for err in record["metrics"]["errors"])
    assert details["token_digest_basis"] == "unavailable"
    assert details["output_text_sha256_16"] is not None


def test_token_id_cmd_resolves_integer_ids_when_stream_omits_them(tmp_path: Path) -> None:
    """Blocker 1: explicit --token-id-cmd supplies integer token IDs for stock EXO."""
    helper = tmp_path / "tok.py"
    helper.write_text(
        "import json, sys\nwords = sys.stdin.read().split()\nprint(json.dumps(list(range(500, 500 + len(words)))))\n",
        encoding="utf-8",
    )
    with exo_pair.MockExoServer(prompt_tokens=8192, emit_token_ids=False) as mock:
        record, details = exo_pair.run_exo_pair(
            base_url=mock.base_url,
            model_key="Llama-3.1-8B-Q4",
            token_id_cmd=f"{sys.executable} {helper}",
            dry_run=False,
        )

    assert exo_pair.validate_cl02_record(record) == []
    assert record["metrics"]["errors"] == []
    assert record["metrics"]["generated_tokens"] == 32
    assert record["metrics"]["token_digest"] == exo_pair.compute_token_digest(list(range(500, 532)))
    assert details["token_digest_basis"] == "token_id_cmd"


@pytest.mark.parametrize("await_behavior", ["timeout", "http_error"])
def test_created_instance_deleted_when_await_fails(await_behavior: str) -> None:
    """Blocker 3: failed/timed-out /instance/await must still delete the created instance."""
    with exo_pair.MockExoServer(prompt_tokens=8192, await_behavior=await_behavior) as mock:
        with pytest.raises(exo_pair.ExoRunnerError):
            exo_pair.run_exo_pair(
                base_url=mock.base_url,
                model_key="Llama-3.1-8B-Q4",
                dry_run=False,
            )
        assert mock.deleted_instances == ["dryrun-pair-inst-001"]
        assert mock.created_instances == {}


def test_host_power_probe_provenance_honest_for_nvidia_smi_and_missing_probe() -> None:
    """Non-blocking + Blocker 2: real nvidia-smi probe yields power + peak VRAM; missing probe stays none."""
    readings = iter([(60.0, 3900.0), (80.0, 4350.0), (70.0, 4100.0)])

    def fake_nvidia() -> tuple[float | None, float | None]:
        return next(readings, (70.0, 4100.0))

    with exo_pair.MockExoServer(prompt_tokens=8192, prefill_delay_s=0.12, token_delay_s=0.005) as mock:
        record, details = exo_pair.run_exo_pair(
            base_url=mock.base_url,
            model_key="Llama-3.1-8B-Q4",
            power_source="nvidia-smi",
            nvidia_probe=fake_nvidia,
            dry_run=False,
        )

    assert exo_pair.validate_cl02_record(record) == []
    assert record["metrics"]["power_source"] == "nvidia-smi"
    assert record["metrics"]["power_avg_w"] is not None and 55.0 <= record["metrics"]["power_avg_w"] <= 85.0
    assert record["metrics"]["energy_j"] is not None and record["metrics"]["energy_j"] > 0.0
    assert record["metrics"]["vram_peak_mb"] == pytest.approx(4350.0)
    assert record["metrics"]["ram_peak_mb"] == pytest.approx(4820.0)
    assert details["host_power_probe"]["effective_source"] == "nvidia-smi"

    # If the requested host probe yields no samples, power_source must fall back to "none"
    # and NOT mislabel EXO's /state/nodeSystem readings as powermetrics.
    with exo_pair.MockExoServer(prompt_tokens=8192, prefill_delay_s=0.05, token_delay_s=0.002) as mock:
        record2, details2 = exo_pair.run_exo_pair(
            base_url=mock.base_url,
            model_key="Llama-3.1-8B-Q4",
            power_source="powermetrics",
            powermetrics_probe=lambda: None,
            dry_run=False,
        )

    assert exo_pair.validate_cl02_record(record2) == []
    assert record2["metrics"]["power_source"] == "none"
    assert record2["metrics"]["energy_j"] is None
    assert record2["metrics"]["power_avg_w"] is None
    assert details2["exo_node_system_power"]["sample_count"] >= 1


def test_wrong_token_id_digest_fails_pair_run(tmp_path: Path) -> None:
    """CLX-3: a wrong strict token-ID digest fails the pair run; the right one
    passes, and pass/fail verdicts plus digest hex are identical across two runs."""
    canonical = exo_pair.compute_token_digest(list(range(500, 532)))
    wrong_canonical = exo_pair.compute_token_digest(list(range(900, 932)))
    right = tmp_path / "right_ids.py"
    right.write_text(
        "import json, sys\nwords = sys.stdin.read().split()\n"
        "print(json.dumps(list(range(500, 500 + len(words)))))\n",
        encoding="utf-8",
    )
    wrong = tmp_path / "wrong_ids.py"
    wrong.write_text(
        "import json, sys\nwords = sys.stdin.read().split()\n"
        "print(json.dumps(list(range(900, 900 + len(words)))))\n",
        encoding="utf-8",
    )

    def run(cmd: Path) -> tuple[dict, dict]:
        with exo_pair.MockExoServer(prompt_tokens=8192, emit_token_ids=False) as mock:
            return exo_pair.run_exo_pair(
                base_url=mock.base_url,
                model_key="Llama-3.1-8B-Q4",
                token_id_cmd=f"{sys.executable} {cmd}",
                expected_token_digest=canonical,
                dry_run=False,
            )

    bad_1, bad_details_1 = run(wrong)
    bad_2, bad_details_2 = run(wrong)
    good_1, good_details_1 = run(right)
    good_2, good_details_2 = run(right)

    expected_err = f"token_digest mismatch: expected {canonical}, got {wrong_canonical}"
    for record, details in ((bad_1, bad_details_1), (bad_2, bad_details_2)):
        assert exo_pair.validate_cl02_record(record) == []
        assert record["metrics"]["generated_tokens"] == 32
        assert record["metrics"]["token_digest"] == wrong_canonical
        assert record["metrics"]["token_digest"] != canonical
        assert record["metrics"]["errors"] == [expected_err]
        assert details["token_digest_basis"] == "token_id_cmd"
        assert details["expected_token_digest"] == canonical
        assert details["token_digest_match"] is False

    for record, details in ((good_1, good_details_1), (good_2, good_details_2)):
        assert exo_pair.validate_cl02_record(record) == []
        assert record["metrics"]["generated_tokens"] == 32
        assert record["metrics"]["token_digest"] == canonical
        assert record["metrics"]["errors"] == []
        assert details["token_digest_basis"] == "token_id_cmd"
        assert details["expected_token_digest"] == canonical
        assert details["token_digest_match"] is True

    assert (bad_1["metrics"]["errors"], bad_1["metrics"]["token_digest"]) == (
        bad_2["metrics"]["errors"],
        bad_2["metrics"]["token_digest"],
    )
    assert (good_1["metrics"]["errors"], good_1["metrics"]["token_digest"]) == (
        good_2["metrics"]["errors"],
        good_2["metrics"]["token_digest"],
    )


def test_expected_token_digest_cli_pass_and_fail_verdicts(tmp_path: Path) -> None:
    """CLX-3: CLI --expected-token-digest returns 1 on mismatch, 0 on match,
    identically across two runs on the canned loopback fixture."""
    canonical = exo_pair.compute_token_digest(list(range(1000, 1032)))
    wrong_expected = "0" * 16

    def invoke(expected: str, name: str) -> tuple[int, dict, dict]:
        out_dir = tmp_path / name
        rc = exo_pair.main(
            [
                "--dry-run",
                "--model",
                "Llama-3.1-8B-Q4",
                "--expected-token-digest",
                expected,
                "--out-dir",
                str(out_dir),
            ]
        )
        run_json = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        details_json = json.loads((out_dir / "exo-pair-details.json").read_text(encoding="utf-8"))
        return rc, run_json, details_json

    fail_1 = invoke(wrong_expected, "fail-1")
    fail_2 = invoke(wrong_expected, "fail-2")
    pass_1 = invoke(canonical, "pass-1")
    pass_2 = invoke(canonical, "pass-2")

    for rc, record, details in (fail_1, fail_2):
        assert rc == 1
        assert exo_pair.validate_cl02_record(record) == []
        assert record["metrics"]["token_digest"] == canonical
        assert record["metrics"]["errors"] == [
            f"token_digest mismatch: expected {wrong_expected}, got {canonical}"
        ]
        assert details["expected_token_digest"] == wrong_expected
        assert details["token_digest_match"] is False

    for rc, record, details in (pass_1, pass_2):
        assert rc == 0
        assert exo_pair.validate_cl02_record(record) == []
        assert record["metrics"]["token_digest"] == canonical
        assert record["metrics"]["errors"] == []
        assert details["expected_token_digest"] == canonical
        assert details["token_digest_match"] is True

    assert (fail_1[0], fail_1[1]["metrics"]["token_digest"]) == (
        fail_2[0],
        fail_2[1]["metrics"]["token_digest"],
    )
    assert (pass_1[0], pass_1[1]["metrics"]["token_digest"]) == (
        pass_2[0],
        pass_2[1]["metrics"]["token_digest"],
    )

    with pytest.raises(SystemExit) as exc_info:
        exo_pair.main(["--dry-run", "--expected-token-digest", "not-a-hex-digest"])
    assert exc_info.value.code == 2

    with pytest.raises(ValueError, match="16 lowercase hex chars"):
        exo_pair.run_exo_pair(
            base_url="http://127.0.0.1:52415",
            expected_token_digest="INVALID",
        )


def test_malformed_token_id_cmd_fails_pair_run(tmp_path: Path) -> None:
    """CLX-3: a token-id-cmd that does not emit an int array fails the run:
    errors recorded, empty digest, no text substitution."""
    canonical = exo_pair.compute_token_digest(list(range(500, 532)))
    broken = tmp_path / "broken_ids.py"
    broken.write_text("print('not json')\n", encoding="utf-8")
    with exo_pair.MockExoServer(prompt_tokens=8192, emit_token_ids=False) as mock:
        record, details = exo_pair.run_exo_pair(
            base_url=mock.base_url,
            model_key="Llama-3.1-8B-Q4",
            token_id_cmd=f"{sys.executable} {broken}",
            expected_token_digest=canonical,
            dry_run=False,
        )
    assert exo_pair.validate_cl02_record(record) == []
    assert any("valid JSON" in err for err in record["metrics"]["errors"])
    assert any(
        f"token_digest mismatch: expected {canonical}, got unavailable" in err
        for err in record["metrics"]["errors"]
    )
    assert record["metrics"]["token_digest"] == ""
    assert details["token_digest_basis"] == "unavailable"
    assert details["expected_token_digest"] == canonical
    assert details["token_digest_match"] is False
