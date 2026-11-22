"""Unit and loopback dry-run tests for the Milestone 1 EXO pair runner (CL03)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from omarchy_cluster import exo_pair


def test_deterministic_prompt_reproducible_and_sized() -> None:
    text_a, sha_a = exo_pair.build_deterministic_prompt(8192, seed=42)
    text_b, sha_b = exo_pair.build_deterministic_prompt(8192, seed=42)
    assert text_a == text_b
    assert sha_a == sha_b
    assert exo_pair.HEX16_RE.fullmatch(sha_a)
    # 8,192 tokens = 8,184 body words + 8-token instruction suffix
    assert len(text_a.split()) >= 8190

    _, sha_other = exo_pair.build_deterministic_prompt(8192, seed=7)
    assert sha_other != sha_a


def test_compute_token_digest_matches_ordered_ids() -> None:
    d1 = exo_pair.compute_token_digest([10, 20, 30])
    d2 = exo_pair.compute_token_digest([10, 20, 30])
    d3 = exo_pair.compute_token_digest([10, 30, 20])
    assert d1 == d2
    assert d1 != d3
    assert exo_pair.HEX16_RE.fullmatch(d1)


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

    two_nodes = single_node + [
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
    # 0..1s avg 60W -> 60J; 1..2s avg 70W -> 70J; total 130J, avg 65W
    assert energy_j == pytest.approx(130.0, rel=1e-4)
    assert avg_w == pytest.approx(65.0, rel=1e-4)


def test_compute_lan_mbps() -> None:
    # 12.5 MB over 1.0 s = 100.0 Mbit/s
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
            "vram_peak_mb": 4200.0,
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
    assert m["ram_peak_mb"] > 0.0
    assert m["vram_peak_mb"] == pytest.approx(4200.0)
    assert m["lan_mbps"] is not None and m["lan_mbps"] > 0.0
    assert m["power_source"] == "none"

    assert details_json["exo_pin"]["commit"] == exo_pair.EXO_PINNED_COMMIT
    assert details_json["placement"]["node_count"] == 2
    assert details_json["placement"]["sharding"] == "Pipeline"
    assert details_json["placement"]["instance_meta"] == "MlxRing"
    assert details_json["token_digest_basis"] == "token_ids"


def test_loopback_server_power_integration_live_mode() -> None:
    with exo_pair.MockExoServer(prompt_tokens=8192, prefill_delay_s=0.08, token_delay_s=0.005) as mock:
        record, details = exo_pair.run_exo_pair(
            base_url=mock.base_url,
            model_key="Llama-3.1-8B-Q4",
            power_source="powermetrics",
            dry_run=False,
        )
        assert mock.deleted_instances == ["dryrun-pair-inst-001"]
        assert mock.last_request is not None
        assert mock.last_request["temperature"] == 0
        assert mock.last_request["seed"] == 42
        assert mock.last_request["max_tokens"] == 32
        assert mock.last_request["use_prefix_cache"] is False

    assert exo_pair.validate_cl02_record(record) == []
    assert record["dry_run"] is False
    assert record["metrics"]["power_source"] == "powermetrics"
    assert record["metrics"]["power_avg_w"] == pytest.approx(72.5, rel=0.05)
    assert record["metrics"]["energy_j"] is not None and record["metrics"]["energy_j"] > 0.0
    assert details["power_sample_count"] >= 1
