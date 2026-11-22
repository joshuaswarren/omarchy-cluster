import json

import pytest

from omarchy_cluster import baseline_solo


def test_solo_argv_forces_one_node():
    assert baseline_solo.solo_argv(["--dry-run"]) == ["--min-nodes", "1", "--dry-run"]
    assert baseline_solo.solo_argv(["--min-nodes", "1"]) == ["--min-nodes", "1"]
    with pytest.raises(SystemExit):
        baseline_solo.solo_argv(["--min-nodes", "2"])


def test_cpu_dry_run_writes_a_cl02_run(tmp_path):
    rc = baseline_solo.main([
        "--dry-run", "--model", "Llama-3.1-8B-Q4", "--prompt-tokens", "8192",
        "--seed", "42", "--power-source", "none", "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["schema"] == "cl02-baseline/1"
    assert run["dry_run"] is True
    for key in ("token_digest", "ttft_s", "decode_tok_s", "total_s", "ram_peak_mb", "errors", "lan_mbps", "power_source"):
        assert key in run["metrics"]
