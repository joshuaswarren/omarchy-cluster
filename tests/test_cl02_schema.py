"""Schema drift guard for the committed ``cl02-baseline/1`` example (SUP-190).

``schemas/cl02-baseline-1.example.json`` is the pinned reference record that
every consumer of the CL02 baseline row is written against. This file fails
if either the example or the emitter drifts away from the pinned schema:
the example must validate, its key sets must equal the constants in
``exo_pair``, and a freshly emitted dry-run record must carry the same
keys at every level.
"""

from __future__ import annotations

import json
from pathlib import Path

from omarchy_cluster import baseline_solo, exo_pair

EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "cl02-baseline-1.example.json"
)

# The eight measured metrics the milestone-1 baseline row exists to carry.
# The remaining four metrics keys (token_digest, generated_tokens, errors,
# power_source) are bookkeeping, not measurements.
NUMERIC_METRICS = [
    "ttft_s",
    "decode_tok_s",
    "total_s",
    "ram_peak_mb",
    "vram_peak_mb",
    "lan_mbps",
    "energy_j",
    "power_avg_w",
]
BOOKKEEPING_METRICS = [
    "errors",
    "generated_tokens",
    "power_source",
    "token_digest",
]


def _load_example() -> dict:
    return json.loads(EXAMPLE_PATH.read_text())


def _key_shape(record: dict) -> dict[str, set[str]]:
    """Map each nested-object field to its key set, for cross-record comparison."""
    return {
        "top": set(record),
        "host": set(record["host"]),
        "engine": set(record["engine"]),
        "model": set(record["model"]),
        "prompt": set(record["prompt"]),
        "sampling": set(record["sampling"]),
        "metrics": set(record["metrics"]),
    }


def test_example_file_is_a_valid_cl02_record() -> None:
    record = _load_example()
    assert exo_pair.validate_cl02_record(record) == []


def test_example_pins_schema_version_and_key_sets() -> None:
    record = _load_example()
    assert record["schema"] == exo_pair.SCHEMA_VERSION == "cl02-baseline/1"
    assert set(record) == exo_pair.TOP_LEVEL_KEYS
    assert set(record["host"]) == exo_pair.HOST_KEYS
    assert set(record["engine"]) == exo_pair.ENGINE_KEYS
    assert set(record["model"]) == exo_pair.MODEL_KEYS
    assert set(record["prompt"]) == exo_pair.PROMPT_KEYS
    assert set(record["sampling"]) == exo_pair.SAMPLING_KEYS
    assert set(record["metrics"]) == exo_pair.METRICS_KEYS


def test_metrics_block_carries_the_eight_measurements() -> None:
    metrics = _load_example()["metrics"]
    assert sorted(NUMERIC_METRICS + BOOKKEEPING_METRICS) == sorted(metrics)
    for key in NUMERIC_METRICS:
        assert key in metrics


def test_emitted_dry_run_matches_example_shape(tmp_path) -> None:
    rc = baseline_solo.main(
        [
            "--dry-run",
            "--model",
            "Llama-3.1-8B-Q4",
            "--prompt-tokens",
            "8192",
            "--seed",
            "42",
            "--power-source",
            "none",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    emitted = json.loads((tmp_path / "run.json").read_text())
    assert exo_pair.validate_cl02_record(emitted) == []
    assert _key_shape(emitted) == _key_shape(_load_example())
