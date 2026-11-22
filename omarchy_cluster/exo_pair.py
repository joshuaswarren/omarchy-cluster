"""Milestone 1 EXO pair (macOS + DGX OS) benchmark runner (CL03).

Pins upstream exo-explore/exo, verifies two-node macOS + DGX OS participation,
executes the 8,192-token EXO comparison workload shape (Llama-3.1-8B-Q4,
8,192-token prompt, greedy decode), and emits CL02's exact 8-metric JSON schema
(``cl02-baseline/1``).

Pure Python 3.9+ stdlib; no third-party runtime dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import re
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

SCHEMA_VERSION = "cl02-baseline/1"

EXO_REPO_URL = "https://github.com/exo-explore/exo"
EXO_PINNED_COMMIT = "a60b95698ca81737d367443674dd5563cbc9559b"
EXO_PINNED_VERSION = "0.3.70"
EXO_ENGINE_VERSION = f"{EXO_PINNED_VERSION}+{EXO_PINNED_COMMIT[:12]}"
EXO_DEFAULT_PORT = 52415
EXO_DEFAULT_NAMESPACE = "omarchy-m1-exo"

# Published author-reported reference shape from docs/heterogeneous-inference-research.md
# (Llama-3.1-8B, 8,192-token prompt, 32 generated tokens; author-reported, not reproduced here).
EXO_REFERENCE_SHAPE = {
    "model": "Llama-3.1-8B",
    "prompt_tokens": 8192,
    "generated_tokens": 32,
    "spark_solo_s": {"prefill_s": 1.47, "generation_s": 2.87, "total_s": 4.34},
    "mac_solo_s": {"prefill_s": 5.57, "generation_s": 0.85, "total_s": 6.42},
    "combined_s": {"prefill_s": 1.47, "generation_s": 0.85, "total_s": 2.32},
}

ALLOWED_ENGINES = frozenset({"mlx-lm", "llama.cpp", "exo"})
ALLOWED_POWER_SOURCES = frozenset({"powermetrics", "nvidia-smi", "none"})
HEX16_RE = re.compile(r"^[0-9a-f]{16}$")

TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "host",
        "engine",
        "model",
        "prompt",
        "sampling",
        "metrics",
        "dry_run",
    }
)
HOST_KEYS = frozenset({"name", "os", "chip", "gpu"})
ENGINE_KEYS = frozenset({"name", "version"})
MODEL_KEYS = frozenset({"id", "revision", "quant"})
PROMPT_KEYS = frozenset({"tokens", "sha256_16"})
SAMPLING_KEYS = frozenset({"temperature", "seed", "max_new_tokens"})
METRICS_KEYS = frozenset(
    {
        "token_digest",
        "generated_tokens",
        "ttft_s",
        "decode_tok_s",
        "total_s",
        "ram_peak_mb",
        "vram_peak_mb",
        "errors",
        "lan_mbps",
        "energy_j",
        "power_avg_w",
        "power_source",
    }
)


@dataclass(frozen=True)
class ModelPin:
    """Pinned model specification for Milestone 1 baseline runs."""

    model_id: str
    exo_model_id: str
    revision: str
    quant: str
    default_prompt_tokens: int
    default_max_new_tokens: int


PINNED_MODELS: dict[str, ModelPin] = {
    "Llama-3.1-8B-Q4": ModelPin(
        model_id="Llama-3.1-8B-Q4",
        exo_model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        revision="241a666dad6cb93c8ff213d39a7f34a36bf26db4",
        quant="4bit",
        default_prompt_tokens=8192,
        default_max_new_tokens=32,
    ),
    "Llama-3.1-8B-BF16": ModelPin(
        model_id="Llama-3.1-8B-BF16",
        exo_model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        revision="f8311090f9ee47782b6f094984a20c856eb841d6",
        quant="bf16",
        default_prompt_tokens=8192,
        default_max_new_tokens=32,
    ),
    "Qwen3-1.7B-Q4": ModelPin(
        model_id="Qwen3-1.7B-Q4",
        exo_model_id="mlx-community/Qwen3-1.7B-4bit",
        revision="3b1b1768f8f8cf8351c712464f906e86c2b8269e",
        quant="4bit",
        default_prompt_tokens=43,
        default_max_new_tokens=64,
    ),
}

# Deterministic single-token English words used to build a reproducible
# prompt calibrated to target_tokens without requiring external tokenizers.
_PROMPT_VOCAB = (
    "alpha",
    "anchor",
    "apple",
    "atlas",
    "badge",
    "beacon",
    "bench",
    "block",
    "bridge",
    "buffer",
    "cable",
    "cache",
    "canal",
    "cargo",
    "chain",
    "clock",
    "cloud",
    "cluster",
    "codec",
    "copper",
    "coral",
    "delta",
    "device",
    "driver",
    "ember",
    "engine",
    "epoch",
    "fiber",
    "field",
    "frame",
    "frost",
    "gauge",
    "grain",
    "graph",
    "harbor",
    "index",
    "input",
    "kernel",
    "layer",
    "ledger",
    "level",
    "limit",
    "linear",
    "logic",
    "matrix",
    "memory",
    "metric",
    "model",
    "module",
    "network",
    "node",
    "offset",
    "orbit",
    "packet",
    "panel",
    "parity",
    "phase",
    "pilot",
    "pixel",
    "plane",
    "queue",
    "ratio",
    "ridge",
    "route",
)

_PROMPT_SUFFIX = "Summarize the sequence in one short sentence."
_PROMPT_SUFFIX_TOKENS = 8


class ExoRunnerError(RuntimeError):
    """Raised when EXO setup, placement, or execution fails."""


class ExoHttpError(ExoRunnerError):
    """Raised on HTTP >= 400 from the EXO REST API."""

    def __init__(self, status: int, reason: str, body: str) -> None:
        super().__init__(f"EXO HTTP {status} {reason}: {body[:300]}")
        self.status = status
        self.reason = reason
        self.body = body


def _is_real_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_cl02_record(record: Mapping[str, Any]) -> list[str]:
    """Validate a run record against ``cl02-baseline/1``.

    Returns a list of validation error strings (empty list == valid).
    """
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["record must be a JSON object"]

    keys = set(record.keys())
    if keys != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - keys)
        extra = sorted(keys - TOP_LEVEL_KEYS)
        errors.append(f"top-level key mismatch: missing={missing} extra={extra}")

    if record.get("schema") != SCHEMA_VERSION:
        errors.append(f"schema must be {SCHEMA_VERSION!r}, got {record.get('schema')!r}")

    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("run_id must be a non-empty string")

    if not isinstance(record.get("dry_run"), bool):
        errors.append("dry_run must be a boolean")

    host = record.get("host")
    if not isinstance(host, Mapping) or set(host.keys()) != HOST_KEYS:
        errors.append("host must have exact keys: name, os, chip, gpu")
    else:
        for k in ("name", "os", "chip"):
            if not isinstance(host.get(k), str) or not host[k].strip():
                errors.append(f"host.{k} must be a non-empty string")
        if host.get("gpu") is not None and not isinstance(host.get("gpu"), str):
            errors.append("host.gpu must be string or null")

    engine = record.get("engine")
    if not isinstance(engine, Mapping) or set(engine.keys()) != ENGINE_KEYS:
        errors.append("engine must have exact keys: name, version")
    else:
        if engine.get("name") not in ALLOWED_ENGINES:
            errors.append(f"engine.name must be one of {sorted(ALLOWED_ENGINES)}")
        if not isinstance(engine.get("version"), str) or not engine["version"].strip():
            errors.append("engine.version must be a non-empty string")

    model = record.get("model")
    if not isinstance(model, Mapping) or set(model.keys()) != MODEL_KEYS:
        errors.append("model must have exact keys: id, revision, quant")
    else:
        for k in ("id", "revision", "quant"):
            if not isinstance(model.get(k), str) or not model[k].strip():
                errors.append(f"model.{k} must be a non-empty string")

    prompt = record.get("prompt")
    if not isinstance(prompt, Mapping) or set(prompt.keys()) != PROMPT_KEYS:
        errors.append("prompt must have exact keys: tokens, sha256_16")
    else:
        if not _is_int(prompt.get("tokens")) or prompt["tokens"] <= 0:
            errors.append("prompt.tokens must be a positive integer")
        if not isinstance(prompt.get("sha256_16"), str) or not HEX16_RE.fullmatch(
            prompt["sha256_16"]
        ):
            errors.append("prompt.sha256_16 must be 16 lowercase hex chars")

    sampling = record.get("sampling")
    if not isinstance(sampling, Mapping) or set(sampling.keys()) != SAMPLING_KEYS:
        errors.append("sampling must have exact keys: temperature, seed, max_new_tokens")
    else:
        if not _is_real_number(sampling.get("temperature")) or sampling["temperature"] != 0:
            errors.append("sampling.temperature must be 0 (greedy)")
        if not _is_int(sampling.get("seed")):
            errors.append("sampling.seed must be an integer")
        if not _is_int(sampling.get("max_new_tokens")) or sampling["max_new_tokens"] <= 0:
            errors.append("sampling.max_new_tokens must be a positive integer")

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics.keys()) != METRICS_KEYS:
        errors.append(f"metrics must have exact keys: {sorted(METRICS_KEYS)}")
    else:
        run_errors = metrics.get("errors")
        if not isinstance(run_errors, list) or not all(isinstance(e, str) for e in run_errors):
            errors.append("metrics.errors must be an array of strings")
            run_errors = []

        gen_tokens = metrics.get("generated_tokens")
        if not _is_int(gen_tokens) or gen_tokens < 0:
            errors.append("metrics.generated_tokens must be a non-negative integer")
            gen_tokens = 0

        digest = metrics.get("token_digest")
        if not isinstance(digest, str):
            errors.append("metrics.token_digest must be a string")
        elif not run_errors:
            # Clean runs must carry a valid 16-hex token-ID digest and positive generated_tokens.
            if gen_tokens <= 0:
                errors.append("metrics.generated_tokens must be > 0 when metrics.errors is empty")
            if not HEX16_RE.fullmatch(digest):
                errors.append("metrics.token_digest must be 16 lowercase hex chars")
        elif digest != "" and not HEX16_RE.fullmatch(digest):
            errors.append("metrics.token_digest must be empty or 16 lowercase hex chars")

        for k in ("ttft_s", "decode_tok_s", "total_s", "ram_peak_mb"):
            v = metrics.get(k)
            if not _is_real_number(v) or float(v) < 0.0:
                errors.append(f"metrics.{k} must be a non-negative finite number")

        for k in ("vram_peak_mb", "lan_mbps", "energy_j", "power_avg_w"):
            v = metrics.get(k)
            if v is not None and (not _is_real_number(v) or float(v) < 0.0):
                errors.append(f"metrics.{k} must be null or a non-negative finite number")

        power_source = metrics.get("power_source")
        if power_source not in ALLOWED_POWER_SOURCES:
            errors.append(
                f"metrics.power_source must be one of {sorted(ALLOWED_POWER_SOURCES)}"
            )
        elif power_source == "none" and (
            metrics.get("energy_j") is not None or metrics.get("power_avg_w") is not None
        ):
            errors.append("metrics.energy_j and power_avg_w must be null when power_source is none")

    return errors


def build_deterministic_prompt(target_tokens: int, seed: int = 42) -> tuple[str, str]:
    """Build a deterministic prompt calibrated to ``target_tokens`` tokens.

    Returns ``(prompt_text, sha256_16)``.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    body_tokens = max(1, target_tokens - _PROMPT_SUFFIX_TOKENS)
    vocab_len = len(_PROMPT_VOCAB)
    words = [_PROMPT_VOCAB[(i * 7 + seed) % vocab_len] for i in range(body_tokens)]
    if target_tokens > _PROMPT_SUFFIX_TOKENS:
        text = " ".join(words) + "\n" + _PROMPT_SUFFIX
    else:
        text = " ".join(words[:target_tokens])
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return text, digest


def compute_token_digest(token_ids: Sequence[int]) -> str:
    """Compute first 16 hex chars of SHA-256 over ordered integer token IDs."""
    ids: list[int] = []
    for item in token_ids:
        if not _is_int(item):
            raise TypeError(f"token_ids must contain integers, got {type(item).__name__}")
        ids.append(item)
    payload = json.dumps(ids, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def resolve_token_ids_via_cmd(cmd: str, generated_text: str) -> list[int]:
    """Run an operator-supplied tokenizer command converting stdin text -> JSON ``list[int]``."""
    argv = shlex.split(cmd)
    if not argv:
        raise ExoRunnerError("Empty --token-id-cmd")
    proc = subprocess.run(
        argv,
        input=generated_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ExoRunnerError(
            f"--token-id-cmd failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}"
        )
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ExoRunnerError(f"--token-id-cmd did not output valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(_is_int(x) for x in parsed):
        raise ExoRunnerError("--token-id-cmd must output a JSON array of integer token IDs")
    return list(parsed)


def verify_exo_checkout(
    repo_dir: Path,
    expected_commit: str = EXO_PINNED_COMMIT,
    *,
    allow_dirty: bool = False,
) -> str:
    """Verify a local EXO git checkout matches ``expected_commit``."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise ExoRunnerError(f"EXO checkout directory does not exist: {repo_dir}")

    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        raise ExoRunnerError(f"git rev-parse HEAD failed in {repo_dir}: {head.stderr.strip()}")
    actual = head.stdout.strip()
    if actual != expected_commit:
        raise ExoRunnerError(
            f"EXO commit mismatch in {repo_dir}: got {actual}, expected {expected_commit}"
        )

    if not allow_dirty:
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0:
            raise ExoRunnerError(f"git status failed in {repo_dir}: {status.stderr.strip()}")
        if status.stdout.strip():
            raise ExoRunnerError(
                f"EXO checkout in {repo_dir} has uncommitted changes; "
                "pass --allow-dirty-exo to override"
            )
    return actual


def read_iface_bytes(iface: str) -> int | None:
    """Read cumulative RX + TX bytes on ``iface`` (Linux sysfs or macOS netstat)."""
    sysfs = Path("/sys/class/net") / iface / "statistics"
    if sysfs.is_dir():
        try:
            rx = int((sysfs / "rx_bytes").read_text().strip())
            tx = int((sysfs / "tx_bytes").read_text().strip())
            return rx + tx
        except (OSError, ValueError):
            return None

    proc = subprocess.run(
        ["netstat", "-ibn"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[0] == iface and "<Link#" in parts[2]:
            try:
                ibytes = int(parts[6])
                obytes = int(parts[9])
                return ibytes + obytes
            except ValueError:
                continue
    return None


def compute_lan_mbps(
    bytes_before: int | None, bytes_after: int | None, elapsed_s: float
) -> float | None:
    """Compute average LAN throughput in Mbit/s from interface byte counters."""
    if bytes_before is None or bytes_after is None or elapsed_s <= 0.0:
        return None
    delta = bytes_after - bytes_before
    if delta < 0:
        return None
    return round((delta * 8.0) / (elapsed_s * 1_000_000.0), 3)


def integrate_power_samples(
    samples: Sequence[tuple[float, float]], t0: float, t1: float
) -> tuple[float | None, float | None]:
    """Trapezoidal integration of ``(timestamp_s, watts)`` over ``[t0, t1]``.

    Returns ``(energy_j, power_avg_w)``.
    """
    elapsed = t1 - t0
    if elapsed <= 0.0 or not samples:
        return None, None
    window = sorted((t, w) for t, w in samples if t0 <= t <= t1 and w >= 0.0)
    if not window:
        nearest = min(samples, key=lambda pair: min(abs(pair[0] - t0), abs(pair[0] - t1)))
        if nearest[1] < 0.0:
            return None, None
        energy_j = nearest[1] * elapsed
        return round(energy_j, 4), round(nearest[1], 3)

    if len(window) == 1:
        watts = window[0][1]
        return round(watts * elapsed, 4), round(watts, 3)

    joules = 0.0
    if window[0][0] > t0:
        joules += window[0][1] * (window[0][0] - t0)
    for idx in range(1, len(window)):
        dt = window[idx][0] - window[idx - 1][0]
        if dt > 0.0:
            joules += 0.5 * (window[idx][1] + window[idx - 1][1]) * dt
    if window[-1][0] < t1:
        joules += window[-1][1] * (t1 - window[-1][0])
    avg_w = joules / elapsed
    return round(joules, 4), round(avg_w, 3)


def _sample_nvidia_smi_once() -> tuple[float | None, float | None]:
    """Sample local ``nvidia-smi`` for ``(power_watts, vram_used_mib)``."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, None
    total_w = 0.0
    total_mib = 0.0
    saw_w = False
    saw_mib = False
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            w = float(parts[0])
            if math.isfinite(w) and w >= 0.0:
                total_w += w
                saw_w = True
        except ValueError:
            pass
        try:
            mib = float(parts[1])
            if math.isfinite(mib) and mib >= 0.0:
                total_mib += mib
                saw_mib = True
        except ValueError:
            pass
    return (total_w if saw_w else None), (total_mib if saw_mib else None)


def _sample_powermetrics_once(sample_ms: int = 200) -> float | None:
    """Sample macOS ``powermetrics --samplers cpu_power,gpu_power`` once for combined watts."""
    proc = subprocess.run(
        [
            "powermetrics",
            "--samplers",
            "cpu_power,gpu_power",
            "-i",
            str(sample_ms),
            "-n",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    match = re.search(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+(?:\.\d+)?)\s*mW", proc.stdout)
    if match:
        return float(match.group(1)) / 1000.0
    gpu = re.search(r"GPU Power:\s*(\d+(?:\.\d+)?)\s*mW", proc.stdout)
    cpu = re.search(r"CPU Power:\s*(\d+(?:\.\d+)?)\s*mW", proc.stdout)
    if gpu or cpu:
        total_mw = (float(gpu.group(1)) if gpu else 0.0) + (float(cpu.group(1)) if cpu else 0.0)
        return total_mw / 1000.0
    return None


class _HardwareProbeSampler:
    """Polls a real host probe (``nvidia-smi`` or ``powermetrics``) during the request window."""

    def __init__(
        self,
        source: str,
        interval_s: float = 0.25,
        *,
        nvidia_probe: Callable[[], tuple[float | None, float | None]] = _sample_nvidia_smi_once,
        powermetrics_probe: Callable[[], float | None] = _sample_powermetrics_once,
    ) -> None:
        self.source = source
        self._interval_s = interval_s
        self._nvidia_probe = nvidia_probe
        self._powermetrics_probe = powermetrics_probe
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.power_samples: list[tuple[float, float]] = []
        self.vram_peak_mib: float | None = None
        self._probe_errors = 0

    def start(self) -> None:
        if self.source == "none":
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.perf_counter()
            try:
                if self.source == "nvidia-smi":
                    watts, vram_mib = self._nvidia_probe()
                    if watts is not None and watts >= 0.0:
                        self.power_samples.append((ts, watts))
                    if (
                        vram_mib is not None
                        and vram_mib >= 0.0
                        and (self.vram_peak_mib is None or vram_mib > self.vram_peak_mib)
                    ):
                        self.vram_peak_mib = round(vram_mib, 2)
                elif self.source == "powermetrics":
                    watts = self._powermetrics_probe()
                    if watts is not None and watts >= 0.0:
                        self.power_samples.append((ts, watts))
            except (OSError, ValueError):
                self._probe_errors += 1
            self._stop.wait(self._interval_s)


class ExoApiClient:
    """Pure-stdlib HTTP/SSE client for the EXO REST API (default port 52415)."""

    def __init__(self, base_url: str, timeout_s: float = 600.0) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Invalid EXO base URL: {base_url!r}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else EXO_DEFAULT_PORT)
        self.base_path = parsed.path.rstrip("/")
        self.timeout_s = timeout_s

    def _connect(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout_s)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)

    def _full_path(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        rel = path if path.startswith("/") else "/" + path
        full = self.base_path + rel
        if params:
            full = full + "?" + urlencode(params)
        return full

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        conn = self._connect()
        try:
            headers = {"Accept": "application/json"}
            payload: bytes | None = None
            if body is not None:
                payload = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
            conn.request(
                method.upper(), self._full_path(path, params), body=payload, headers=headers
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise ExoHttpError(resp.status, resp.reason, raw)
            if not raw.strip():
                return None
            return json.loads(raw)
        finally:
            conn.close()

    def get_state(self) -> dict[str, Any]:
        data = self.request_json("GET", "/state")
        if not isinstance(data, dict):
            raise ExoRunnerError(f"Expected dict from GET /state, got {type(data).__name__}")
        return data

    def get_state_path(self, path: str) -> Any:
        try:
            return self.request_json("GET", f"/state/{path.lstrip('/')}")
        except ExoHttpError as exc:
            if exc.status == 404:
                return None
            raise

    def get_instance_previews(self, model_id: str) -> list[dict[str, Any]]:
        data = self.request_json("GET", "/instance/previews", params={"model_id": model_id})
        if isinstance(data, dict) and isinstance(data.get("previews"), list):
            return data["previews"]
        if isinstance(data, list):
            return data
        raise ExoRunnerError(f"Unexpected /instance/previews response shape: {type(data).__name__}")

    def create_instance(self, instance: Mapping[str, Any]) -> Any:
        return self.request_json("POST", "/instance", body={"instance": dict(instance)})

    def delete_instance(self, instance_id: str) -> None:
        try:
            self.request_json("DELETE", f"/instance/{instance_id}")
        except ExoHttpError as exc:
            if exc.status != 404:
                raise

    def await_instance(self, model_id: str, timeout_s: float = 180.0) -> dict[str, Any]:
        conn = self._connect()
        try:
            path = self._full_path(
                "/instance/await",
                {"model_id": model_id, "timeout_seconds": min(int(timeout_s), 300)},
            )
            conn.request("GET", path, headers={"Accept": "text/event-stream"})
            resp = conn.getresponse()
            if resp.status >= 400:
                raw = resp.read().decode("utf-8", errors="replace")
                raise ExoHttpError(resp.status, resp.reason, raw)
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                event_type = event.get("type")
                if event_type == "ready":
                    return event
                if event_type == "timeout":
                    raise ExoRunnerError(
                        f"EXO /instance/await timed out for {model_id}: {event.get('message', '')}"
                    )
            raise ExoRunnerError(f"EXO /instance/await closed without ready event for {model_id}")
        finally:
            conn.close()

    def stream_bench_chat_completions(self, payload: Mapping[str, Any]) -> Iterator[str]:
        body = dict(payload)
        body["stream"] = True
        encoded = json.dumps(body).encode()
        conn = self._connect()
        try:
            conn.request(
                "POST",
                self._full_path("/bench/chat/completions"),
                body=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            resp = conn.getresponse()
            if resp.status >= 400:
                raw = resp.read().decode("utf-8", errors="replace")
                raise ExoHttpError(resp.status, resp.reason, raw)
            for raw_line in resp:
                yield raw_line.decode("utf-8", errors="replace")
        finally:
            conn.close()


def _extract_instance_node_ids(instance: Mapping[str, Any]) -> list[str]:
    """Extract participating node IDs from an EXO Instance object."""
    inner = instance
    if len(instance) == 1:
        sole_val = next(iter(instance.values()))
        if isinstance(sole_val, Mapping):
            inner = sole_val
    assignments = inner.get("shardAssignments") or inner.get("shard_assignments") or {}
    if isinstance(assignments, Mapping):
        node_to_runner = assignments.get("nodeToRunner") or assignments.get("node_to_runner")
        if isinstance(node_to_runner, Mapping):
            return sorted(str(k) for k in node_to_runner)
    return []


def _extract_instance_id(instance: Mapping[str, Any]) -> str | None:
    if "instanceId" in instance:
        return str(instance["instanceId"])
    if "instance_id" in instance:
        return str(instance["instance_id"])
    if len(instance) == 1:
        sole_val = next(iter(instance.values()))
        if isinstance(sole_val, Mapping):
            val = sole_val.get("instanceId") or sole_val.get("instance_id")
            if val is not None:
                return str(val)
    return None


def select_pair_placement(
    previews: Sequence[Mapping[str, Any]],
    *,
    min_nodes: int = 2,
    sharding: str = "Pipeline",
    instance_meta: str | None = "MlxRing",
) -> tuple[dict[str, Any], list[str], str, str]:
    """Select an EXO placement preview spanning at least ``min_nodes`` nodes.

    Returns ``(instance_dict, node_ids, chosen_sharding, chosen_instance_meta)``.
    """
    candidates: list[tuple[int, dict[str, Any], list[str], str, str]] = []
    for preview in previews:
        if preview.get("error"):
            continue
        inst = preview.get("instance")
        if not isinstance(inst, Mapping):
            continue
        node_ids = _extract_instance_node_ids(inst)
        if len(node_ids) < min_nodes:
            continue
        p_sharding = str(preview.get("sharding") or "")
        p_meta = str(preview.get("instance_meta") or preview.get("instanceMeta") or "")
        score = 0
        if p_sharding.lower() == sharding.lower():
            score += 4
        if instance_meta and p_meta.lower() == instance_meta.lower():
            score += 2
        if len(node_ids) == min_nodes:
            score += 1
        candidates.append((score, dict(inst), node_ids, p_sharding, p_meta))

    if not candidates:
        raise ExoRunnerError(
            f"No valid EXO placement spans >= {min_nodes} nodes "
            f"(requested sharding={sharding}, instance_meta={instance_meta})"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, chosen, node_ids, chosen_sharding, chosen_meta = candidates[0]
    return chosen, node_ids, chosen_sharding, chosen_meta


class _ExoNodeSystemSampler:
    """Polls EXO ``/state/nodeSystem`` for auxiliary cluster telemetry in details JSON."""

    def __init__(self, client: ExoApiClient, node_ids: Sequence[str], interval_s: float = 0.25):
        self._client = client
        self._node_ids = list(node_ids)
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []
        self.per_node_samples: dict[str, list[tuple[float, float]]] = {
            nid: [] for nid in self._node_ids
        }
        self._poll_errors = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.perf_counter()
            try:
                node_sys = self._client.get_state_path("nodeSystem")
                if isinstance(node_sys, Mapping):
                    total_w = 0.0
                    saw_any = False
                    for nid in self._node_ids:
                        entry = node_sys.get(nid)
                        if isinstance(entry, Mapping):
                            power = entry.get("sysPower")
                            if _is_real_number(power) and float(power) > 0.0:
                                w = float(power)
                                total_w += w
                                saw_any = True
                                self.per_node_samples[nid].append((ts, w))
                    if saw_any:
                        self.samples.append((ts, total_w))
            except (ExoRunnerError, OSError, http.client.HTTPException, ValueError):
                self._poll_errors += 1
            self._stop.wait(self._interval_s)


def _summarize_cluster_nodes(
    state: Mapping[str, Any], node_ids: Sequence[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Derive CL02 ``host`` object and per-node summary from EXO ``/state``."""
    identities = state.get("nodeIdentities") or state.get("node_identities") or {}
    memories = state.get("nodeMemory") or state.get("node_memory") or {}
    nodes_detail: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    oses: list[str] = []
    chips: list[str] = []
    gpus: list[str] = []

    for nid in node_ids:
        ident = identities.get(nid, {}) if isinstance(identities, Mapping) else {}
        mem = memories.get(nid, {}) if isinstance(memories, Mapping) else {}
        friendly = str(ident.get("friendlyName") or ident.get("name") or nid)
        os_ver = str(ident.get("osVersion") or ident.get("os") or "unknown-os")
        chip = str(ident.get("chipId") or ident.get("chip") or "unknown-chip")
        gpu = ident.get("gpu") or chip
        names.append(friendly)
        oses.append(os_ver)
        chips.append(chip)
        if gpu:
            gpus.append(str(gpu))
        nodes_detail[nid] = {
            "friendly_name": friendly,
            "os": os_ver,
            "chip": chip,
            "gpu": str(gpu) if gpu else None,
            "memory_snapshot_after": dict(mem) if isinstance(mem, Mapping) else {},
        }

    host_obj = {
        "name": "+".join(names) if names else socket.gethostname(),
        "os": " + ".join(oses) if oses else "unknown",
        "chip": " + ".join(chips) if chips else "unknown",
        "gpu": " + ".join(gpus) if gpus else None,
    }
    return host_obj, nodes_detail


def _extract_memory_peaks_mb(
    gen_stats: Mapping[str, Any] | None,
    *,
    sampled_vram_peak_mb: float | None = None,
) -> tuple[float, float | None]:
    """Return serving-workload ``(ram_peak_mb, vram_peak_mb)`` in MiB.

    ``ram_peak_mb`` comes strictly from ``generation_stats.peak_memory_usage``
    (never mixed with post-run host-wide OS ``nodeMemory`` usage).
    ``vram_peak_mb`` comes strictly from in-request peak device sampling or
    ``generation_stats.peak_vram_usage`` (``None`` if not measured as a peak).
    """
    mib = 1024.0 * 1024.0
    ram_bytes = 0.0
    vram_mb: float | None = None

    if isinstance(gen_stats, Mapping):
        peak = gen_stats.get("peak_memory_usage")
        if isinstance(peak, Mapping):
            raw = peak.get("inBytes") or peak.get("in_bytes")
            if _is_real_number(raw) and float(raw) > ram_bytes:
                ram_bytes = float(raw)

        vram_peak = gen_stats.get("peak_vram_usage")
        if isinstance(vram_peak, Mapping):
            raw_v = vram_peak.get("inBytes") or vram_peak.get("in_bytes")
            if _is_real_number(raw_v) and float(raw_v) > 0.0:
                vram_mb = round(float(raw_v) / mib, 2)

    if sampled_vram_peak_mb is not None and (vram_mb is None or sampled_vram_peak_mb > vram_mb):
        vram_mb = round(sampled_vram_peak_mb, 2)

    return round(ram_bytes / mib, 2), vram_mb


def run_exo_pair(
    *,
    base_url: str,
    model_key: str = "Llama-3.1-8B-Q4",
    prompt_tokens: int | None = None,
    max_new_tokens: int | None = None,
    seed: int = 42,
    min_nodes: int = 2,
    sharding: str = "Pipeline",
    instance_meta: str | None = "MlxRing",
    iface: str | None = None,
    power_source: str = "none",
    exo_commit: str = EXO_PINNED_COMMIT,
    exo_version: str = EXO_PINNED_VERSION,
    dry_run: bool = False,
    await_timeout_s: float = 180.0,
    request_timeout_s: float = 600.0,
    mock_lan_bytes: int | None = None,
    reuse_existing_instance: bool = False,
    token_id_cmd: str | None = None,
    nvidia_probe: Callable[[], tuple[float | None, float | None]] = _sample_nvidia_smi_once,
    powermetrics_probe: Callable[[], float | None] = _sample_powermetrics_once,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one Milestone 1 EXO pair measurement and return ``(cl02_record, pair_details)``."""
    if model_key not in PINNED_MODELS:
        raise ValueError(
            f"Unknown model {model_key!r}; expected one of {sorted(PINNED_MODELS)}"
        )
    if power_source not in ALLOWED_POWER_SOURCES:
        raise ValueError(
            f"power_source must be one of {sorted(ALLOWED_POWER_SOURCES)}, got {power_source!r}"
        )

    pin = PINNED_MODELS[model_key]
    target_prompt_tokens = (
        prompt_tokens if prompt_tokens is not None else pin.default_prompt_tokens
    )
    target_max_new = (
        max_new_tokens if max_new_tokens is not None else pin.default_max_new_tokens
    )
    prompt_text, prompt_sha16 = build_deterministic_prompt(target_prompt_tokens, seed=seed)

    client = ExoApiClient(base_url, timeout_s=request_timeout_s)
    state_before = client.get_state()

    created_instance_id: str | None = None
    instance_created = False
    chosen_sharding = sharding
    chosen_meta = instance_meta or ""
    node_ids: list[str] = []

    if reuse_existing_instance:
        existing_instances = state_before.get("instances") or {}
        if isinstance(existing_instances, Mapping):
            for inst in existing_instances.values():
                if isinstance(inst, Mapping):
                    nids = _extract_instance_node_ids(inst)
                    if len(nids) >= min_nodes:
                        node_ids = nids
                        chosen_sharding = "existing"
                        chosen_meta = "existing"
                        break

    errors: list[str] = []
    token_ids: list[int] = []
    token_texts: list[str] = []
    gen_stats: dict[str, Any] | None = None
    t_start = 0.0
    t_first: float | None = None
    t_end = 0.0
    bytes_before: int | None = None
    bytes_after: int | None = None
    state_after: dict[str, Any] = state_before
    exo_sampler: _ExoNodeSystemSampler | None = None
    hw_sampler: _HardwareProbeSampler | None = None

    # Cover creation, readiness await, sampling, and inference in one try/finally
    # so any failure after create_instance still deletes the created instance.
    try:
        if not node_ids:
            previews = client.get_instance_previews(pin.exo_model_id)
            chosen_instance, node_ids, chosen_sharding, chosen_meta = select_pair_placement(
                previews,
                min_nodes=min_nodes,
                sharding=sharding,
                instance_meta=instance_meta,
            )
            created_instance_id = _extract_instance_id(chosen_instance)
            create_resp = client.create_instance(chosen_instance)
            instance_created = True
            if created_instance_id is None and isinstance(create_resp, Mapping):
                resp_id = create_resp.get("instance_id") or create_resp.get("instanceId")
                if resp_id is not None:
                    created_instance_id = str(resp_id)
            ready_event = client.await_instance(pin.exo_model_id, timeout_s=await_timeout_s)
            if created_instance_id is None and isinstance(ready_event.get("instance"), Mapping):
                created_instance_id = _extract_instance_id(ready_event["instance"])

        exo_sampler = _ExoNodeSystemSampler(client, node_ids, interval_s=0.1 if dry_run else 0.25)
        hw_sampler = _HardwareProbeSampler(
            power_source,
            interval_s=0.1 if dry_run else 0.25,
            nvidia_probe=nvidia_probe,
            powermetrics_probe=powermetrics_probe,
        )
        exo_sampler.start()
        hw_sampler.start()

        bytes_before = read_iface_bytes(iface) if iface else None
        payload = {
            "model": pin.exo_model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": target_max_new,
            "temperature": 0,
            "seed": seed,
            "logprobs": True,
            "use_prefix_cache": False,
        }
        t_start = time.perf_counter()
        try:
            for raw_line in client.stream_bench_chat_completions(payload):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(": generation_stats "):
                    try:
                        parsed_stats = json.loads(line[len(": generation_stats ") :])
                        if isinstance(parsed_stats, dict):
                            gen_stats = parsed_stats
                    except json.JSONDecodeError as exc:
                        errors.append(f"malformed generation_stats comment: {exc}")
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    errors.append(f"malformed SSE chunk: {exc}")
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0]
                delta = choice0.get("delta") or {}
                content = delta.get("content")
                tok_id = delta.get("token_id")
                if tok_id is None:
                    logprobs = choice0.get("logprobs")
                    if isinstance(logprobs, Mapping):
                        lp_content = logprobs.get("content")
                        if isinstance(lp_content, list) and lp_content:
                            tok_id = lp_content[0].get("token_id")
                if content or tok_id is not None:
                    if t_first is None:
                        t_first = time.perf_counter()
                    if _is_int(tok_id):
                        token_ids.append(tok_id)
                    if isinstance(content, str):
                        token_texts.append(content)
        except (ExoRunnerError, OSError, http.client.HTTPException, ValueError) as exc:
            errors.append(f"inference request failed: {exc}")
        t_end = time.perf_counter()
        if iface:
            bytes_after = read_iface_bytes(iface)
        state_after = client.get_state()
    finally:
        if hw_sampler is not None:
            hw_sampler.stop()
        if exo_sampler is not None:
            exo_sampler.stop()
        if instance_created and created_instance_id is not None:
            try:
                client.delete_instance(created_instance_id)
            except (ExoRunnerError, OSError, http.client.HTTPException, ValueError) as exc:
                errors.append(f"instance cleanup failed ({created_instance_id}): {exc}")

    host_obj, nodes_detail = _summarize_cluster_nodes(state_after, node_ids)
    total_s = max(0.0, t_end - t_start)
    ttft_s = max(0.0, (t_first - t_start) if t_first is not None else total_s)

    joined_text = "".join(token_texts)
    output_text_sha256_16 = (
        hashlib.sha256(joined_text.encode()).hexdigest()[:16] if joined_text else None
    )

    # Strictly enforce integer token-ID basis for metrics.token_digest; never substitute text.
    if token_ids:
        generated_tokens = len(token_ids)
        token_digest = compute_token_digest(token_ids)
        digest_basis = "stream_token_ids"
    elif token_texts and token_id_cmd:
        try:
            resolved_ids = resolve_token_ids_via_cmd(token_id_cmd, joined_text)
        except ExoRunnerError as exc:
            generated_tokens = len(token_texts)
            token_digest = ""
            digest_basis = "unavailable"
            errors.append(str(exc))
        else:
            if resolved_ids:
                generated_tokens = len(resolved_ids)
                token_digest = compute_token_digest(resolved_ids)
                digest_basis = "token_id_cmd"
            else:
                generated_tokens = len(token_texts)
                token_digest = ""
                digest_basis = "unavailable"
                errors.append("--token-id-cmd returned an empty token ID list")
    elif token_texts:
        generated_tokens = len(token_texts)
        token_digest = ""
        digest_basis = "unavailable"
        errors.append(
            "EXO /bench/chat/completions omitted integer token_id; refusing to substitute "
            "text chunks for token-ID token_digest (pass --token-id-cmd or patch EXO)"
        )
    else:
        generated_tokens = 0
        token_digest = ""
        digest_basis = "none"
        if not errors:
            errors.append("no tokens generated")

    decode_s = max(0.0, t_end - t_first) if t_first is not None else 0.0
    if generated_tokens > 1 and decode_s > 0.0:
        decode_tok_s = (generated_tokens - 1) / decode_s
    elif isinstance(gen_stats, Mapping) and _is_real_number(gen_stats.get("generation_tps")):
        decode_tok_s = float(gen_stats["generation_tps"])
    else:
        decode_tok_s = 0.0

    reported_prompt_tokens = target_prompt_tokens
    if isinstance(gen_stats, Mapping) and _is_int(gen_stats.get("prompt_tokens")):
        reported_prompt_tokens = int(gen_stats["prompt_tokens"])

    sampled_vram_peak = hw_sampler.vram_peak_mib if hw_sampler is not None else None
    ram_peak_mb, vram_peak_mb = _extract_memory_peaks_mb(
        gen_stats, sampled_vram_peak_mb=sampled_vram_peak
    )

    if mock_lan_bytes is not None:
        lan_mbps = compute_lan_mbps(0, mock_lan_bytes, total_s)
    else:
        lan_mbps = compute_lan_mbps(bytes_before, bytes_after, total_s)

    # Power telemetry in cl02-baseline/1 comes strictly from the actually invoked host probe
    # (nvidia-smi or powermetrics); EXO /state/nodeSystem readings stay in details JSON only.
    hw_samples = hw_sampler.power_samples if hw_sampler is not None else []
    energy_j, power_avg_w = integrate_power_samples(hw_samples, t_start, t_end)
    if energy_j is not None and power_source in ("nvidia-smi", "powermetrics"):
        effective_power_source = power_source
    else:
        energy_j = None
        power_avg_w = None
        effective_power_source = "none"

    exo_sys_samples = exo_sampler.samples if exo_sampler is not None else []
    exo_sys_energy_j, exo_sys_avg_w = integrate_power_samples(exo_sys_samples, t_start, t_end)

    now_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", host_obj["name"]).strip("-") or "exo-pair"
    run_id = f"{now_stamp}-{host_slug}"

    cl02_record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "host": host_obj,
        "engine": {
            "name": "exo",
            "version": f"{exo_version}+{exo_commit[:12]}",
        },
        "model": {
            "id": pin.model_id,
            "revision": pin.revision,
            "quant": pin.quant,
        },
        "prompt": {
            "tokens": reported_prompt_tokens,
            "sha256_16": prompt_sha16,
        },
        "sampling": {
            "temperature": 0,
            "seed": seed,
            "max_new_tokens": target_max_new,
        },
        "metrics": {
            "token_digest": token_digest,
            "generated_tokens": generated_tokens,
            "ttft_s": round(ttft_s, 4),
            "decode_tok_s": round(decode_tok_s, 3),
            "total_s": round(total_s, 4),
            "ram_peak_mb": ram_peak_mb,
            "vram_peak_mb": vram_peak_mb,
            "errors": errors,
            "lan_mbps": lan_mbps,
            "energy_j": energy_j,
            "power_avg_w": power_avg_w,
            "power_source": effective_power_source,
        },
        "dry_run": dry_run,
    }

    validation_errors = validate_cl02_record(cl02_record)
    if validation_errors:
        raise ExoRunnerError(
            "Internal error: emitted record failed cl02-baseline/1 validation: "
            + "; ".join(validation_errors)
        )

    pair_details: dict[str, Any] = {
        "run_id": run_id,
        "exo_pin": {
            "repo": EXO_REPO_URL,
            "commit": exo_commit,
            "version": exo_version,
            "exo_model_id": pin.exo_model_id,
            "namespace": EXO_DEFAULT_NAMESPACE,
        },
        "placement": {
            "node_ids": node_ids,
            "node_count": len(node_ids),
            "sharding": chosen_sharding,
            "instance_meta": chosen_meta,
            "instance_id": created_instance_id,
        },
        "nodes": nodes_detail,
        "token_digest_basis": digest_basis,
        "output_text_sha256_16": output_text_sha256_16,
        "generation_stats": gen_stats,
        "exo_node_system_power": {
            "source": "exo:/state/nodeSystem:sysPower",
            "sample_count": len(exo_sys_samples),
            "energy_j": exo_sys_energy_j,
            "power_avg_w": exo_sys_avg_w,
        },
        "host_power_probe": {
            "requested_source": power_source,
            "effective_source": effective_power_source,
            "sample_count": len(hw_samples),
        },
        "reference_shape": EXO_REFERENCE_SHAPE,
        "dry_run": dry_run,
    }
    return cl02_record, pair_details


class MockExoServer:
    """In-process loopback mock of EXO ``a60b95698ca8`` for dry-run self-tests."""

    def __init__(
        self,
        *,
        prompt_tokens: int = 8192,
        prefill_delay_s: float = 0.04,
        token_delay_s: float = 0.002,
        emit_token_ids: bool = True,
        await_behavior: str = "ready",
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.prefill_delay_s = prefill_delay_s
        self.token_delay_s = token_delay_s
        self.emit_token_ids = emit_token_ids
        self.await_behavior = await_behavior
        self.created_instances: dict[str, dict[str, Any]] = {}
        self.deleted_instances: list[str] = []
        self.last_request: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def __enter__(self) -> MockExoServer:  # noqa: PYI034
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send_json(self, status: int, body: Any) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _state_payload(self) -> dict[str, Any]:
                with mock._lock:
                    instances = dict(mock.created_instances)
                return {
                    "nodeIdentities": {
                        "node-mac-m2max": {
                            "friendlyName": "mac-m2max-macos",
                            "osVersion": "macOS 26.6.2",
                            "chipId": "Apple M2 Max",
                            "gpu": "Apple M2 Max GPU",
                        },
                        "node-spark-gb10": {
                            "friendlyName": "dgx-spark-dgxos",
                            "osVersion": "Ubuntu 24.04 (DGX OS)",
                            "chipId": "NVIDIA GB10",
                            "gpu": "NVIDIA GB10 CUDA",
                        },
                    },
                    "nodeMemory": {
                        "node-mac-m2max": {
                            "ramTotal": {"inBytes": 64 * 1024**3},
                            "ramAvailable": {"inBytes": 60 * 1024**3},
                        },
                        "node-spark-gb10": {
                            "ramTotal": {"inBytes": 128 * 1024**3},
                            "ramAvailable": {"inBytes": 125 * 1024**3},
                            "vramUsed": {"inBytes": 4200 * 1024**2},
                        },
                    },
                    "nodeSystem": {
                        "node-mac-m2max": {"sysPower": 24.5, "gpuUsage": 0.42, "temp": 58.0},
                        "node-spark-gb10": {"sysPower": 48.0, "gpuUsage": 0.78, "temp": 64.0},
                    },
                    "instances": instances,
                }

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                if parsed.path == "/node_id":
                    self._send_json(200, "node-mac-m2max")
                    return
                if parsed.path == "/state":
                    self._send_json(200, self._state_payload())
                    return
                if parsed.path == "/state/nodeSystem":
                    self._send_json(200, self._state_payload()["nodeSystem"])
                    return
                if parsed.path == "/state/nodeMemory":
                    self._send_json(200, self._state_payload()["nodeMemory"])
                    return
                if parsed.path == "/instance/previews":
                    inst = {
                        "MlxRingInstance": {
                            "instanceId": "dryrun-pair-inst-001",
                            "shardAssignments": {
                                "modelId": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                                "nodeToRunner": {
                                    "node-mac-m2max": "runner-mac",
                                    "node-spark-gb10": "runner-spark",
                                },
                            },
                        }
                    }
                    self._send_json(
                        200,
                        [
                            {
                                "sharding": "Pipeline",
                                "instance_meta": "MlxRing",
                                "error": None,
                                "instance": inst,
                            }
                        ],
                    )
                    return
                if parsed.path == "/instance/await":
                    if mock.await_behavior == "http_error":
                        self._send_json(503, {"detail": "simulated await failure"})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if mock.await_behavior == "timeout":
                        event = json.dumps({"type": "timeout", "message": "simulated timeout"})
                    else:
                        with mock._lock:
                            inst = next(iter(mock.created_instances.values()), {})
                        event = json.dumps({"type": "ready", "instance": inst})
                    self.wfile.write(f"data: {event}\n\n".encode())
                    self.wfile.flush()
                    return
                self._send_json(404, {"detail": f"not found: {parsed.path}"})

            def do_POST(self) -> None:
                parsed = urlsplit(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                if parsed.path == "/instance":
                    inst = body.get("instance") or {}
                    inst_id = _extract_instance_id(inst) or "dryrun-pair-inst-001"
                    with mock._lock:
                        mock.created_instances[inst_id] = inst
                    self._send_json(200, {"status": "accepted", "instance_id": inst_id})
                    return
                if parsed.path == "/bench/chat/completions":
                    mock.last_request = body
                    max_tokens = int(body.get("max_tokens", 32))
                    model_id = str(body.get("model", ""))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    time.sleep(mock.prefill_delay_s)
                    for i in range(max_tokens):
                        word = _PROMPT_VOCAB[i % len(_PROMPT_VOCAB)]
                        piece = word if i == 0 else " " + word
                        delta: dict[str, Any] = {"content": piece}
                        if mock.emit_token_ids:
                            delta["token_id"] = 1000 + i
                        chunk = {
                            "id": "dryrun-cmpl-001",
                            "object": "chat.completion.chunk",
                            "created": 1763784000,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if i + 1 < max_tokens:
                            time.sleep(mock.token_delay_s)
                    stats = {
                        "prompt_tokens": mock.prompt_tokens,
                        "generation_tokens": max_tokens,
                        "prompt_tps": round(
                            mock.prompt_tokens / max(mock.prefill_delay_s, 1e-4), 2
                        ),
                        "generation_tps": round(
                            max_tokens / max(max_tokens * mock.token_delay_s, 1e-4), 2
                        ),
                        "peak_memory_usage": {"inBytes": 4820 * 1024 * 1024},
                        "prefix_cache_hit": "none",
                    }
                    self.wfile.write(f": generation_stats {json.dumps(stats)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                self._send_json(404, {"detail": f"not found: {parsed.path}"})

            def do_DELETE(self) -> None:
                parsed = urlsplit(self.path)
                if parsed.path.startswith("/instance/"):
                    inst_id = parsed.path[len("/instance/") :]
                    with mock._lock:
                        mock.created_instances.pop(inst_id, None)
                        mock.deleted_instances.append(inst_id)
                    self._send_json(200, {"deleted": inst_id})
                    return
                self._send_json(404, {"detail": f"not found: {parsed.path}"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def format_setup_instructions() -> str:
    """Print reproducible EXO pin + launch commands for macOS and DGX OS."""
    return f"""Milestone 1 EXO pair setup (pinned commit {EXO_PINNED_COMMIT}, v{EXO_PINNED_VERSION})

1. macOS node (Apple Silicon, Metal MLX):
   git clone {EXO_REPO_URL} ~/src/exo
   git -C ~/src/exo checkout {EXO_PINNED_COMMIT}
   cd ~/src/exo && uv sync --extra mlx
   uv run exo --namespace {EXO_DEFAULT_NAMESPACE} --api-port {EXO_DEFAULT_PORT}

2. DGX Spark node (DGX OS / Ubuntu ARM64, CUDA 13):
   git clone {EXO_REPO_URL} ~/src/exo
   git -C ~/src/exo checkout {EXO_PINNED_COMMIT}
   cd ~/src/exo && uv sync --extra mlx-cuda13
   uv run exo --namespace {EXO_DEFAULT_NAMESPACE} --api-port {EXO_DEFAULT_PORT}

3. Run Milestone 1 EXO pair baseline (8,192-token Llama-3.1-8B-Q4 greedy, 32 tokens):
   # Stock EXO at {EXO_PINNED_COMMIT[:12]} does not emit integer token_id on
   # /bench/chat/completions SSE chunks; pass --token-id-cmd backed by the pinned
   # model tokenizer (or patch EXO to emit token_id) so token_digest uses real IDs.
   python3 -m omarchy_cluster.exo_pair \\
     --exo-url http://<coordinator-ip>:{EXO_DEFAULT_PORT} \\
     --exo-repo ~/src/exo \\
     --model Llama-3.1-8B-Q4 \\
     --iface <lan-iface> \\
     --power-source nvidia-smi \\
     --token-id-cmd '<tokenizer-cmd>' \\
     --out-dir receipts/milestone1-exo-pair/live-$(date -u +%Y%m%dT%H%M%SZ)
"""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 1 EXO pair (macOS + DGX OS) runner emitting cl02-baseline/1 JSON."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end against an in-process loopback mock EXO pair server.",
    )
    parser.add_argument(
        "--print-setup",
        action="store_true",
        help="Print pinned EXO install/launch commands for macOS + DGX OS and exit.",
    )
    parser.add_argument(
        "--exo-url",
        default=f"http://127.0.0.1:{EXO_DEFAULT_PORT}",
        help=f"Coordinator EXO API URL (default: http://127.0.0.1:{EXO_DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--exo-repo",
        type=Path,
        default=None,
        help="Local EXO git checkout to verify against pinned commit before running.",
    )
    parser.add_argument(
        "--allow-dirty-exo",
        action="store_true",
        help="Allow dirty working tree in --exo-repo (commit SHA is still enforced).",
    )
    parser.add_argument(
        "--model",
        choices=sorted(PINNED_MODELS.keys()),
        default="Llama-3.1-8B-Q4",
        help="Pinned Milestone 1 model (default: Llama-3.1-8B-Q4).",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="Override target prompt token count (default: 8192 for Llama-3.1-8B-*).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generated token count (default: 32 for Llama-3.1-8B-*).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Greedy decoding seed (default: 42).")
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=2,
        help="Minimum participating EXO nodes required in placement (default: 2).",
    )
    parser.add_argument(
        "--sharding",
        default="Pipeline",
        choices=("Pipeline", "Tensor"),
        help="Preferred EXO sharding strategy (default: Pipeline).",
    )
    parser.add_argument(
        "--instance-meta",
        default="MlxRing",
        choices=("MlxRing", "MlxJaccl"),
        help="Preferred EXO backend transport (default: MlxRing for macOS+Linux LAN).",
    )
    parser.add_argument(
        "--iface",
        default=None,
        help="Network interface to measure LAN throughput (lan_mbps) across the request.",
    )
    parser.add_argument(
        "--power-source",
        default="none",
        choices=sorted(ALLOWED_POWER_SOURCES),
        help="Host power probe to invoke (nvidia-smi | powermetrics | none; default: none).",
    )
    parser.add_argument(
        "--token-id-cmd",
        default=None,
        help=(
            "Command reading generated text on stdin and writing a JSON array of integer "
            "token IDs to stdout (required when EXO stream omits integer token_id)."
        ),
    )
    parser.add_argument(
        "--reuse-instance",
        action="store_true",
        help="Reuse an already-loaded multi-node EXO instance instead of creating/deleting one.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write run.json (cl02-baseline/1) and exo-pair-details.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.print_setup:
        print(format_setup_instructions())
        return 0

    exo_commit = EXO_PINNED_COMMIT
    if args.exo_repo is not None:
        exo_commit = verify_exo_checkout(
            args.exo_repo, EXO_PINNED_COMMIT, allow_dirty=args.allow_dirty_exo
        )

    pin = PINNED_MODELS[args.model]
    prompt_tokens = (
        args.prompt_tokens if args.prompt_tokens is not None else pin.default_prompt_tokens
    )

    if args.dry_run:
        with MockExoServer(prompt_tokens=prompt_tokens) as mock:
            record, details = run_exo_pair(
                base_url=mock.base_url,
                model_key=args.model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                min_nodes=args.min_nodes,
                sharding=args.sharding,
                instance_meta=args.instance_meta,
                iface=None,
                power_source="none",
                exo_commit=exo_commit,
                dry_run=True,
                mock_lan_bytes=16 * 1024 * 1024,
            )
            if not mock.deleted_instances:
                raise ExoRunnerError("Dry-run teardown failed: mock instance was not deleted")
    else:
        record, details = run_exo_pair(
            base_url=args.exo_url,
            model_key=args.model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            min_nodes=args.min_nodes,
            sharding=args.sharding,
            instance_meta=args.instance_meta,
            iface=args.iface,
            power_source=args.power_source,
            exo_commit=exo_commit,
            dry_run=False,
            reuse_existing_instance=args.reuse_instance,
            token_id_cmd=args.token_id_cmd,
        )

    validation_errors = validate_cl02_record(record)
    if validation_errors:
        raise ExoRunnerError("cl02-baseline/1 validation failed: " + "; ".join(validation_errors))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "run.json").write_text(
            json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        (args.out_dir / "exo-pair-details.json").write_text(
            json.dumps(details, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    print(json.dumps(record, indent=2))
    return 0 if not record["metrics"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
