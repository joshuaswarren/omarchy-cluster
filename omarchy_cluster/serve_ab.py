"""C-a cross-machine serve A/B: one llama.cpp arm per host.

One host, one llama.cpp build (CUDA or Vulkan), one pinned Qwen3-4B GGUF, one
runner. Three rounds of a fixed three-prompt set run greedy through the same
request path omarchy_cluster.llamacpp_engine measures, and the receipt records
TTFT and decode medians plus a token-id digest per run. A llama-server binary
without a usable GPU device refuses before anything is measured, so a half
provisioned host cannot produce a quiet zero.

    python -m omarchy_cluster.serve_ab --label m2max-apple-vulkan \
        --device vulkan --llama-server ~/llama.cpp/build/bin/llama-server \
        --out ~/scratch/serve-ab/m2max-apple-vulkan.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .llamacpp_engine import generate, server_env, wait_http

SCHEMA = "omarchy-cluster-serve-ab-v1"
CARD = "C-a"
SPEC = {
    "repo": "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF",
    "revision": "4edb920b6f14e3b9284d4502a6485103d72cde05",
    "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "sha256": "8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc",
    "bytes": 2497280448,
}
PROMPTS = [
    ("p1-short", "Summarize the water cycle in one paragraph."),
    ("p2-switch", "Explain how a transistor can function as a switch in a simple circuit."),
    ("p3-long",
     "Describe the full path a single web request takes through a small three-tier "
     "deployment: a browser on a laptop, a reverse proxy on the edge host, an "
     "application process in a container, and a database on a second machine. Name "
     "the protocol hop at each boundary, what state each tier keeps, and which "
     "single component fails first when the database disk fills. Then state the "
     "one metric you would alert on to catch that failure before users notice it."),
]
N_PREDICT = 128
DEFAULT_CTX = 4096
DEFAULT_THREADS = 8
DEFAULT_PORT = 8129
DEFAULT_ROUNDS = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_gguf(path: Path) -> None:
    url = "https://huggingface.co/%s/resolve/%s/%s" % (SPEC["repo"], SPEC["revision"], SPEC["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as src, tmp_path.open("wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
        if tmp_path.stat().st_size != SPEC["bytes"] or sha256_file(tmp_path) != SPEC["sha256"]:
            raise SystemExit("downloaded GGUF does not match the pinned sha256; delete %s" % path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_gguf(path: Path) -> None:
    if path.exists():
        if sha256_file(path) != SPEC["sha256"]:
            raise SystemExit("GGUF at %s does not match the pinned sha256; replace it" % path)
        return
    print("downloading %s (%d bytes) at revision %s" % (SPEC["file"], SPEC["bytes"], SPEC["revision"]), flush=True)
    download_gguf(path)


def parse_devices(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and line != "Available devices:" and line != "(none)":
            lines.append(line)
    return lines


def llama_devices(llama_server: str) -> list[str]:
    out = subprocess.run([llama_server, "--list-devices"], capture_output=True, text=True, check=False)
    return parse_devices(out.stdout + "\n" + out.stderr)


def llama_version(llama_server: str) -> str:
    out = subprocess.run([llama_server, "--version"], capture_output=True, text=True, check=False)
    return " ".join((out.stdout + out.stderr).split())[:160]


def tokenize_digest(server_url: str, text: str) -> str:
    req = urllib.request.Request(server_url + "/tokenize",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        ids = json.loads(r.read())["tokens"]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def run_one(server_url: str, prompt: str) -> dict:
    res = generate(server_url, {"prompt": prompt, "max_tokens": N_PREDICT,
                                "sampling": {"temperature": 0}})
    res["digest"] = tokenize_digest(server_url, res["text"])
    del res["text"]
    return res


def scrub(value: str) -> str:
    """Public-receipt scrub: drop the home prefix and never carry a hostname."""
    home = os.path.expanduser("~")
    return value.replace(home + "/", "~/") if value.startswith(home) else value


def uname_string() -> str:
    u = os.uname()
    return " ".join([u.sysname, u.release, u.version, u.machine])


def machine_info() -> dict:
    compat_path = Path("/proc/device-tree/compatible")
    model = chip = None
    if compat_path.exists():
        compat = compat_path.read_bytes().decode(errors="replace").split("\0")
        model = next((x for x in compat if re.fullmatch(r"apple,j\w+", x)), None)
        chip = next((x for x in compat if re.fullmatch(r"apple,t\d+", x)), None)
    return {"uname": uname_string(), "model": model, "chip": chip}


def medians(runs: list[dict]) -> dict:
    out = {}
    for pid in {r["prompt"] for r in runs}:
        sel = [r for r in runs if r["prompt"] == pid]
        out[pid] = {"ttft_ms_median": round(statistics.median(r["prefill_ms"] for r in sel), 2),
                    "tokps_median": round(statistics.median(r["tokps"] for r in sel), 2)}
    return out


def server_command(llama_server: str, gguf: Path, port: int, threads: int, ctx: int, dev: str | None) -> list[str]:
    """llama-server argv for one arm. `-dev <name>` pins the GPU: a host that
    exposes several Vulkan devices would otherwise let llama-server spread
    layers across all of them, which is a split serve, not one arm."""
    cmd = [llama_server, "-m", str(gguf), "--host", "127.0.0.1",
           "--port", str(port), "-c", str(ctx), "-t", str(threads), "-ngl", "99"]
    if dev:
        cmd += ["-dev", dev]
    return cmd


def device_names(devices: list[str]) -> list[str]:
    return [line.split(":", 1)[0].strip() for line in devices if ":" in line]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="arm label recorded in the receipt, e.g. m2max-apple-vulkan")
    p.add_argument("--device", required=True, choices=["cuda", "vulkan"], help="backend family of the llama.cpp build")
    p.add_argument("--llama-server", required=True, help="path to the host's GPU llama-server binary")
    p.add_argument("--dev", default=None, help="device name to pin, e.g. Vulkan0 or CUDA0; required on multi-GPU hosts")
    p.add_argument("--gguf", type=Path, default=Path.home() / "scratch/serve-ab" / SPEC["file"])
    p.add_argument("--out", type=Path, required=True, help="receipt JSON path")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    args = p.parse_args(argv)

    devices = llama_devices(args.llama_server)
    if not devices:
        print("NO-GPU-DEVICE: %s lists no GPU backend; provision a CUDA or Vulkan "
              "llama.cpp build first" % args.llama_server, file=sys.stderr)
        return 3
    if args.dev and args.dev not in device_names(devices):
        print("NO-SUCH-DEVICE: %s does not list %s; seen: %s" % (
            args.llama_server, args.dev, ", ".join(device_names(devices))), file=sys.stderr)
        return 3
    ensure_gguf(args.gguf)
    version = llama_version(args.llama_server)
    binary_sha = sha256_file(Path(args.llama_server))

    cmd = server_command(args.llama_server, args.gguf, args.port, args.threads, args.ctx, args.dev)
    print("llama-server: %s" % " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd, env=server_env(), stdin=subprocess.DEVNULL)
    server_url = "http://127.0.0.1:%d" % args.port
    runs: list[dict] = []
    try:
        wait_http(server_url + "/health", deadline_s=600.0, proc=server)
        for rnd in range(1, args.rounds + 1):
            for pid, prompt in PROMPTS:
                rec = run_one(server_url, prompt)
                rec.update({"round": rnd, "prompt": pid})
                runs.append(rec)
                print("round %d %s: ttft %.0f ms, %.2f tok/s, %d tokens, digest %s"
                      % (rnd, pid, rec["prefill_ms"], rec["tokps"],
                         rec["completion_tokens"], rec["digest"][:12]), flush=True)
    finally:
        server.terminate()
        server.wait(10)

    digest = hashlib.sha256("".join(r["digest"] for r in runs).encode()).hexdigest()
    receipt = {
        "schema": SCHEMA, "card": CARD, "label": args.label, "device": args.device,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spec": SPEC, "llama_server": {"path": scrub(str(args.llama_server)), "sha256": binary_sha,
                                       "version": version, "devices": devices,
                                       "pinned_dev": args.dev},
        "gguf": {"path": scrub(str(args.gguf)), "sha256": SPEC["sha256"]},
        "settings": {"rounds": args.rounds, "prompts": [pid for pid, _ in PROMPTS],
                     "n_predict": N_PREDICT, "ctx": args.ctx, "threads": args.threads,
                     "sampling": "greedy", "cache_prompt": False},
        "runs": runs, "medians": medians(runs), "run_digest": digest,
        "host": machine_info(),
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    overall = medians(runs)
    print("Card row: serve A/B arm %s (%s): decode medians %s tok/s, TTFT medians %s ms, "
          "%d rounds x %d prompts, %d greedy tokens, run digest %s, src llama-server %s"
          % (args.label, args.device,
             "/".join(str(overall[pid]["tokps_median"]) for pid, _ in PROMPTS),
             "/".join(str(overall[pid]["ttft_ms_median"]) for pid, _ in PROMPTS),
             args.rounds, len(PROMPTS), N_PREDICT, digest[:12], version), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
