#!/usr/bin/env bash
# esper-node.sh — join the omarchy-cluster llama.cpp split as the NVIDIA RPC node.
#
# Target host: a Dell laptop with a 4 GB Quadro M1200 (Maxwell, sm_50) on Omarchy
# (Arch) Linux. Idempotent: installs packages, builds ggml-rpc-server with CUDA
# targeting sm_50 at the exact llama.cpp commit the M1 Max node (jw16) runs — so the
# RPC protocol matches — opens ONLY the RPC ports to the two cluster LANs while
# running, removes the rules on exit, and serves in the foreground until Ctrl-C.
#
# On completion it serves TWO RPC nodes (reruns reuse the build and start in seconds):
#   tcp/50052 — the Quadro GPU (device CUDA0)
#   tcp/50053 — CPU + system RAM (device CPU), ~24 GB usable
# Both run under systemd-inhibit so the laptop cannot sleep while serving.
#
# Stop it with Ctrl-C. Re-run any time; safe to run again.
set -euo pipefail

COMMIT=65840ed53c8653bfbf3e9014d9cbf71ac8c08725   # llama.cpp commit jw16 runs (RPC-protocol identical)
PORT=50052       # GPU (CUDA0) node
PORT_CPU=50053   # CPU/RAM node
REPO="$HOME/src/llama.cpp-esper"
BUILD="$REPO/build-cuda-rpc"
BIN="$BUILD/bin/ggml-rpc-server"
CUDA12_PKG="cuda-12.9.1-2-x86_64.pkg.tar.zst"     # newest CUDA 12.x in the Arch Linux Archive
ARCHIVE="https://archive.archlinux.org/packages/c/cuda"

say() { printf '\n===== %s\n' "$*"; }

SUDO=""
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then SUDO="sudo"; fi

say "esper RPC node setup: llama.cpp ${COMMIT:0:12}, CUDA sm_50, tcp/$PORT"

# --- 1. packages -------------------------------------------------------------
say "installing base-devel cmake git curl ufw"
$SUDO pacman -S --noconfirm --needed base-devel cmake git curl ufw

# --- 2. GCC 14 from the Arch Linux Archive (host compiler for nvcc) ------------
# Two proven gotchas (reproduced on a fresh Arch chroot, 2026-10-06):
#   * gcc14 is NOT in the repos anymore (only gcc 16 + gcc15) — it lives in the
#     archive. It must be installed BEFORE the CUDA toolkit, because the archive
#     cuda-12.9.1-2 package itself depends on gcc14.
#   * gcc14-libs supplies libgcc_s.so; if the libdir still lacks it, nvcc's host
#     link test dies with "/usr/bin/ld: cannot find -lgcc_s".
say "ensuring GCC 14 host compiler for nvcc"
if ! command -v g++-14 >/dev/null 2>&1; then
  base="https://archive.archlinux.org/packages/g"
  pkg14=$(curl -fsSL "$base/gcc14/" \
          | grep -oE 'gcc14-14[^"?]*-x86_64\.pkg\.tar\.zst' | sort -uV | tail -n1)
  [[ -n "$pkg14" ]] || { echo "ERROR: no gcc14 package found in the archive" >&2; exit 1; }
  pkgv="${pkg14#gcc14-}"; pkgv="${pkgv%-x86_64.pkg.tar.zst}"
  pkg14l="gcc14-libs-$pkgv-x86_64.pkg.tar.zst"
  say "installing $pkg14l and $pkg14 from the Arch Linux Archive (one transaction)"
  $SUDO pacman -U --noconfirm \
    "$base/gcc14-libs/${pkg14l//'+'/%2B}" \
    "$base/gcc14/${pkg14//'+'/%2B}"
fi

# --- 3. CUDA toolchain that can still target sm_50 ----------------------------
# Arch's current cuda (13.x) dropped Maxwell (sm_50); CUDA <= 12.9 compiles it fine.
# The NVIDIA driver (580.x) still supports Maxwell and runs CUDA 12 binaries.
nvcc_ok() {
  local v
  v=$("$1" --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+$') || return 1
  [[ -n "$v" && "$v" -le 12 ]]
}

NVCC=""
if command -v nvcc >/dev/null 2>&1 && nvcc_ok nvcc; then
  NVCC="$(command -v nvcc)"
elif [[ -x /opt/cuda/bin/nvcc ]] && nvcc_ok /opt/cuda/bin/nvcc; then
  NVCC=/opt/cuda/bin/nvcc
fi

if [[ -z "$NVCC" ]]; then
  say "no CUDA <=12 toolkit found — installing $CUDA12_PKG from the Arch Linux Archive"
  say "(several GB download, unattended; Arch's current cuda 13.x cannot compile sm_50)"
  $SUDO pacman -U --noconfirm "$ARCHIVE/$CUDA12_PKG"
  NVCC=/opt/cuda/bin/nvcc
  nvcc_ok "$NVCC" || { echo "ERROR: $NVCC still cannot target sm_50" >&2; exit 1; }
fi
say "using CUDA toolchain: $NVCC ($("$NVCC" --version | grep -m1 release))"
export PATH="$(dirname "$NVCC"):$PATH"

LIBDIR14="$(dirname "$(g++-14 -print-file-name=libgcc.a)")"
if [[ -d "$LIBDIR14" && ! -e "$LIBDIR14/libgcc_s.so" ]]; then
  say "linking system libgcc_s into $LIBDIR14 (Arch gcc14 packaging gap)"
  $SUDO ln -sf /usr/lib/libgcc_s.so.1 "$LIBDIR14/libgcc_s.so"
fi

# --- 4. llama.cpp source at jw16's exact commit --------------------------------
say "fetching llama.cpp into $REPO and checking out $COMMIT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$REPO"
fi
git -C "$REPO" fetch origin
git -C "$REPO" checkout -q "$COMMIT" 2>/dev/null \
  || { git -C "$REPO" fetch origin "$COMMIT"; git -C "$REPO" checkout -q "$COMMIT"; }
say "checked out: $(git -C "$REPO" rev-parse --short=12 HEAD)"

# --- 5. build ggml-rpc-server with CUDA for sm_50 -------------------------------
if [[ -x "$BIN" ]]; then
  say "ggml-rpc-server already built at $BIN — skipping configure/compile"
else
  say "configuring cmake: GGML_CUDA=ON GGML_RPC=ON CMAKE_CUDA_ARCHITECTURES=50"
  CMAKE_ARGS=(
    -DGGML_CUDA=ON -DGGML_RPC=ON
    -DCMAKE_CUDA_ARCHITECTURES=50
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_HOST_COMPILER=g++-14
    -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
  )
  cmake -S "$REPO" -B "$BUILD" "${CMAKE_ARGS[@]}"

  say "compiling ggml-rpc-server (CUDA kernels for sm_50 — the long step, 10–30 min on this machine)"
  cmake --build "$BUILD" --target ggml-rpc-server -j"$(nproc)"
  [[ -x "$BIN" ]] || { echo "ERROR: $BIN was not built" >&2; exit 1; }
fi

# --- 6. firewall: open the RPC ports only to the cluster LANs, only while running --
if pgrep -f "$BIN" >/dev/null 2>&1; then
  say "stopping previous ggml-rpc-server instances"
  pkill -f "$BIN" || true
  sleep 1
fi

say "ufw: allow tcp/$PORT (GPU) and tcp/$PORT_CPU (CPU) from 192.168.10.0/24 and 192.168.3.0/24 (removed on exit)"
$SUDO ufw allow from 192.168.10.0/24 to any port "$PORT" proto tcp
$SUDO ufw allow from 192.168.3.0/24 to any port "$PORT" proto tcp
$SUDO ufw allow from 192.168.10.0/24 to any port "$PORT_CPU" proto tcp
$SUDO ufw allow from 192.168.3.0/24 to any port "$PORT_CPU" proto tcp
$SUDO ufw status verbose | grep -m1 -i status || true

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  rc=$?
  pkill -f "$BIN" 2>/dev/null || true
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill "$pid" 2>/dev/null || true; done
  for p in "$PORT" "$PORT_CPU"; do
    $SUDO ufw delete allow from 192.168.10.0/24 to any port "$p" proto tcp >/dev/null 2>&1 || true
    $SUDO ufw delete allow from 192.168.3.0/24 to any port "$p" proto tcp >/dev/null 2>&1 || true
  done
  echo; echo "===== servers stopped, firewall rules for tcp/$PORT and tcp/$PORT_CPU removed"
  exit "$rc"
}
trap cleanup EXIT INT TERM

wait_port() { # port seconds label — returns 0 once localhost accepts a connection
  local port=$1 secs=$2 label=$3 i
  for ((i = 0; i < secs; i++)); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then exec 3>&- 3<&- || true; return 0; fi
    sleep 1
  done
  echo "WARNING: $label did not open tcp/$port within ${secs}s" >&2
  return 1
}

say "GPU state"
nvidia-smi

INHIBIT=(systemd-inhibit --what=sleep:idle:handle-lid-switch --why="omarchy-cluster rpc node")

say "starting CUDA rpc-server on 0.0.0.0:$PORT (device CUDA0)"
"${INHIBIT[@]}" "$BIN" -d CUDA0 -H 0.0.0.0 -p "$PORT" &
PIDS+=($!)

say "starting CPU rpc-server on 0.0.0.0:$PORT_CPU (device CPU, $(nproc) threads)"
"${INHIBIT[@]}" "$BIN" -d CPU -H 0.0.0.0 -p "$PORT_CPU" -t "$(nproc)" &
PIDS+=($!)

wait_port "$PORT" 60 "CUDA rpc-server" || \
  echo ">>> WARNING: the CUDA node is not up — the cluster can still use tcp/$PORT_CPU (CPU/RAM)." >&2
wait_port "$PORT_CPU" 60 "CPU rpc-server" || { echo "ERROR: CPU rpc-server failed to start" >&2; exit 1; }

say "RPC servers ready: tcp/$PORT (Quadro GPU) + tcp/$PORT_CPU (CPU/RAM) — Ctrl-C to stop"
say "sleep/lid/idle are inhibited while these run; from the M1 Max node use --rpc <this-host-ip>:$PORT and/or :$PORT_CPU"
wait
