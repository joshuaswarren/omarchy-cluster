#!/usr/bin/env bash
# Build PhoneInference.app: UIKit host (main.m) + llama.cpp rpc-server as rpc_server_main.
# Prereq: build-ios.sh (CPU) or build-ios-metal.sh (CPU + Metal) built the ggml static libs.
# METAL=1 links the Metal build; rpc-server then serves MTL0 by default (-d CPU selects the CPU).
# Signing and install: xtool install.
set -euo pipefail
LLAMA=~/src/llama.cpp
SDK=${SDK:-~/sdk/SDKs/iPhoneOS27.0.sdk}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/out
APP=$OUT/PhoneInference.app
FLAGS=(--target=arm64-apple-ios17.0 -march=armv8.2-a+dotprod+fp16 -O3 -DNDEBUG -isysroot "$SDK")
DEFS=(-DGGML_USE_CPU -DGGML_USE_RPC)
if [[ ${METAL:-0} == 1 ]]; then
  B=$LLAMA/build-ios-metal
  DEFS+=(-DGGML_USE_METAL)
  EXTRA=("$B/ggml/src/ggml-metal/libggml-metal.a" -framework Metal)
else
  B=$LLAMA/build-ios
  EXTRA=()
fi
mkdir -p "$OUT" "$APP"
clang++ "${FLAGS[@]}" -std=c++17 -Dmain=rpc_server_main "${DEFS[@]}" \
  -I"$LLAMA/ggml/include" \
  -c "$LLAMA/tools/rpc/rpc-server.cpp" -o "$OUT/rpc-server.o"
clang "${FLAGS[@]}" -fobjc-arc -c "$HERE/main.m" -o "$OUT/main.o"
clang++ "${FLAGS[@]}" -fuse-ld=lld -Wl,-headerpad_max_install_names \
  "$OUT/main.o" "$OUT/rpc-server.o" \
  "$B/ggml/src/libggml.a" "$B/ggml/src/libggml-cpu.a" "${EXTRA[@]}" \
  "$B/ggml/src/ggml-rpc/libggml-rpc.a" "$B/ggml/src/libggml-base.a" \
  -framework UIKit -framework Foundation -lobjc -lm \
  ~/sdk/SDKs/iPhoneOS.sdk/usr/lib/clang/21/lib/darwin/libclang_rt.ios.a -o "$APP/PhoneInference"
cp "$HERE/Info.plist" "$APP/Info.plist"
file "$APP/PhoneInference"
sha256sum "$APP/PhoneInference"
