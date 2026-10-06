#!/usr/bin/env bash
# Build PhoneInference.app on Linux: llama.cpp's ggml (CPU + Metal + RPC) cross-compiled for
# iOS, rpc-server.cpp renamed to rpc_server_main, and a UIKit host (PhoneInference/main.m).
#
# Metal: GGML_METAL_EMBED_LIBRARY embeds the shader source and the phone compiles it at
# start, so Linux needs no Metal compiler. The deployment target is iOS 18.0 so every
# @available check in ggml-metal resolves at compile time (no compiler-rt needed).
#
# Usage: LLAMA=~/src/llama.cpp SDK=~/sdk/iPhoneOS.sdk ios/build.sh [OUT_DIR]
#   LLAMA  llama.cpp checkout (tested at 65840ed)
#   SDK    output of ios/prepare-sdk.sh
# Needs clang, ld64.lld and cmake. Signing and install: see ios/README.md (xtool install).
set -euo pipefail
LLAMA=${LLAMA:?llama.cpp checkout}
SDK=${SDK:?arm64-ready iPhoneOS SDK from ios/prepare-sdk.sh}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/out}
TARGET=arm64-apple-ios18.0
MARCH=-march=armv8.2-a+dotprod+fp16   # matches the Linux build; keeps CPU kernels identical
BUILD=$OUT/ggml-ios
FLAGS=(--target=$TARGET $MARCH -O3 -DNDEBUG -isysroot "$SDK")

cmake -B "$BUILD" -S "$LLAMA" \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_OSX_SYSROOT="$SDK" -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_C_FLAGS="--target=$TARGET $MARCH" -DCMAKE_CXX_FLAGS="--target=$TARGET $MARCH" \
  -DCMAKE_OBJC_FLAGS="--target=$TARGET $MARCH" -DCMAKE_ASM_FLAGS="--target=$TARGET" \
  -DBUILD_SHARED_LIBS=OFF -DGGML_RPC=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$BUILD" --target ggml ggml-cpu ggml-metal ggml-rpc ggml-base -j"$(nproc)"

APP=$OUT/PhoneInference.app
mkdir -p "$APP"
clang++ "${FLAGS[@]}" -std=c++17 -Dmain=rpc_server_main -DGGML_USE_CPU -DGGML_USE_RPC -DGGML_USE_METAL \
  -I"$LLAMA/ggml/include" -c "$LLAMA/tools/rpc/rpc-server.cpp" -o "$OUT/rpc-server.o"
clang "${FLAGS[@]}" -fobjc-arc -c "$HERE/PhoneInference/main.m" -o "$OUT/main.o"
L=$BUILD/ggml/src
clang++ "${FLAGS[@]}" -fuse-ld=lld -Wl,-headerpad_max_install_names "$OUT/main.o" "$OUT/rpc-server.o" \
  "$L/libggml.a" "$L/libggml-cpu.a" "$L/ggml-metal/libggml-metal.a" "$L/ggml-rpc/libggml-rpc.a" \
  "$L/libggml-base.a" -framework Metal -framework UIKit -framework Foundation -lobjc -lm \
  -o "$APP/PhoneInference"
cp "$HERE/PhoneInference/Info.plist" "$APP/Info.plist"
echo "unsigned app: $APP"
