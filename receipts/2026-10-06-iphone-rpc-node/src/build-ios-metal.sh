#!/usr/bin/env bash
# Cross-build ggml (CPU + Metal + RPC) static libs for iOS arm64 from Linux.
# GGML_METAL_EMBED_LIBRARY embeds the .metal source; the phone compiles it at runtime (no xcrun/metal on Linux).
set -euo pipefail
LLAMA=~/src/llama.cpp
SDK=${SDK:-~/sdk/SDKs/iPhoneOS27.0.sdk}
ARCH="arm64-apple-ios17.0"
MARCH="-march=armv8.2-a+dotprod+fp16"
B=$LLAMA/build-ios-metal
cmake -B "$B" -S "$LLAMA" \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_OSX_SYSROOT="$SDK" -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_C_FLAGS="--target=$ARCH $MARCH" -DCMAKE_CXX_FLAGS="--target=$ARCH $MARCH" \
  -DCMAKE_OBJC_FLAGS="--target=$ARCH $MARCH" -DCMAKE_ASM_FLAGS="--target=$ARCH" \
  -DCMAKE_EXE_LINKER_FLAGS="--target=$ARCH -isysroot $SDK -fuse-ld=lld" \
  -DBUILD_SHARED_LIBS=OFF -DGGML_RPC=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$B" --target ggml ggml-cpu ggml-metal ggml-rpc ggml-base -j4
find "$B/ggml/src" -name "*.a" -printf "%p %s\n"
