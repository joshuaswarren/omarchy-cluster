#!/usr/bin/env bash
# Cross-build llama.cpp ggml-rpc-server for iOS arm64 from m1max-linux (Linux).
# Pure C/C++ - no Swift - so the FINDINGS 15/16 Swift/SDK version matrix does not apply.
set -euo pipefail
LLAMA=~/src/llama.cpp
SDK=${SDK:-~/sdk/SDKs/iPhoneOS27.0.sdk}
ARCH="arm64-apple-ios17.0"
MARCH="-march=armv8.2-a+dotprod+fp16"   # common denominator M1 Max (m1max-linux) and A17 Pro (phone) -> identical kernels for token identity
cmake -B "$LLAMA/build-ios" -S "$LLAMA" \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_OSX_SYSROOT="$SDK" -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_C_FLAGS="--target=$ARCH $MARCH" \
  -DCMAKE_CXX_FLAGS="--target=$ARCH $MARCH" \
  -DCMAKE_EXE_LINKER_FLAGS="--target=$ARCH -isysroot $SDK -fuse-ld=lld" \
  -DBUILD_SHARED_LIBS=OFF -DGGML_RPC=ON -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build "$LLAMA/build-ios" --target ggml-rpc-server -j10
file "$LLAMA/build-ios/bin/ggml-rpc-server"
