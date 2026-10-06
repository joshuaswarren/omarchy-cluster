#!/usr/bin/env bash
# Make an iPhoneOS SDK that clang + ld64.lld on Linux can link plain arm64 apps against.
#
# The iPhoneOS 27 SDK's .tbd stubs list only arm64e targets ([ arm64e-ios, arm64e.x1-ios ]),
# so ld64.lld finds no symbols for an arm64 binary. This copies the SDK and rewrites
# arm64e.x1-ios to arm64-ios in every .tbd. The source SDK is not modified.
#
# Usage: prepare-sdk.sh SOURCE_SDK DEST_SDK
#   SOURCE_SDK e.g. the one `xtool setup` installs:
#   ~/.cache/xtool/darwin-iPhoneOS27.0.xtoolsdk/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk
set -euo pipefail
src=${1:?source iPhoneOS.sdk}
dst=${2:?destination directory}
[[ -f $src/SDKSettings.json || -f $src/SDKSettings.plist ]] || { echo "not an SDK: $src" >&2; exit 1; }
[[ ! -e $dst ]] || { echo "destination exists: $dst" >&2; exit 1; }
mkdir -p "$(dirname "$dst")"
cp -a "$src" "$dst"
find "$dst" -name '*.tbd' -type f -print0 | xargs -0 sed -i 's/arm64e\.x1-ios/arm64-ios/g'
echo "arm64-ready SDK: $dst ($(grep -rl 'arm64-ios' --include='*.tbd' "$dst" | wc -l) stubs)"
