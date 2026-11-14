#!/usr/bin/env bash
# Write omarchy-cluster-<version>.tar.gz beside the PKGBUILD from the committed tree.
set -euo pipefail

here="$(dirname "$(readlink -f "$0")")"
root="$(git -C "$here" rev-parse --show-toplevel)"
version="$(sed -n 's/^pkgver=//p' "$here/PKGBUILD")"
git -C "$root" archive --format=tar.gz --prefix="omarchy-cluster-$version/" -o "$here/omarchy-cluster-$version.tar.gz" HEAD
echo "wrote $here/omarchy-cluster-$version.tar.gz"
