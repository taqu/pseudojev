#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cat "${REPO_ROOT}/VERSION")"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
BUILD_DIR="${REPO_ROOT}/build-release"

echo "=== pjev ${VERSION} release build ==="
echo "Platform: ${OS}-${ARCH}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF

cmake --build "${BUILD_DIR}" --config Release --parallel

echo ""
echo "=== Packaging ==="
cd "${BUILD_DIR}"
cpack -C Release

echo ""
echo "=== Release artifacts ==="
ls -lh pjev-*.tar.gz pjev-*.zip 2>/dev/null || true
