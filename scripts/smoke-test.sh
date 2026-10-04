#!/usr/bin/env bash
# Smoke test for pjev release artifact.
# Usage: ./smoke-test.sh /path/to/release/dir [model.gguf]
set -euo pipefail

RELEASE_DIR="${1:-}"
MODEL="${2:-model.gguf}"

if [ -z "$RELEASE_DIR" ]; then
    echo "Usage: $0 <release-dir> [model.gguf]"
    exit 1
fi

PJEV="${RELEASE_DIR}/pjev"
if [ ! -x "$PJEV" ]; then
    echo "ERROR: $PJEV not found or not executable"
    exit 1
fi

echo "=== pjev smoke test ==="
echo "Release: $RELEASE_DIR"

echo ""
echo "-- version check --"
"$PJEV" --version

echo ""
echo "-- startup (requires model at $RELEASE_DIR/$MODEL) --"
echo "Start server manually and run: curl http://127.0.0.1:8080/health"
echo "Then: curl http://127.0.0.1:8080/version"
echo "(Automated HTTP testing requires a real model)"

echo ""
echo "=== Smoke test checks complete ==="
