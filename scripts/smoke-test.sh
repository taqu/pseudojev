#!/usr/bin/env bash
# Smoke test for a pjev release directory (unpacked archive).
# Usage: ./smoke-test.sh /path/to/release/dir
# With the four runtime resources in the release directory (see README: Runtime resources) it also
# loads them and runs one decision per task kind.
set -euo pipefail

RELEASE_DIR="${1:-}"

if [ -z "$RELEASE_DIR" ]; then
    echo "Usage: $0 <release-dir>"
    exit 1
fi

PJEV="${RELEASE_DIR}/pjev"
[ -x "$PJEV" ] || PJEV="${RELEASE_DIR}/pjev.exe"
if [ ! -x "$PJEV" ]; then
    echo "ERROR: pjev executable not found in $RELEASE_DIR"
    exit 1
fi

echo "=== pjev smoke test ==="
echo "Release: $RELEASE_DIR"

echo ""
echo "-- version --"
"$PJEV" --version

echo ""
echo "-- package contents --"
for f in LICENSE README.md THIRD-PARTY-NOTICES.md pjev.json; do
    [ -f "${RELEASE_DIR}/$f" ] && echo "ok: $f" || { echo "MISSING: $f"; exit 1; }
done

echo ""
echo "-- diagnostics --"
"$PJEV" diagnostics

if "$PJEV" diagnostics --json | grep -q '"accessible": false'; then
    echo ""
    echo "Runtime resources not present: skipping decisions."
    exit 0
fi

echo ""
echo "-- decisions (in-process) --"
"$PJEV" diagnostics --load
"$PJEV" noul --direct --state "The sky is blue." --question "Is it daytime?"
"$PJEV" choice --direct --question "Which queue?" --option billing:"Billing" --option returns:"Returns"
"$PJEV" score --direct --question "How urgent is a damaged order?" \
    --level "Not urgent" --level "Somewhat urgent" --level "Very urgent"

echo ""
echo "=== Smoke test checks complete ==="
