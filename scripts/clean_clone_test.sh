#!/usr/bin/env bash
# The evaluator's path on the committed state: fresh clone -> make setup -> make test (-> make demo with a key).
# Set SKIP_DEMO=1 to skip the live demo even when AI_API_KEY is set (e.g. on a rate-limited free-tier key).
set -euo pipefail
src="$(cd "$(dirname "$0")/.." && pwd)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone --quiet "$src" "$tmp/repo"
cd "$tmp/repo"
make setup
make test
if [ -n "${AI_API_KEY:-}" ] && [ -z "${SKIP_DEMO:-}" ]; then
  make demo
else
  echo "(skipping make demo: AI_API_KEY not set or SKIP_DEMO=1)"
fi
echo "CLEAN CLONE TEST: PASS"
