#!/usr/bin/env bash
# Fail (exit 1) if an API key or secret appears in tracked files or in the git history, or if a .env file is
# tracked. Obvious test fakes (containing fake/dummy/example/placeholder) and <PLACEHOLDER> values are allowed.
set -uo pipefail
cd "$(dirname "$0")/.."

PATTERNS=(
  'sk-[A-Za-z0-9_-]{20,}'
  'sk-ant-'
  'gsk_[A-Za-z0-9]{20,}'
  'AIza[0-9A-Za-z_-]{30,}'
  'gh[pous]_[A-Za-z0-9]{30,}'
  '(api[_-]?key|token|secret)[[:space:]]*[:=][[:space:]]*['"'"'"][^'"'"'"$ ]{8,}'
)
# looks_real( marks old test lines (in history) that build random fakes at runtime.
ALLOW='fake|dummy|example|placeholder|<[A-Za-z_ -]+>|looks_real\('
SELF=':!scripts/check_secrets.sh'
status=0

for pattern in "${PATTERNS[@]}"; do
  hits=$(git grep -nIiE -e "$pattern" -- . "$SELF" 2>/dev/null | grep -viE "$ALLOW" || true)
  if [ -n "$hits" ]; then
    echo "SECRET PATTERN in tracked files: $pattern"
    echo "$hits" | sed -E 's/(.{0,120}).*/  \1/'
    status=1
  fi
done

tracked_env=$(git ls-files | grep -E '(^|/)\.env$' || true)
if [ -n "$tracked_env" ]; then
  echo "A .env file is tracked by git: $tracked_env"
  status=1
fi
example=$(git show HEAD:.env.example 2>/dev/null || true)
if [ "$example" != "AI_API_KEY=" ]; then
  echo ".env.example must contain exactly 'AI_API_KEY=' (with no value)"
  status=1
fi

for pattern in "${PATTERNS[@]}"; do
  hits=$(git log -p --all -- . "$SELF" 2>/dev/null | grep -E '^[+-]' | grep -iE -e "$pattern" | grep -viE "$ALLOW" || true)
  if [ -n "$hits" ]; then
    echo "SECRET PATTERN in git history: $pattern"
    echo "WARNING: if a key was ever committed, rotate it and recreate the repo history."
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "SECRET SCAN: PASS"
fi
exit "$status"
