#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
tmp_file="/tmp/documind-smoke-$$.txt"
trap 'rm -f "$tmp_file"' EXIT

printf 'DocuMind smoke test content\n' > "$tmp_file"
if ! uv run python ingest.py "$tmp_file"; then
  echo "Smoke ingest failed" >&2
  exit 1
fi
