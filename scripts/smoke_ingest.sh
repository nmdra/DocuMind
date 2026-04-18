#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
tmp_file="/tmp/documind-smoke-$$.txt"
printf 'DocuMind smoke test content\n' > "$tmp_file"
python3 -m uv run python ingest.py "$tmp_file"
rm -f "$tmp_file"
