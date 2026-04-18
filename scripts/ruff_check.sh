#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m uv run ruff format --check .
python3 -m uv run ruff check .
