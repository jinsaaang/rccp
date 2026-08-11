#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python scripts/make_analysis_figures.py

echo
echo "Reproduction artifacts are under outputs/."
