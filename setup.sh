#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Environment ready. Activate it with: source .venv/bin/activate"
echo "FineWeb-Edu is not downloaded by setup.sh. See README.md."
