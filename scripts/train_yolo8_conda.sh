#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda run -n yolo8 python scripts/train.py "$@"
