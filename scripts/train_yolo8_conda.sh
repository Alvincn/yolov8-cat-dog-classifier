#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
echo "[1/2] 使用 Conda 环境 yolo8 启动训练..."
echo "[2/2] 下面开始显示 YOLOv8 训练日志，请保持终端窗口打开。"
conda run --no-capture-output -n yolo8 python scripts/train.py "$@"
