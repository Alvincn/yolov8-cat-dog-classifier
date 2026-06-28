from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_environment() -> None:
    cache_dir = Path(".cache")
    os.environ.setdefault("MPLCONFIGDIR", str((cache_dir / "matplotlib").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str((cache_dir / "ultralytics").resolve()))
