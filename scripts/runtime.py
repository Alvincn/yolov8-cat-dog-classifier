from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_environment() -> None:
    cache_dir = Path(".cache")
    os.environ.setdefault("MPLCONFIGDIR", str((cache_dir / "matplotlib").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str((cache_dir / "ultralytics").resolve()))


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"
