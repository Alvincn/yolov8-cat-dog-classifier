from __future__ import annotations

from contextlib import nullcontext
import importlib
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


def patch_ultralytics_mps_autocast(resolved_device: str) -> None:
    if resolved_device != "mps":
        return

    try:
        torch_utils = importlib.import_module("ultralytics.utils.torch_utils")
    except Exception:
        return

    if getattr(torch_utils.autocast, "_catdog_mps_safe", False):
        return

    original_autocast = torch_utils.autocast

    def mps_safe_autocast(enabled: bool = True, device: str = "cuda"):
        if device == "mps":
            return nullcontext()
        return original_autocast(enabled=enabled, device=device)

    mps_safe_autocast._catdog_mps_safe = True  # type: ignore[attr-defined]
    torch_utils.autocast = mps_safe_autocast

    try:
        validator = importlib.import_module("ultralytics.engine.validator")
        validator.autocast = mps_safe_autocast
    except Exception:
        pass
