import os
import sys
from types import SimpleNamespace

from scripts.runtime import configure_runtime_environment
from scripts.runtime import patch_ultralytics_mps_autocast
from scripts.runtime import resolve_device


def test_configure_runtime_environment_sets_project_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    configure_runtime_environment()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / ".cache" / "matplotlib")
    assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / ".cache" / "ultralytics")


def test_resolve_device_keeps_explicit_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"


def test_resolve_device_uses_mps_when_available(monkeypatch):
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("auto") == "mps"


def test_resolve_device_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch):
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("auto") == "cpu"


def test_patch_ultralytics_mps_autocast_uses_null_context_for_mps(monkeypatch):
    calls = []

    def original_autocast(enabled=True, device="cuda"):
        calls.append((enabled, device))
        raise RuntimeError("should not call original for mps")

    torch_utils = SimpleNamespace(autocast=original_autocast)
    validator = SimpleNamespace(autocast=original_autocast)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", torch_utils)
    monkeypatch.setitem(sys.modules, "ultralytics.engine.validator", validator)

    patch_ultralytics_mps_autocast("mps")

    with torch_utils.autocast(enabled=False, device="mps"):
        pass
    with validator.autocast(enabled=True, device="mps"):
        pass

    assert calls == []


def test_patch_ultralytics_mps_autocast_keeps_original_for_cpu(monkeypatch):
    calls = []

    def original_autocast(enabled=True, device="cuda"):
        calls.append((enabled, device))
        return "original-context"

    torch_utils = SimpleNamespace(autocast=original_autocast)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", torch_utils)

    patch_ultralytics_mps_autocast("cpu")

    assert torch_utils.autocast(enabled=False, device="cpu") == "original-context"
    assert calls == [(False, "cpu")]
