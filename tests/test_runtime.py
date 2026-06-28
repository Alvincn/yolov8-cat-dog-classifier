import os
import sys
from types import SimpleNamespace

from scripts.runtime import configure_runtime_environment
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
