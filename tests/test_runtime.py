import os

from scripts.runtime import configure_runtime_environment


def test_configure_runtime_environment_sets_project_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    configure_runtime_environment()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / ".cache" / "matplotlib")
    assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / ".cache" / "ultralytics")
