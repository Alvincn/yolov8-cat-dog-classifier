from pathlib import Path


def test_conda_train_wrapper_uses_yolo8_environment():
    content = Path("scripts/train_yolo8_conda.sh").read_text()

    assert "conda run -n yolo8" in content
    assert "python scripts/train.py" in content


def test_conda_predict_wrapper_uses_yolo8_environment():
    content = Path("scripts/predict_yolo8_conda.sh").read_text()

    assert "conda run -n yolo8" in content
    assert "python scripts/predict.py" in content


def test_readme_documents_yolo8_conda_environment():
    content = Path("README.md").read_text()

    assert "Conda 环境" in content
    assert "yolo8" in content
    assert "scripts/train_yolo8_conda.sh" in content
