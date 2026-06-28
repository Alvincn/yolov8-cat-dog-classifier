from pathlib import Path


def test_conda_train_wrapper_uses_yolo8_environment():
    content = Path("scripts/train_yolo8_conda.sh").read_text()

    assert "conda run --no-capture-output -n yolo8" in content
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


def test_readme_does_not_document_global_or_python_venv_training():
    content = Path("README.md").read_text()

    forbidden_phrases = (
        "Python 3.11",
        "Python 3.12",
        "python3.11",
        "python3.12",
        "python3 -m venv",
        "pip install -r requirements.txt",
    )

    for phrase in forbidden_phrases:
        assert phrase not in content
