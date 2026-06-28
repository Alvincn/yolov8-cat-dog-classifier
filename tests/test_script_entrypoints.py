import subprocess
import sys


def test_train_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "训练 YOLOv8 猫狗分类模型" in result.stdout


def test_predict_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/predict.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "使用训练好的模型预测猫或狗" in result.stdout
