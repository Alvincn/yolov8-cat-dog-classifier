from pathlib import Path

import pytest

from scripts.train import validate_dataset_dir
from scripts.train import build_parser
from scripts.train import log_training_start


def test_validate_dataset_dir_requires_train_and_val_folders(tmp_path):
    dataset = tmp_path / "cat_dog_cls"
    dataset.mkdir()

    with pytest.raises(FileNotFoundError, match="train"):
        validate_dataset_dir(dataset)

    (dataset / "train").mkdir()

    with pytest.raises(FileNotFoundError, match="val"):
        validate_dataset_dir(dataset)

    (dataset / "val").mkdir()

    assert validate_dataset_dir(dataset) == dataset


def test_build_parser_uses_non_nested_default_output_dir():
    args = build_parser().parse_args([])

    assert args.project is None
    assert args.name == "cat_dog_yolov8n"
    assert args.device == "auto"


def test_log_training_start_prints_human_readable_progress(capsys):
    args = build_parser().parse_args(["--epochs", "1", "--batch", "4"])

    log_training_start(args)

    output = capsys.readouterr().out
    assert "开始训练 YOLOv8 猫狗分类模型" in output
    assert "数据集: data/cat_dog_cls" in output
    assert "训练轮数: 1" in output
    assert "批大小: 4" in output
    assert "如果长时间没有新输出" in output
