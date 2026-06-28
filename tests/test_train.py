from pathlib import Path

import pytest

from scripts.train import validate_dataset_dir
from scripts.train import build_parser


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
    assert args.device == "cpu"
