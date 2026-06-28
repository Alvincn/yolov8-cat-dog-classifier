from pathlib import Path

import pytest

from scripts.train import validate_dataset_dir


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
