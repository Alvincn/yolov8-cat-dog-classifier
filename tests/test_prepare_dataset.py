from pathlib import Path

from PIL import Image

from scripts.prepare_dataset import prepare_dataset


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))
    image.save(path)


def test_prepare_dataset_splits_valid_images_and_skips_invalid_files(tmp_path):
    source = tmp_path / "dataset" / "PetImages"
    for class_name in ("Cat", "Dog"):
        for index in range(5):
            write_image(source / class_name / f"{index}.jpg")
        (source / class_name / "bad.jpg").write_text("not an image")
        (source / class_name / "Thumbs.db").write_text("not a jpg")

    output = tmp_path / "data" / "cat_dog_cls"

    summary = prepare_dataset(source, output, val_ratio=0.4, seed=7)

    assert summary["Cat"]["valid"] == 5
    assert summary["Dog"]["valid"] == 5
    assert summary["Cat"]["skipped"] == 1
    assert summary["Dog"]["skipped"] == 1
    assert len(list((output / "train" / "Cat").glob("*.jpg"))) == 3
    assert len(list((output / "val" / "Cat").glob("*.jpg"))) == 2
    assert len(list((output / "train" / "Dog").glob("*.jpg"))) == 3
    assert len(list((output / "val" / "Dog").glob("*.jpg"))) == 2
