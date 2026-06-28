from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


CLASSES = ("Cat", "Dog")


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def copy_split(
    files: list[Path],
    output_dir: Path,
    class_name: str,
    val_ratio: float,
) -> tuple[int, int]:
    val_count = int(round(len(files) * val_ratio))
    val_files = set(files[:val_count])
    train_count = 0
    val_copied = 0

    for source in files:
        split = "val" if source in val_files else "train"
        target_dir = output_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)
        if split == "val":
            val_copied += 1
        else:
            train_count += 1

    return train_count, val_copied


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = True,
) -> dict[str, dict[str, int]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {source_dir}")

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name in CLASSES:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"类别目录不存在: {class_dir}")

        valid_files = []
        skipped = 0
        for path in sorted(class_dir.glob("*.jpg")):
            if is_valid_image(path):
                valid_files.append(path)
            else:
                skipped += 1

        rng.shuffle(valid_files)
        train_count, val_count = copy_split(valid_files, output_dir, class_name, val_ratio)
        summary[class_name] = {
            "valid": len(valid_files),
            "skipped": skipped,
            "train": train_count,
            "val": val_count,
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 YOLOv8 猫狗分类数据集")
    parser.add_argument("--source", type=Path, default=Path("dataset/PetImages"))
    parser.add_argument("--output", type=Path, default=Path("data/cat_dog_cls"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(args.source, args.output, args.val_ratio, args.seed)
    for class_name, stats in summary.items():
        print(
            f"{class_name}: valid={stats['valid']}, skipped={stats['skipped']}, "
            f"train={stats['train']}, val={stats['val']}"
        )
    print(f"准备完成: {args.output}")


if __name__ == "__main__":
    main()
