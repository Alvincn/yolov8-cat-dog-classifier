# YOLOv8 猫狗分类器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 搭建一个可以在本机 Mac 上训练 YOLOv8 猫狗分类模型的完整小项目。

**Architecture:** 保留原始数据集不动，先用准备脚本把 `dataset/PetImages/Cat` 和 `dataset/PetImages/Dog` 清洗并拆分成 YOLOv8 分类训练目录。训练脚本只负责调用 Ultralytics YOLOv8 分类模型，预测脚本只负责加载训练好的权重并输出单张图片的分类结果。

**Tech Stack:** Python、Ultralytics YOLOv8、Pillow、pytest、Apple Silicon MPS。

---

## 文件职责

- `requirements.txt`：列出运行、训练、测试需要安装的 Python 包。
- `.gitignore`：避免把虚拟环境、训练结果、准备后的数据副本、模型权重提交进 Git。
- `README.md`：中文教程，面向新手解释项目目的、安装、准备数据、训练和预测。
- `scripts/prepare_dataset.py`：验证图片是否可读，并生成 `data/cat_dog_cls/train|val/Cat|Dog`。
- `scripts/train.py`：检查准备好的数据集，然后用 `yolov8n-cls.pt` 训练分类模型。
- `scripts/predict.py`：加载训练好的 `best.pt`，对单张图片输出 `Cat` 或 `Dog` 及置信度。
- `tests/test_prepare_dataset.py`：测试数据准备逻辑，包括跳过损坏文件、按比例拆分、生成正确目录结构。
- `tests/test_train.py`：测试训练前的数据集路径检查。
- `tests/test_predict.py`：测试预测前的文件路径检查。

## Task 1: 项目基础文件

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`

- [x] **Step 1: 创建依赖文件**

写入 `requirements.txt`：

```txt
ultralytics>=8.0,<9.0
pillow>=10.0
pytest>=8.0
```

- [x] **Step 2: 创建忽略规则**

写入 `.gitignore`：

```gitignore
.venv/
__pycache__/
.pytest_cache/

data/cat_dog_cls/
runs/
*.pt
*.onnx

.DS_Store
Thumbs.db
```

- [x] **Step 3: 创建中文 README 初稿**

写入 `README.md`，包含：

```markdown
# YOLOv8 猫狗分类训练项目

这个项目会训练一个模型，用来判断一张动物图片是猫还是狗。

我们使用的是 YOLOv8 的“分类模型”，不是“检测模型”。分类模型回答“这张图是什么类别”，检测模型回答“图中物体在哪里”。你的需求是判断猫还是狗，所以分类模型更合适。

## 数据集

原始数据放在：

- `dataset/PetImages/Cat`
- `dataset/PetImages/Dog`

原始数据不会被修改。脚本会把可用图片复制到 `data/cat_dog_cls`，并自动拆分训练集和验证集。

## 安装依赖

建议使用 Python 3.11 或 3.12 创建虚拟环境。当前机器默认 Python 是 3.14，很多机器学习库可能还没有完全适配，所以不建议直接用系统默认 Python 训练。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 准备数据

```bash
python scripts/prepare_dataset.py
```

## 训练模型

```bash
python scripts/train.py
```

## 预测单张图片

```bash
python scripts/predict.py dataset/PetImages/Cat/0.jpg
```
```

- [x] **Step 4: 检查基础文件**

Run: `test -f requirements.txt && test -f .gitignore && test -f README.md`

Expected: exit code 0.

- [x] **Step 5: Commit**

```bash
git add requirements.txt .gitignore README.md
git commit -m "Add project setup docs"
```

## Task 2: 数据准备逻辑

**Files:**
- Create: `scripts/prepare_dataset.py`
- Create: `tests/test_prepare_dataset.py`

- [x] **Step 1: 写失败测试**

创建 `tests/test_prepare_dataset.py`，测试准备逻辑会跳过坏图片并生成训练/验证目录：

```python
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
```

- [x] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_prepare_dataset.py -v`

Expected: FAIL because `scripts.prepare_dataset` does not exist yet.

- [x] **Step 3: 实现数据准备脚本**

创建 `scripts/prepare_dataset.py`，包含：

```python
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


def copy_split(files: list[Path], output_dir: Path, class_name: str, val_ratio: float) -> tuple[int, int]:
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
```

- [x] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_prepare_dataset.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add scripts/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "Add dataset preparation script"
```

## Task 3: 训练脚本

**Files:**
- Create: `scripts/train.py`
- Create: `tests/test_train.py`
- Modify: `README.md`

- [x] **Step 1: 写失败测试**

创建 `tests/test_train.py`：

```python
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
```

- [x] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_train.py -v`

Expected: FAIL because `scripts.train` does not exist yet.

- [x] **Step 3: 实现训练脚本**

创建 `scripts/train.py`：

```python
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def validate_dataset_dir(dataset_dir: Path) -> Path:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"准备后的数据集目录不存在: {dataset_dir}")
    for split in ("train", "val"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"缺少 {split} 目录: {split_dir}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 猫狗分类模型")
    parser.add_argument("--data", type=Path, default=Path("data/cat_dog_cls"))
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="cat_dog_yolov8n")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = validate_dataset_dir(args.data)
    model = YOLO(args.model)
    model.train(
        data=str(dataset_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_train.py -v`

Expected: PASS.

- [x] **Step 5: 更新 README 的训练说明**

补充说明：

```markdown
训练时默认使用 `device=mps`，意思是让 PyTorch 尽量使用 Apple Silicon 的图形/神经计算能力，而不是只用 CPU。如果你的环境不支持 MPS，可以改成：

```bash
python scripts/train.py --device cpu
```
```

- [x] **Step 6: Commit**

```bash
git add scripts/train.py tests/test_train.py README.md
git commit -m "Add YOLOv8 training script"
```

## Task 4: 预测脚本

**Files:**
- Create: `scripts/predict.py`
- Create: `tests/test_predict.py`
- Modify: `README.md`

- [x] **Step 1: 写失败测试**

创建 `tests/test_predict.py`：

```python
from pathlib import Path

import pytest

from scripts.predict import validate_input_paths


def test_validate_input_paths_requires_model_and_image(tmp_path):
    model = tmp_path / "best.pt"
    image = tmp_path / "image.jpg"

    with pytest.raises(FileNotFoundError, match="模型"):
        validate_input_paths(model, image)

    model.write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="图片"):
        validate_input_paths(model, image)

    image.write_bytes(b"image")

    assert validate_input_paths(model, image) == (model, image)
```

- [x] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_predict.py -v`

Expected: FAIL because `scripts.predict` does not exist yet.

- [x] **Step 3: 实现预测脚本**

创建 `scripts/predict.py`：

```python
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = Path("runs/classify/cat_dog_yolov8n/weights/best.pt")


def validate_input_paths(model_path: Path, image_path: Path) -> tuple[Path, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    return model_path, image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的模型预测猫或狗")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path, image_path = validate_input_paths(args.model, args.image)
    model = YOLO(str(model_path))
    results = model.predict(str(image_path), device=args.device, verbose=False)
    probs = results[0].probs
    class_index = int(probs.top1)
    confidence = float(probs.top1conf)
    class_name = results[0].names[class_index]
    print(f"预测结果: {class_name}")
    print(f"置信度: {confidence:.4f}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_predict.py -v`

Expected: PASS.

- [x] **Step 5: 更新 README 的预测说明**

补充说明：

```markdown
预测脚本默认读取训练输出中的 `best.pt`。如果你的模型文件在其他位置，可以这样指定：

```bash
python scripts/predict.py path/to/image.jpg --model path/to/best.pt
```
```

- [x] **Step 6: Commit**

```bash
git add scripts/predict.py tests/test_predict.py README.md
git commit -m "Add prediction script"
```

## Task 5: 本地验证

**Files:**
- Modify: `README.md`

- [x] **Step 1: 运行全部单元测试**

Run: `python3 -m pytest -v`

Expected: all tests PASS.

- [x] **Step 2: 运行数据准备脚本**

Run: `python3 scripts/prepare_dataset.py`

Expected: prints valid/skipped/train/val counts for `Cat` and `Dog`.

- [x] **Step 3: 检查准备后的目录**

Run: `find data/cat_dog_cls -maxdepth 2 -type d -print`

Expected: output includes train/val and Cat/Dog class directories.

- [x] **Step 4: 如果依赖和权重下载可用，运行 1 个 epoch 训练冒烟测试**

Run: `python3 scripts/train.py --epochs 1 --name smoke_test`

Expected: training starts and writes output under `runs/classify/smoke_test`.

- [x] **Step 5: 根据验证结果更新 README**

如果因为 Python 版本、依赖安装或网络限制无法训练，在 README 写清楚原因和下一步建议。

- [x] **Step 6: Commit**

```bash
git add README.md
git commit -m "Document local verification results"
```

## 执行备注

实际执行时发现两点环境差异，并已更新实现和 README：

- 当前 Python/Torch 环境中 `torch.backends.mps.is_available()` 返回 `False`，所以训练和预测脚本默认设备从 `mps` 改为 `cpu`。如果以后安装了支持 MPS 的 PyTorch 环境，可以手动加 `--device mps`。
- Ultralytics 的默认分类输出目录已经是 `runs/classify`，所以训练脚本默认不再传 `project`，只传 `name`，避免生成 `runs/classify/runs/...` 这种重复路径。

已完成的额外验证：

- 使用 `/tmp/catdog_yolov8_smoke` 小样本数据集完成 1 个 epoch 的 YOLOv8 CPU 冒烟训练。
- 使用冒烟训练产出的 `best.pt` 跑通 `scripts/predict.py`。
- 为脚本入口、缓存目录、默认 CPU 设备补充了回归测试。

## 2026-06-28 Conda 环境调整

用户要求不要直接使用全局环境训练。当前机器的 Conda 中已有 YOLOv8 环境：

- 环境名：`yolo8`
- Python：3.8.20
- Ultralytics：8.4.80
- Torch：2.4.1
- MPS：不可用

因此当前训练和预测入口调整为包装脚本：

```bash
./scripts/train_yolo8_conda.sh
./scripts/predict_yolo8_conda.sh path/to/image.jpg
```

包装脚本内部使用：

```bash
conda run -n yolo8 python scripts/train.py
conda run -n yolo8 python scripts/predict.py path/to/image.jpg
```

这样做的原因是：新手最容易在终端里忘记当前 Python 来自哪里。包装脚本把“必须使用 yolo8 环境”固定下来，减少误用全局 Python 或其他虚拟环境的风险。

`README.md` 和设计文档已经同步更新为 Conda 优先说明。项目中保留 `requirements.txt`，但它现在只是备用依赖清单，不是当前推荐训练入口。
