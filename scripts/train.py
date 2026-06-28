from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_dataset_dir(dataset_dir: Path) -> Path:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"准备后的数据集目录不存在: {dataset_dir}")
    for split in ("train", "val"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"缺少 {split} 目录: {split_dir}")
    return dataset_dir


def train_model(args: argparse.Namespace) -> None:
    from scripts.runtime import configure_runtime_environment

    configure_runtime_environment()
    from ultralytics import YOLO

    dataset_dir = validate_dataset_dir(args.data)
    log_training_start(args)
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
    print("训练命令已结束。请查看 runs/classify 目录中的训练结果。", flush=True)


def log_training_start(args: argparse.Namespace) -> None:
    print("开始训练 YOLOv8 猫狗分类模型", flush=True)
    print(f"数据集: {args.data}", flush=True)
    print(f"模型: {args.model}", flush=True)
    print(f"训练轮数: {args.epochs}", flush=True)
    print(f"图片尺寸: {args.imgsz}", flush=True)
    print(f"批大小: {args.batch}", flush=True)
    print(f"设备: {args.device}", flush=True)
    print(f"运行名: {args.name}", flush=True)
    print("接下来 Ultralytics 会输出每个 epoch 的 loss、准确率和保存路径。", flush=True)
    print("如果长时间没有新输出，通常是在扫描图片、建立缓存或进行 CPU 训练。", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 猫狗分类模型")
    parser.add_argument("--data", type=Path, default=Path("data/cat_dog_cls"))
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default="cat_dog_yolov8n")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
