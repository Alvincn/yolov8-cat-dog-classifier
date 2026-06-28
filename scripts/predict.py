from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = Path("runs/classify/cat_dog_yolov8n/weights/best.pt")


def validate_input_paths(model_path: Path, image_path: Path) -> tuple[Path, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    return model_path, image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用训练好的模型预测猫或狗")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    from scripts.runtime import configure_runtime_environment

    configure_runtime_environment()
    from ultralytics import YOLO

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
