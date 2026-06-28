from __future__ import annotations

import argparse
import json
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from scripts.runtime import configure_runtime_environment
from scripts.runtime import patch_ultralytics_mps_autocast
from scripts.runtime import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_MODEL = Path("runs/classify/cat_dog_yolov8n_mps_final/weights/best.pt")


def translate_class_name(class_name: str) -> str:
    labels = {
        "cat": "猫",
        "dog": "狗",
    }
    return labels.get(class_name.lower(), class_name)


def parse_multipart_image(body: bytes, content_type: str) -> tuple[bytes, str]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("上传表单缺少 boundary")
    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
    delimiter = f"--{boundary}".encode()

    for part in body.split(delimiter):
        if b'name="image"' not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            raise ValueError("图片上传格式不正确")
        header_bytes = part[:header_end]
        image_bytes = part[header_end + 4 :]
        image_bytes = image_bytes.rstrip(b"\r\n-")
        filename = "upload.jpg"
        for header_line in header_bytes.decode("utf-8", errors="ignore").split("\r\n"):
            if "filename=" in header_line:
                filename = header_line.split("filename=", 1)[1].strip().strip('"')
                break
        if not image_bytes:
            raise ValueError("图片文件为空")
        return image_bytes, filename

    raise ValueError("没有收到图片文件")


class PredictionService:
    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        model_loader: Callable[[str], object] | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = resolve_device(device)
        self._model_loader = model_loader
        self._model: object | None = None

    def _load_model(self) -> object:
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if self._model is None:
            if self._model_loader is None:
                configure_runtime_environment()
                patch_ultralytics_mps_autocast(self.device)
                from ultralytics import YOLO

                self._model_loader = YOLO
            self._model = self._model_loader(str(self.model_path))
        return self._model

    def predict(self, image_bytes: bytes, suffix: str = ".jpg") -> dict[str, object]:
        model = self._load_model()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
            image_file.write(image_bytes)
            image_path = Path(image_file.name)

        try:
            results = model.predict(str(image_path), device=self.device, verbose=False)
        finally:
            image_path.unlink(missing_ok=True)

        probs = results[0].probs
        class_index = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = results[0].names[class_index]
        print(
            f"预测结果: {class_name} / {translate_class_name(class_name)}, "
            f"置信度: {confidence:.4f}",
            flush=True,
        )
        return {
            "class_name": class_name,
            "label": translate_class_name(class_name),
            "confidence": confidence,
        }


def build_handler(service: PredictionService) -> type[BaseHTTPRequestHandler]:
    class CatDogHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
                return
            self._send_json({"error": "页面不存在"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
                return

            try:
                image_bytes, filename = self._read_uploaded_image()
                suffix = Path(filename).suffix or ".jpg"
                result = service.predict(image_bytes, suffix=suffix)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_uploaded_image(self) -> tuple[bytes, str]:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("请使用表单上传图片")

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            return parse_multipart_image(body, content_type)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self._send_json({"error": f"文件不存在: {path}"}, HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return CatDogHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动猫狗分类 Web 服务")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = PredictionService(args.model, args.device)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    print(f"Web 页面: http://{args.host}:{args.port}", flush=True)
    print(f"模型文件: {args.model}", flush=True)
    print(f"运行设备: {service.device}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
