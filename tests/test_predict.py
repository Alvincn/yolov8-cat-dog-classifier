import pytest

from scripts.predict import build_parser
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


def test_build_parser_defaults_to_cpu_device():
    args = build_parser().parse_args(["sample.jpg"])

    assert args.device == "cpu"
