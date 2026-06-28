from pathlib import Path
from types import SimpleNamespace

from server import PredictionService
from server import parse_multipart_image
from server import translate_class_name


def test_translate_class_name_maps_cat_and_dog():
    assert translate_class_name("Cat") == "猫"
    assert translate_class_name("Dog") == "狗"
    assert translate_class_name("Other") == "Other"


def test_prediction_service_returns_readable_result(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_text("fake model")
    captured = {}

    class FakeModel:
        def predict(self, image_path, device, verbose):
            captured["image_path"] = Path(image_path)
            captured["device"] = device
            captured["verbose"] = verbose
            return [
                SimpleNamespace(
                    probs=SimpleNamespace(top1=1, top1conf=0.8765),
                    names={0: "Cat", 1: "Dog"},
                )
            ]

    service = PredictionService(
        model_path=model_path,
        device="cpu",
        model_loader=lambda path: FakeModel(),
    )

    result = service.predict(b"fake image bytes", suffix=".jpg")

    assert result == {
        "class_name": "Dog",
        "label": "狗",
        "confidence": 0.8765,
    }
    assert captured["device"] == "cpu"
    assert captured["verbose"] is False
    assert not captured["image_path"].exists()


def test_parse_multipart_image_reads_upload():
    boundary = "----cat-dog-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="cat.jpg"\r\n'
        "Content-Type: image/jpeg\r\n"
        "\r\n"
        "fake-image"
        "\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    image_bytes, filename = parse_multipart_image(
        body, f"multipart/form-data; boundary={boundary}"
    )

    assert image_bytes == b"fake-image"
    assert filename == "cat.jpg"
