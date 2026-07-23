import numpy as np
from app.domain.geometry import PixelBox
from app.inference.pretrained import PretrainedGlassesInferenceEngine
from app.services.preannotation import FaceCandidate


class Input:
    name = "images"


class Session:
    def get_inputs(self) -> list[Input]:
        return [Input()]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _outputs, inputs) -> list[np.ndarray]:
        assert inputs["images"].shape == (3, 3, 256, 256)
        # any-glasses and sunglasses logits for eyeglasses, sunglasses, and no glasses.
        return [np.asarray([[8, -8], [8, 8], [-8, -8]], dtype=np.float32)]


class FaceDetector:
    def detect(self, _image: np.ndarray) -> tuple[FaceCandidate, ...]:
        return (
            FaceCandidate(PixelBox(10, 10, 50, 70), 0.95),
            FaceCandidate(PixelBox(60, 10, 100, 70), 0.95),
            FaceCandidate(PixelBox(110, 10, 150, 70), 0.95),
        )


def test_each_face_gets_exactly_one_mutually_exclusive_class() -> None:
    engine = PretrainedGlassesInferenceEngine(Session(), FaceDetector(), "pretrained")

    result = engine.infer(np.zeros((100, 180, 3), dtype=np.uint8), 0.25, 0.45)

    assert [item.class_name.value for item in result.detections] == [
        "eyeglasses",
        "sunglasses",
        "no_glasses",
    ]
    assert len({item.box for item in result.detections}) == 3
    assert result.device == "YuNet + CPUExecutionProvider"
