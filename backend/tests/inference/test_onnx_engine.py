import numpy as np
from app.inference.onnx import OnnxInferenceEngine


class Input:
    name = "images"


class Session:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[Input]:
        return [Input()]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _outputs, _inputs) -> list[np.ndarray]:
        return [self.output]


def test_decodes_letterboxed_yolo_output() -> None:
    output = np.zeros((1, 7, 1), dtype=np.float32)
    output[0, :, 0] = [320, 320, 320, 160, 0.05, 0.9, 0.1]
    engine = OnnxInferenceEngine(Session(output), "model", 640)

    result = engine.infer(np.zeros((300, 600, 3), dtype=np.uint8), 0.25, 0.45)

    assert result.device == "CPUExecutionProvider"
    assert result.detections[0].class_name.value == "eyeglasses"
    box = result.detections[0].box
    assert (round(box.x1), round(box.y1), round(box.x2), round(box.y2)) == (150, 75, 450, 225)


def test_nms_is_class_aware() -> None:
    output = np.zeros((1, 7, 2), dtype=np.float32)
    output[0, :, 0] = [320, 320, 200, 200, 0.05, 0.9, 0.1]
    output[0, :, 1] = [320, 320, 200, 200, 0.05, 0.1, 0.8]
    engine = OnnxInferenceEngine(Session(output), "model", 640)

    result = engine.infer(np.zeros((640, 640, 3), dtype=np.uint8), 0.25, 0.45)

    assert len(result.detections) == 2
