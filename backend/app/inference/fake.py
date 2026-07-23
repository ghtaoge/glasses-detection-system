import time

import numpy as np

from app.domain.geometry import PixelBox
from app.domain.inference import Detection, InferenceResult
from app.domain.labels import ClassName


class FakeInferenceEngine:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def infer(self, image: np.ndarray, confidence: float, _iou: float) -> InferenceResult:
        started = time.perf_counter()
        height, width = image.shape[:2]
        detections = ()
        if confidence <= 0.91:
            detections = (
                Detection(
                    PixelBox(width * 0.08, height * 0.18, width * 0.46, height * 0.82),
                    1,
                    ClassName.EYEGLASSES,
                    0.91,
                ),
                Detection(
                    PixelBox(width * 0.54, height * 0.2, width * 0.92, height * 0.84),
                    2,
                    ClassName.SUNGLASSES,
                    0.87,
                ),
            )
        return InferenceResult(
            width,
            height,
            detections,
            self.model_id,
            "CPUExecutionProvider (simulated)",
            (time.perf_counter() - started) * 1000,
        )
