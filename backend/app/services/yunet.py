from pathlib import Path

import cv2
import numpy as np

from app.domain.geometry import PixelBox
from app.services.preannotation import FaceCandidate


class YuNetFaceDetector:
    def __init__(self, model_path: Path) -> None:
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.80, 0.3, 5000)

    def detect(self, image: np.ndarray) -> tuple[FaceCandidate, ...]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return ()
        return tuple(
            FaceCandidate(
                box=PixelBox(
                    float(face[0]),
                    float(face[1]),
                    float(face[0] + face[2]),
                    float(face[1] + face[3]),
                ).clamp(width, height),
                confidence=float(face[-1]),
            )
            for face in faces
            if face[2] > 0 and face[3] > 0
        )
