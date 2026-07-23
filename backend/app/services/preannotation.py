from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.domain.geometry import PixelBox


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    box: PixelBox
    confidence: float


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> tuple[FaceCandidate, ...]:
        raise NotImplementedError


def choose_review_state(candidates: tuple[FaceCandidate, ...], width: int, height: int) -> str:
    valid = [candidate for candidate in candidates if candidate.confidence >= 0.80]
    if len(valid) != 1:
        return "needs_review"
    box = valid[0].box
    area_ratio = ((box.x2 - box.x1) * (box.y2 - box.y1)) / (width * height)
    return "auto_annotated" if 0.08 <= area_ratio <= 0.90 else "needs_review"
