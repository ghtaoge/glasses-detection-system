from dataclasses import asdict, dataclass

from app.domain.geometry import PixelBox
from app.domain.labels import ClassName


@dataclass(frozen=True, slots=True)
class Detection:
    box: PixelBox
    class_id: int
    class_name: ClassName
    confidence: float

    def as_dict(self) -> dict:
        return {
            "box": asdict(self.box),
            "class_id": self.class_id,
            "class_name": self.class_name.value,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class InferenceResult:
    width: int
    height: int
    detections: tuple[Detection, ...]
    model_id: str
    device: str
    duration_ms: float

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "detections": [item.as_dict() for item in self.detections],
            "model_id": self.model_id,
            "device": self.device,
            "duration_ms": self.duration_ms,
        }
