from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("box values must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive area")

    def clamp(self, width: int, height: int) -> "PixelBox":
        x1 = min(max(0.0, self.x1), float(width))
        y1 = min(max(0.0, self.y1), float(height))
        x2 = min(max(0.0, self.x2), float(width))
        y2 = min(max(0.0, self.y2), float(height))
        return PixelBox(x1, y1, x2, y2)

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        box = self.clamp(width, height)
        return (
            (box.x1 + box.x2) / (2 * width),
            (box.y1 + box.y2) / (2 * height),
            (box.x2 - box.x1) / width,
            (box.y2 - box.y1) / height,
        )

    @classmethod
    def from_yolo(
        cls, value: tuple[float, float, float, float], width: int, height: int
    ) -> "PixelBox":
        center_x, center_y, box_width, box_height = value
        return cls(
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        )
