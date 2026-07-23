from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class TrainingState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


ALLOWED_TRANSITIONS = {
    TrainingState.QUEUED: {
        TrainingState.RUNNING,
        TrainingState.CANCELLING,
        TrainingState.INTERRUPTED,
        TrainingState.FAILED,
    },
    TrainingState.RUNNING: {
        TrainingState.CANCELLING,
        TrainingState.COMPLETED,
        TrainingState.FAILED,
        TrainingState.INTERRUPTED,
    },
    TrainingState.CANCELLING: {TrainingState.FAILED, TrainingState.INTERRUPTED},
    TrainingState.COMPLETED: set(),
    TrainingState.FAILED: set(),
    TrainingState.INTERRUPTED: set(),
}


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    preset: Literal["quick", "standard"] = "quick"
    epochs: int = 5
    image_size: int = 416
    batch_size: int = 4
    patience: int = 3
    device: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self) -> None:
        if not 1 <= self.epochs <= 300:
            raise ValueError("epochs must be between 1 and 300")
        if self.image_size not in {320, 416, 512, 640, 768}:
            raise ValueError("unsupported image size")
        if not 1 <= self.batch_size <= 128:
            raise ValueError("batch size must be between 1 and 128")
        if not 0 <= self.patience <= 50:
            raise ValueError("patience must be between 0 and 50")

    @classmethod
    def from_payload(cls, payload: dict) -> "TrainingSettings":
        preset = payload.get("preset", "quick")
        defaults = (
            {"epochs": 5, "image_size": 416, "batch_size": 4, "patience": 3}
            if preset == "quick"
            else {"epochs": 80, "image_size": 640, "batch_size": 16, "patience": 15}
        )
        return cls(
            preset=preset,
            epochs=payload.get("epochs", defaults["epochs"]),
            image_size=payload.get("image_size", defaults["image_size"]),
            batch_size=payload.get("batch_size", defaults["batch_size"]),
            patience=payload.get("patience", defaults["patience"]),
            device=payload.get("device", "auto"),
        )
