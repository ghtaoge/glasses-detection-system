import pytest
from app.domain.training import TrainingSettings


def test_quick_preset_defaults() -> None:
    settings = TrainingSettings.from_payload({"preset": "quick"})

    assert settings.epochs == 5
    assert settings.image_size == 416
    assert settings.batch_size == 4


def test_image_size_must_be_supported() -> None:
    with pytest.raises(ValueError, match="image size"):
        TrainingSettings.from_payload({"preset": "quick", "image_size": 500})
