import pytest
from app.domain.geometry import PixelBox
from app.domain.labels import ClassName, class_id


def test_fixed_class_order() -> None:
    assert class_id(ClassName.NO_GLASSES) == 0
    assert class_id(ClassName.EYEGLASSES) == 1
    assert class_id(ClassName.SUNGLASSES) == 2


def test_pixel_box_round_trip() -> None:
    box = PixelBox(20, 10, 120, 90)
    assert box.to_yolo(200, 100) == pytest.approx((0.35, 0.5, 0.5, 0.8))
    restored = PixelBox.from_yolo(box.to_yolo(200, 100), 200, 100)
    assert (restored.x1, restored.y1, restored.x2, restored.y2) == pytest.approx(
        (box.x1, box.y1, box.x2, box.y2)
    )


def test_invalid_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive area"):
        PixelBox(10, 10, 10, 30)
