import io
import zipfile

import pytest
from app.core.errors import AppError
from app.services.storage import ManagedStorage
from app.services.uploads import UploadValidator
from PIL import Image


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(output, "PNG")
    return output.getvalue()


def test_validates_real_image() -> None:
    result = UploadValidator(1024 * 1024, 10_000).validate_image("face.png", image_bytes())
    assert (result.width, result.height, result.extension) == (32, 24, ".png")


def test_rejects_extension_spoof() -> None:
    with pytest.raises(AppError, match="图片内容无效"):
        UploadValidator(1024, 10_000).validate_image("face.jpg", b"not an image")


def test_rejects_zip_traversal() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../escape.jpg", image_bytes())
    with pytest.raises(AppError, match="非法路径"):
        UploadValidator(1024 * 1024, 10_000).archive_entries(output.getvalue())


def test_storage_rejects_parent_path(tmp_path) -> None:
    with pytest.raises(AppError):
        ManagedStorage(tmp_path).resolve("../escape.jpg")
