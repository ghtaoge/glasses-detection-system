import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

import imagehash
from PIL import Image, UnidentifiedImageError

from app.core.errors import AppError


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    content: bytes
    extension: str
    width: int
    height: int
    sha256: str
    phash: str


class UploadValidator:
    _formats = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

    def __init__(self, max_bytes: int, max_pixels: int) -> None:
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels

    def validate_image(self, filename: str, content: bytes) -> ValidatedImage:
        if not content or len(content) > self.max_bytes:
            raise AppError("IMAGE_SIZE_INVALID", "图片为空或超过大小限制", 413)
        try:
            with Image.open(io.BytesIO(content)) as probe:
                detected_format = probe.format
                probe.verify()
            with Image.open(io.BytesIO(content)) as decoded:
                decoded.load()
                width, height = decoded.size
                if width * height > self.max_pixels:
                    raise AppError("IMAGE_PIXELS_EXCEEDED", "图片像素数量超过限制", 413)
                if detected_format not in self._formats:
                    raise AppError("IMAGE_FORMAT_UNSUPPORTED", "仅支持 JPG、PNG 和 WebP")
                rgb = decoded.convert("RGB")
                phash = str(imagehash.phash(rgb))
        except AppError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise AppError("IMAGE_SIGNATURE_INVALID", "图片内容无效") from exc
        return ValidatedImage(
            content=content,
            extension=self._formats[detected_format],
            width=width,
            height=height,
            sha256=hashlib.sha256(content).hexdigest(),
            phash=phash,
        )

    def archive_entries(self, content: bytes) -> list[tuple[str, bytes]]:
        if len(content) > 512 * 1024 * 1024:
            raise AppError("ARCHIVE_SIZE_INVALID", "压缩包超过大小限制", 413)
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise AppError("ARCHIVE_INVALID", "压缩包损坏") from exc
        entries: list[tuple[str, bytes]] = []
        total = 0
        for info in archive.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if info.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
                raise AppError("ARCHIVE_PATH_INVALID", "压缩包包含非法路径")
            if len(entries) >= 5_000:
                raise AppError("ARCHIVE_ENTRY_LIMIT", "压缩包文件数量超过限制")
            total += info.file_size
            if total > 2 * 1024 * 1024 * 1024:
                raise AppError("ARCHIVE_EXPANDED_LIMIT", "压缩包解压大小超过限制")
            if info.compress_size and info.file_size / info.compress_size > 100:
                raise AppError("ARCHIVE_RATIO_INVALID", "压缩比异常")
            entries.append((path.as_posix(), archive.read(info)))
        return entries
