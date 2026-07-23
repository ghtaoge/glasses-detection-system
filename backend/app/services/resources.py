import hashlib
import json
import os
import tempfile
from pathlib import Path

import httpx

from app.core.errors import AppError


class ResourceService:
    def __init__(self, manifest_path: Path, target_dir: Path) -> None:
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        entry = self._entry(name)
        return self.target_dir / entry["filename"]

    def valid(self, name: str) -> bool:
        entry = self._entry(name)
        target = self.path(name)
        return (
            target.is_file()
            and target.stat().st_size == entry["bytes"]
            and self._sha256(target) == entry["sha256"]
        )

    def fetch(self, name: str) -> Path:
        entry = self._entry(name)
        target = self.path(name)
        if self.valid(name):
            return target
        partial: Path | None = None
        try:
            with httpx.stream("GET", entry["url"], follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    dir=self.target_dir, suffix=".partial", delete=False
                ) as handle:
                    partial = Path(handle.name)
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > entry["bytes"]:
                            raise AppError("RESOURCE_SIZE_MISMATCH", "资源文件大小不匹配")
                        handle.write(chunk)
            if partial.stat().st_size != entry["bytes"] or self._sha256(partial) != entry["sha256"]:
                raise AppError("RESOURCE_CHECKSUM_MISMATCH", "资源文件校验失败")
            os.replace(partial, target)
            return target
        finally:
            if partial is not None and partial.exists():
                partial.unlink()

    def status(self) -> list[dict]:
        return [
            {
                "name": name,
                "filename": entry["filename"],
                "license": entry["license"],
                "ready": self.valid(name),
            }
            for name, entry in self.manifest.items()
        ]

    def _entry(self, name: str) -> dict:
        if name not in self.manifest:
            raise AppError("RESOURCE_NOT_ALLOWED", "未知资源", 404)
        return self.manifest[name]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
