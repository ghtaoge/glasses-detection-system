import os
import tempfile
from pathlib import Path, PurePosixPath

from app.core.errors import AppError


class ManagedStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        if not relative or PurePosixPath(relative).is_absolute():
            raise AppError("PATH_OUTSIDE_MANAGED_ROOT", "非法存储路径")
        candidate = (self.root / Path(*PurePosixPath(relative).parts)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise AppError("PATH_OUTSIDE_MANAGED_ROOT", "非法存储路径")
        return candidate

    def relative(self, path: Path) -> str:
        resolved = path.resolve()
        if self.root not in resolved.parents:
            raise AppError("PATH_OUTSIDE_MANAGED_ROOT", "文件不在受控目录中")
        return resolved.relative_to(self.root).as_posix()

    def publish_bytes(self, relative: str, content: bytes) -> Path:
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".partial", delete=False
            ) as handle:
                partial = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, target)
        finally:
            if partial is not None and partial.exists():
                partial.unlink()
        return target
