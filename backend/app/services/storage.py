"""受控数据根目录及原子文件发布工具。"""

import os
import tempfile
from pathlib import Path, PurePosixPath

from app.core.errors import AppError


class ManagedStorage:
    """确保数据库中的相对路径永远不能逃逸应用数据目录。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        if not relative or PurePosixPath(relative).is_absolute():
            raise AppError("PATH_OUTSIDE_MANAGED_ROOT", "非法存储路径")
        # resolve 会折叠 ``..`` 和符号链接；随后用父目录关系做最终边界检查。
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
        """在目标目录写临时文件、刷盘，再以原子替换发布完整内容。"""

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
            # 临时文件与目标位于同一目录，因此 os.replace 不会跨文件系统退化。
            os.replace(partial, target)
        finally:
            if partial is not None and partial.exists():
                partial.unlink()
        return target
