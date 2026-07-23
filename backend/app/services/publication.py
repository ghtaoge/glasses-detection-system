"""把可编辑标注发布为不可变、可复现的 YOLO 数据版本。"""

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.core.errors import AppError
from app.domain.geometry import PixelBox
from app.domain.labels import CLASS_NAMES, ClassName, class_id
from app.repositories.models import (
    DatasetRow,
    DatasetVersionRow,
    ImageRow,
    VersionImageRow,
)
from app.services.storage import ManagedStorage


class DatasetPublisher:
    """校验草稿数据并原子发布图片、标签、清单和数据集元数据。"""

    def __init__(self, session_factory, storage: ManagedStorage, min_per_class: int = 10) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.min_per_class = min_per_class

    def check(self, dataset_id: str) -> dict:
        with self.session_factory() as session:
            dataset = session.get(DatasetRow, dataset_id)
            if dataset is None:
                raise AppError("DATASET_NOT_FOUND", "数据集不存在", 404)
            images = session.scalars(
                select(ImageRow)
                .where(ImageRow.dataset_id == dataset_id)
                .options(selectinload(ImageRow.annotations))
            ).all()
        errors: list[dict] = []
        counts = Counter()
        pending = 0
        for image in images:
            if image.review_state == "needs_review" or not image.annotations:
                pending += 1
            for annotation in image.annotations:
                counts[annotation.class_name] += 1
        if pending:
            errors.append(
                {"code": "DATASET_REVIEW_INCOMPLETE", "message": f"仍有 {pending} 张图片待复核"}
            )
        for name in CLASS_NAMES:
            if counts[name.value] < self.min_per_class:
                errors.append(
                    {
                        "code": "CLASS_SAMPLE_INSUFFICIENT",
                        "message": f"{name.value} 至少需要 {self.min_per_class} 个标注",
                    }
                )
        return {
            "ready": not errors,
            "errors": errors,
            "warnings": [
                {"code": "IDENTITY_LEAKAGE_UNCHECKED", "message": "源数据没有人物身份标签"}
            ],
            "image_count": len(images),
            "class_counts": dict(counts),
            "pending_count": pending,
        }

    def publish(self, dataset_id: str, seed: int = 20260723) -> dict:
        validation = self.check(dataset_id)
        if not validation["ready"]:
            raise AppError("DATASET_NOT_READY", "数据集未通过发布检查")
        with self.session_factory() as session:
            images = session.scalars(
                select(ImageRow)
                .where(ImageRow.dataset_id == dataset_id)
                .options(selectinload(ImageRow.annotations))
                .order_by(ImageRow.id)
            ).all()
            number = (
                session.scalar(
                    select(func.max(DatasetVersionRow.number)).where(
                        DatasetVersionRow.dataset_id == dataset_id
                    )
                )
                or 0
            ) + 1
            version_id = __import__("uuid").uuid4().hex
            relative_root = f"datasets/{dataset_id}/versions/{version_id}"
            root = self.storage.resolve(relative_root)
            # 文件先写入同一文件系统中的 staging 目录。全部成功后再 rename，避免
            # 训练进程观察到只包含部分图片或标签的版本目录。
            staging = root.with_name(root.name + ".partial")
            if staging.exists():
                shutil.rmtree(staging)
            split_map = self._split(images, seed)
            manifest_items = []
            split_counts = Counter()
            try:
                for image in images:
                    split = split_map[image.id]
                    split_counts[split] += 1
                    source = self.storage.resolve(image.relative_path)
                    image_target = staging / "images" / split / f"{image.id}{source.suffix.lower()}"
                    label_target = staging / "labels" / split / f"{image.id}.txt"
                    image_target.parent.mkdir(parents=True, exist_ok=True)
                    label_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, image_target)
                    frozen = []
                    lines = []
                    for annotation in image.annotations:
                        box = PixelBox(annotation.x1, annotation.y1, annotation.x2, annotation.y2)
                        values = box.to_yolo(image.width, image.height)
                        lines.append(
                            f"{class_id(ClassName(annotation.class_name))} "
                            + " ".join(f"{value:.6f}" for value in values)
                        )
                        frozen.append(
                            {
                                "class_name": annotation.class_name,
                                "box": {
                                    "x1": annotation.x1,
                                    "y1": annotation.y1,
                                    "x2": annotation.x2,
                                    "y2": annotation.y2,
                                },
                            }
                        )
                    label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    manifest_items.append(
                        {
                            "image_id": image.id,
                            "split": split,
                            "sha256": image.sha256,
                            "annotations": frozen,
                        }
                    )
                yaml = (
                    "path: .\n"
                    "train: images/train\n"
                    "val: images/val\n"
                    "test: images/test\n"
                    "names:\n"
                    "  0: no_glasses\n"
                    "  1: eyeglasses\n"
                    "  2: sunglasses\n"
                )
                (staging / "data.yaml").write_text(yaml, encoding="utf-8")
                manifest = {
                    "dataset_id": dataset_id,
                    "version": number,
                    "seed": seed,
                    "identity_leakage_checked": False,
                    "items": manifest_items,
                }
                # 清单摘要基于排序、无多余空格的规范 JSON，与展示用缩进文件分离。
                # 相同数据、标注、划分和 seed 会得到相同摘要。
                canonical = json.dumps(
                    manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                digest = hashlib.sha256(canonical.encode()).hexdigest()
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                root.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(root)
                version = DatasetVersionRow(
                    id=version_id,
                    dataset_id=dataset_id,
                    number=number,
                    root_path=relative_root,
                    manifest_sha256=digest,
                    split_counts=dict(split_counts),
                    class_counts=validation["class_counts"],
                    identity_leakage_checked=False,
                )
                session.add(version)
                for item in manifest_items:
                    session.add(
                        VersionImageRow(
                            version_id=version_id,
                            image_id=item["image_id"],
                            split=item["split"],
                            frozen_annotations=item["annotations"],
                        )
                    )
                session.commit()
            except BaseException:
                if staging.exists():
                    shutil.rmtree(staging)
                if root.exists():
                    shutil.rmtree(root)
                raise
        return {
            "id": version_id,
            "dataset_id": dataset_id,
            "number": number,
            "root_path": relative_root,
            "manifest_sha256": digest,
            "split_counts": dict(split_counts),
            "class_counts": validation["class_counts"],
            "identity_leakage_checked": False,
        }

    @staticmethod
    def _split(images: list[ImageRow], seed: int) -> dict[str, str]:
        """按感知哈希分组划分，防止近重复图片泄漏到不同集合。"""

        groups: dict[str, list[ImageRow]] = defaultdict(list)
        for image in images:
            groups[image.phash].append(image)
        ordered = list(groups.values())
        random.Random(seed).shuffle(ordered)
        result: dict[str, str] = {}
        total = len(images)
        targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
        counts = Counter()
        for group in ordered:
            # 整组放入当前缺口最大的集合；近重复组绝不会被拆开。
            split = max(targets, key=lambda name: targets[name] - counts[name])
            for image in group:
                result[image.id] = split
            counts[split] += len(group)
        return result
