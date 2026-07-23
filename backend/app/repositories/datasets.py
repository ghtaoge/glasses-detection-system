from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.domain.geometry import PixelBox
from app.domain.labels import ClassName
from app.repositories.models import (
    AnnotationRow,
    DatasetRow,
    DatasetVersionRow,
    ImageRow,
)


@dataclass(frozen=True, slots=True)
class AnnotationData:
    id: str
    class_name: str
    box: PixelBox
    source: str


class DatasetRepository:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def create(self, name: str) -> DatasetRow:
        clean_name = name.strip()
        if not clean_name:
            raise AppError("DATASET_NAME_REQUIRED", "数据集名称不能为空")
        with self.session_factory() as session:
            row = DatasetRow(name=clean_name)
            session.add(row)
            session.commit()
            return row

    def list(self) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(select(DatasetRow).order_by(DatasetRow.created_at.desc())).all()
            result = []
            for row in rows:
                image_count = session.scalar(
                    select(func.count(ImageRow.id)).where(ImageRow.dataset_id == row.id)
                )
                pending_count = session.scalar(
                    select(func.count(ImageRow.id)).where(
                        ImageRow.dataset_id == row.id, ImageRow.review_state == "needs_review"
                    )
                )
                result.append(
                    {
                        "id": row.id,
                        "name": row.name,
                        "image_count": image_count or 0,
                        "pending_count": pending_count or 0,
                        "created_at": row.created_at,
                    }
                )
            return result

    def get(self, dataset_id: str, session: Session | None = None) -> DatasetRow:
        owns_session = session is None
        session = session or self.session_factory()
        try:
            row = session.get(DatasetRow, dataset_id)
            if row is None:
                raise AppError("DATASET_NOT_FOUND", "数据集不存在", 404)
            if owns_session:
                session.expunge(row)
            return row
        finally:
            if owns_session:
                session.close()

    def add_image(
        self,
        dataset_id: str,
        relative_path: str,
        original_name: str,
        width: int,
        height: int,
        sha256: str,
        phash: str,
        imported_class: str | None,
    ) -> tuple[ImageRow, bool]:
        with self.session_factory() as session:
            self.get(dataset_id, session)
            duplicate = session.scalar(
                select(ImageRow).where(ImageRow.dataset_id == dataset_id, ImageRow.sha256 == sha256)
            )
            if duplicate is not None:
                return duplicate, False
            row = ImageRow(
                dataset_id=dataset_id,
                relative_path=relative_path,
                original_name=original_name,
                width=width,
                height=height,
                sha256=sha256,
                phash=phash,
                imported_class=imported_class,
            )
            session.add(row)
            session.commit()
            return row, True

    def list_images(self, dataset_id: str, review_state: str | None = None) -> list[dict]:
        with self.session_factory() as session:
            query = (
                select(ImageRow)
                .where(ImageRow.dataset_id == dataset_id)
                .options(selectinload(ImageRow.annotations))
                .order_by(ImageRow.created_at, ImageRow.id)
            )
            if review_state:
                query = query.where(ImageRow.review_state == review_state)
            return [self._serialize_image(row) for row in session.scalars(query).all()]

    def get_image(self, image_id: str) -> dict:
        with self.session_factory() as session:
            row = session.scalar(
                select(ImageRow)
                .where(ImageRow.id == image_id)
                .options(selectinload(ImageRow.annotations))
            )
            if row is None:
                raise AppError("IMAGE_NOT_FOUND", "图片不存在", 404)
            return self._serialize_image(row)

    def replace_annotations(self, image_id: str, annotations: list[dict]) -> dict:
        with self.session_factory() as session:
            image = session.get(ImageRow, image_id)
            if image is None:
                raise AppError("IMAGE_NOT_FOUND", "图片不存在", 404)
            session.query(AnnotationRow).filter(AnnotationRow.image_id == image_id).delete()
            for annotation in annotations:
                try:
                    class_name = ClassName(annotation["class_name"])
                    box = PixelBox(**annotation["box"]).clamp(image.width, image.height)
                except (KeyError, TypeError, ValueError) as exc:
                    raise AppError("ANNOTATION_INVALID", "标注类别或坐标无效") from exc
                session.add(
                    AnnotationRow(
                        image_id=image_id,
                        class_name=class_name.value,
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        source=annotation.get("source", "manual"),
                    )
                )
            image.review_state = "reviewed" if annotations else "needs_review"
            session.commit()
        return self.get_image(image_id)

    def versions(self, dataset_id: str) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(DatasetVersionRow)
                .where(DatasetVersionRow.dataset_id == dataset_id)
                .order_by(DatasetVersionRow.number.desc())
            ).all()
            return [self._serialize_version(row) for row in rows]

    def version(self, version_id: str) -> dict:
        with self.session_factory() as session:
            row = session.get(DatasetVersionRow, version_id)
            if row is None:
                raise AppError("DATASET_VERSION_NOT_FOUND", "数据集版本不存在", 404)
            return self._serialize_version(row)

    @staticmethod
    def _serialize_image(row: ImageRow) -> dict:
        return {
            "id": row.id,
            "dataset_id": row.dataset_id,
            "relative_path": row.relative_path,
            "url": f"/api/files/{row.relative_path}",
            "original_name": row.original_name,
            "width": row.width,
            "height": row.height,
            "sha256": row.sha256,
            "phash": row.phash,
            "review_state": row.review_state,
            "imported_class": row.imported_class,
            "annotations": [
                {
                    "id": annotation.id,
                    "class_name": annotation.class_name,
                    "box": {
                        "x1": annotation.x1,
                        "y1": annotation.y1,
                        "x2": annotation.x2,
                        "y2": annotation.y2,
                    },
                    "source": annotation.source,
                }
                for annotation in row.annotations
            ],
        }

    @staticmethod
    def _serialize_version(row: DatasetVersionRow) -> dict:
        return {
            "id": row.id,
            "dataset_id": row.dataset_id,
            "number": row.number,
            "root_path": row.root_path,
            "manifest_sha256": row.manifest_sha256,
            "split_counts": row.split_counts,
            "class_counts": row.class_counts,
            "identity_leakage_checked": row.identity_leakage_checked,
            "created_at": row.created_at,
        }
