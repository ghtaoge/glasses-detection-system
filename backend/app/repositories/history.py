from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.core.errors import AppError
from app.repositories.models import DetectionItemRow, DetectionRecordRow


class HistoryRepository:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def create(self, result: dict, source: str, original_path: str, annotated_path: str) -> dict:
        with self.session_factory() as session:
            row = DetectionRecordRow(
                source=source,
                model_id=result["model_id"],
                original_path=original_path,
                annotated_path=annotated_path,
                width=result["width"],
                height=result["height"],
                duration_ms=result["duration_ms"],
                device=result["device"],
            )
            for item in result["detections"]:
                box = item["box"]
                row.detections.append(
                    DetectionItemRow(
                        class_name=item["class_name"],
                        confidence=item["confidence"],
                        x1=box["x1"],
                        y1=box["y1"],
                        x2=box["x2"],
                        y2=box["y2"],
                    )
                )
            session.add(row)
            session.commit()
            return self.serialize(row)

    def list(
        self,
        source: str | None = None,
        class_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 50,
    ) -> list[dict]:
        with self.session_factory() as session:
            query = (
                select(DetectionRecordRow)
                .where(DetectionRecordRow.delete_state == "active")
                .options(selectinload(DetectionRecordRow.detections))
                .order_by(DetectionRecordRow.created_at.desc(), DetectionRecordRow.id.desc())
                .limit(limit)
            )
            if source:
                query = query.where(DetectionRecordRow.source == source)
            if class_name:
                query = query.join(DetectionItemRow).where(
                    DetectionItemRow.class_name == class_name
                )
            if date_from:
                query = query.where(DetectionRecordRow.created_at >= date_from)
            if date_to:
                query = query.where(DetectionRecordRow.created_at <= date_to)
            return [self.serialize(row) for row in session.scalars(query).unique().all()]

    def get(self, record_id: str, include_pending: bool = False) -> dict:
        with self.session_factory() as session:
            row = session.scalar(
                select(DetectionRecordRow)
                .where(DetectionRecordRow.id == record_id)
                .options(selectinload(DetectionRecordRow.detections))
            )
            if row is None or (row.delete_state != "active" and not include_pending):
                raise AppError("HISTORY_NOT_FOUND", "检测记录不存在", 404)
            return self.serialize(row)

    def mark_pending(self, record_id: str) -> dict:
        with self.session_factory() as session:
            row = session.get(DetectionRecordRow, record_id)
            if row is None:
                raise AppError("HISTORY_NOT_FOUND", "检测记录不存在", 404)
            row.delete_state = "pending"
            session.commit()
        return self.get(record_id, include_pending=True)

    def remove(self, record_id: str) -> None:
        with self.session_factory() as session:
            session.execute(delete(DetectionItemRow).where(DetectionItemRow.record_id == record_id))
            session.execute(delete(DetectionRecordRow).where(DetectionRecordRow.id == record_id))
            session.commit()

    def counts(self) -> dict:
        rows = self.list(limit=10000)
        return {
            "total": len(rows),
            "image": sum(row["source"] == "image" for row in rows),
            "camera": sum(row["source"] == "camera" for row in rows),
        }

    @staticmethod
    def serialize(row: DetectionRecordRow) -> dict:
        return {
            "id": row.id,
            "source": row.source,
            "model_id": row.model_id,
            "original_path": row.original_path,
            "annotated_path": row.annotated_path,
            "original_url": f"/api/history/{row.id}/original",
            "annotated_url": f"/api/history/{row.id}/annotated",
            "width": row.width,
            "height": row.height,
            "duration_ms": row.duration_ms,
            "device": row.device,
            "delete_state": row.delete_state,
            "created_at": row.created_at,
            "detections": [
                {
                    "id": item.id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "box": {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2},
                }
                for item in row.detections
            ],
        }
