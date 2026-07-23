"""训练事件和模型注册表的同步 SQLAlchemy 仓储。"""

from __future__ import annotations

from sqlalchemy import func, select, update

from app.core.errors import AppError
from app.domain.training import ALLOWED_TRANSITIONS, TrainingState
from app.repositories.models import ModelRow, TrainingEventRow, TrainingTaskRow, utc_now


class TrainingRepository:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def create(self, version_id: str, settings: dict, resumed_from_id: str | None = None) -> dict:
        with self.session_factory() as session:
            active = session.scalar(
                select(func.count(TrainingTaskRow.id)).where(
                    TrainingTaskRow.state.in_(["queued", "running", "cancelling"])
                )
            )
            # 本地工作站只允许一个活动训练，避免两个 worker 同时争用 GPU 和产物目录。
            if active:
                raise AppError("TRAINING_ALREADY_ACTIVE", "已有训练任务正在运行", 409)
            row = TrainingTaskRow(
                dataset_version_id=version_id,
                settings=settings,
                resumed_from_id=resumed_from_id,
            )
            session.add(row)
            session.commit()
            return self.serialize_task(row)

    def get(self, task_id: str) -> dict:
        with self.session_factory() as session:
            row = session.get(TrainingTaskRow, task_id)
            if row is None:
                raise AppError("TRAINING_TASK_NOT_FOUND", "训练任务不存在", 404)
            return self.serialize_task(row)

    def list(self) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(TrainingTaskRow).order_by(TrainingTaskRow.created_at.desc())
            ).all()
            return [self.serialize_task(row) for row in rows]

    def transition(
        self,
        task_id: str,
        target: TrainingState,
        result: dict | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict:
        with self.session_factory() as session:
            row = session.get(TrainingTaskRow, task_id)
            if row is None:
                raise AppError("TRAINING_TASK_NOT_FOUND", "训练任务不存在", 404)
            current = TrainingState(row.state)
            # 所有调用方都必须经过同一状态图，防止取消中的任务被误标为完成。
            if target not in ALLOWED_TRANSITIONS[current]:
                raise AppError(
                    "TRAINING_STATE_INVALID", f"不能从 {current.value} 切换到 {target.value}", 409
                )
            row.state = target.value
            row.result = result if result is not None else row.result
            row.error_code = error_code
            row.error_message = error_message
            row.updated_at = utc_now()
            session.commit()
            return self.serialize_task(row)

    def append_event(self, task_id: str, event_type: str, payload: dict) -> dict:
        with self.session_factory() as session:
            # sequence 是任务内递增游标，SSE 客户端可用 Last-Event-ID 断点续传。
            sequence = (
                session.scalar(
                    select(func.max(TrainingEventRow.sequence)).where(
                        TrainingEventRow.task_id == task_id
                    )
                )
                or 0
            ) + 1
            row = TrainingEventRow(
                task_id=task_id, sequence=sequence, event_type=event_type, payload=payload
            )
            session.add(row)
            session.commit()
            return self.serialize_event(row)

    def events_after(self, task_id: str, sequence: int) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(TrainingEventRow)
                .where(
                    TrainingEventRow.task_id == task_id,
                    TrainingEventRow.sequence > sequence,
                )
                .order_by(TrainingEventRow.sequence)
            ).all()
            return [self.serialize_event(row) for row in rows]

    def interrupt_active(self) -> int:
        with self.session_factory() as session:
            result = session.execute(
                update(TrainingTaskRow)
                .where(TrainingTaskRow.state.in_(["queued", "running", "cancelling"]))
                .values(
                    state="interrupted",
                    error_code="PROCESS_INTERRUPTED",
                    error_message="服务重启导致训练中断",
                    updated_at=utc_now(),
                )
            )
            session.commit()
            return result.rowcount

    @staticmethod
    def serialize_task(row: TrainingTaskRow) -> dict:
        return {
            "id": row.id,
            "dataset_version_id": row.dataset_version_id,
            "state": row.state,
            "settings": row.settings,
            "result": row.result,
            "error_code": row.error_code,
            "error_message": row.error_message,
            "resumed_from_id": row.resumed_from_id,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    @staticmethod
    def serialize_event(row: TrainingEventRow) -> dict:
        return {
            "sequence": row.sequence,
            "event_type": row.event_type,
            "payload": row.payload,
            "created_at": row.created_at,
        }


class ModelRepository:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def create(
        self,
        task_id: str | None,
        name: str,
        onnx_path: str,
        onnx_sha256: str,
        class_names: list[str],
        metrics: dict,
        quality_status: str | None = None,
    ) -> dict:
        with self.session_factory() as session:
            row = ModelRow(
                training_task_id=task_id,
                name=name,
                onnx_path=onnx_path,
                onnx_sha256=onnx_sha256,
                class_names=class_names,
                metrics=metrics,
                quality_status=quality_status
                or ("passed" if metrics.get("map50", 0) >= 0.80 else "below_target"),
            )
            session.add(row)
            session.commit()
            return self.serialize(row)

    def upsert_pretrained(
        self,
        name: str,
        onnx_path: str,
        onnx_sha256: str,
        class_names: list[str],
        metrics: dict,
    ) -> dict:
        """Create or refresh the single application-owned pretrained model record."""

        with self.session_factory() as session:
            row = session.scalar(select(ModelRow).where(ModelRow.onnx_path == onnx_path))
            if row is None:
                row = ModelRow(training_task_id=None, onnx_path=onnx_path)
                session.add(row)
            row.name = name
            row.onnx_sha256 = onnx_sha256
            row.class_names = class_names
            row.metrics = metrics
            row.quality_status = "pretrained"
            session.commit()
            return self.serialize(row)

    def list(self) -> list[dict]:
        with self.session_factory() as session:
            return [
                self.serialize(row)
                for row in session.scalars(
                    select(ModelRow).order_by(ModelRow.created_at.desc())
                ).all()
            ]

    def get(self, model_id: str) -> dict:
        with self.session_factory() as session:
            row = session.get(ModelRow, model_id)
            if row is None:
                raise AppError("MODEL_NOT_FOUND", "模型不存在", 404)
            return self.serialize(row)

    def activate(self, model_id: str) -> dict:
        with self.session_factory() as session:
            row = session.get(ModelRow, model_id)
            if row is None:
                raise AppError("MODEL_NOT_FOUND", "模型不存在", 404)
            # 清除旧标记与启用新模型位于同一事务，外部永远不会观察到两个活动模型。
            session.execute(update(ModelRow).values(active=False))
            row.active = True
            session.commit()
            return self.serialize(row)

    def active(self) -> dict | None:
        with self.session_factory() as session:
            row = session.scalar(select(ModelRow).where(ModelRow.active.is_(True)))
            return self.serialize(row) if row else None

    @staticmethod
    def serialize(row: ModelRow) -> dict:
        return {
            "id": row.id,
            "training_task_id": row.training_task_id,
            "name": row.name,
            "onnx_path": row.onnx_path,
            "onnx_sha256": row.onnx_sha256,
            "class_names": row.class_names,
            "metrics": row.metrics,
            "quality_status": row.quality_status,
            "is_active": row.active,
            "created_at": row.created_at,
        }
