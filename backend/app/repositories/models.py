import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    images: Mapped[list["ImageRow"]] = relationship(cascade="all, delete-orphan")


class ImageRow(Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    relative_path: Mapped[str] = mapped_column(String, unique=True)
    original_name: Mapped[str] = mapped_column(String(300))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    phash: Mapped[str] = mapped_column(String(32), index=True)
    review_state: Mapped[str] = mapped_column(String(32), default="needs_review", index=True)
    imported_class: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    annotations: Mapped[list["AnnotationRow"]] = relationship(cascade="all, delete-orphan")


class AnnotationRow(Base):
    __tablename__ = "annotations"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    class_name: Mapped[str] = mapped_column(String(32))
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="manual")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DatasetVersionRow(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    root_path: Mapped[str] = mapped_column(String)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    split_counts: Mapped[dict] = mapped_column(JSON)
    class_counts: Mapped[dict] = mapped_column(JSON)
    identity_leakage_checked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    __table_args__ = (Index("uq_dataset_version_number", "dataset_id", "number", unique=True),)


class VersionImageRow(Base):
    __tablename__ = "version_images"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"))
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id", ondelete="RESTRICT"))
    split: Mapped[str] = mapped_column(String(8), index=True)
    frozen_annotations: Mapped[list] = mapped_column(JSON)


class DataSourceRow(Base):
    __tablename__ = "data_sources"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    source_key: Mapped[str] = mapped_column(String(100))
    source_version: Mapped[str] = mapped_column(String(50))
    license_id: Mapped[str] = mapped_column(String(100))
    acceptance_snapshot: Mapped[dict] = mapped_column(JSON)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingTaskRow(Base):
    __tablename__ = "training_tasks"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    state: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    settings: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    resumed_from_id: Mapped[str | None] = mapped_column(ForeignKey("training_tasks.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingEventRow(Base):
    __tablename__ = "training_events"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id", ondelete="CASCADE"))
    sequence: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    __table_args__ = (Index("uq_training_event_sequence", "task_id", "sequence", unique=True),)


class ModelRow(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    training_task_id: Mapped[str | None] = mapped_column(ForeignKey("training_tasks.id"))
    name: Mapped[str] = mapped_column(String(200))
    onnx_path: Mapped[str] = mapped_column(String)
    onnx_sha256: Mapped[str] = mapped_column(String(64))
    class_names: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    quality_status: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DetectionRecordRow(Base):
    __tablename__ = "detection_records"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    source: Mapped[str] = mapped_column(String(16), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), index=True)
    original_path: Mapped[str] = mapped_column(String)
    annotated_path: Mapped[str] = mapped_column(String)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[float] = mapped_column(Float)
    device: Mapped[str] = mapped_column(String(64))
    delete_state: Mapped[str] = mapped_column(String(16), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    detections: Mapped[list["DetectionItemRow"]] = relationship(cascade="all, delete-orphan")


class DetectionItemRow(Base):
    __tablename__ = "detection_items"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    record_id: Mapped[str] = mapped_column(ForeignKey("detection_records.id", ondelete="CASCADE"))
    class_name: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
