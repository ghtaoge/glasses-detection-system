import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.domain.labels import ClassName

router = APIRouter(prefix="/api", tags=["datasets"])


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PublishRequest(BaseModel):
    seed: int = 20260723


@router.get("/data-sources")
def data_sources() -> list[dict]:
    return [
        {
            "key": "kaggle-glasses-coverings-v2",
            "title": "Glasses and Coverings",
            "version": "2",
            "license_id": "CC-BY-NC-4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "page_url": "https://www.kaggle.com/datasets/mantasu/glasses-and-coverings",
            "non_commercial_only": True,
        }
    ]


@router.post("/datasets", status_code=201)
def create_dataset(payload: DatasetCreate, request: Request) -> dict:
    row = request.app.state.datasets.create(payload.name)
    return {"id": row.id, "name": row.name, "image_count": 0, "pending_count": 0}


@router.get("/datasets")
def list_datasets(request: Request) -> list[dict]:
    return request.app.state.datasets.list()


def _import_entries(
    request: Request,
    dataset_id: str,
    entries: list[tuple[str, bytes]],
    forced_class: str | None = None,
) -> dict:
    mapping = {
        "plain": "no_glasses",
        "glasses": "eyeglasses",
        "sunglasses": "sunglasses",
        "sunglasses-imagenet": "sunglasses",
    }
    counts = {"imported": 0, "duplicate": 0, "invalid": 0}
    for name, content in entries:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        folder = PurePosixPath(name).parts[-2] if len(PurePosixPath(name).parts) > 1 else ""
        class_name = forced_class or mapping.get(folder)
        try:
            validated = request.app.state.validator.validate_image(
                PurePosixPath(name).name, content
            )
        except AppError:
            counts["invalid"] += 1
            continue
        relative = f"datasets/{dataset_id}/images/{validated.sha256}{validated.extension}"
        request.app.state.storage.publish_bytes(relative, validated.content)
        _, created = request.app.state.datasets.add_image(
            dataset_id,
            relative,
            PurePosixPath(name).name,
            validated.width,
            validated.height,
            validated.sha256,
            validated.phash,
            class_name,
        )
        counts["imported" if created else "duplicate"] += 1
    return counts


@router.post("/datasets/{dataset_id}/imports/images")
async def import_images(
    request: Request,
    dataset_id: str,
    files: Annotated[list[UploadFile], File()],
    class_name: Annotated[ClassName | None, Form()] = None,
) -> dict:
    entries = [(file.filename or "image", await file.read()) for file in files]
    return _import_entries(request, dataset_id, entries, class_name.value if class_name else None)


@router.post("/datasets/{dataset_id}/imports/archive")
async def import_archive(
    request: Request, dataset_id: str, file: Annotated[UploadFile, File()]
) -> dict:
    entries = request.app.state.validator.archive_entries(await file.read())
    return _import_entries(request, dataset_id, entries)


@router.post("/datasets/{dataset_id}/imports/source")
def import_source(
    dataset_id: str,
    request: Request,
    accepted_license: str = Form(...),
    non_commercial_teaching: bool = Form(...),
) -> dict:
    if accepted_license != "CC-BY-NC-4.0" or not non_commercial_teaching:
        raise AppError("LICENSE_ACCEPTANCE_REQUIRED", "必须确认非商业教学许可")
    with tempfile.TemporaryDirectory() as directory:
        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "mantasu/glasses-and-coverings",
            "-p",
            directory,
            "--unzip",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        except (subprocess.SubprocessError, OSError) as exc:
            raise AppError(
                "DATA_SOURCE_DOWNLOAD_FAILED", "公开数据下载失败，请检查 Kaggle 凭据", 502
            ) from exc
        entries = [
            (path.relative_to(directory).as_posix(), path.read_bytes())
            for path in Path(directory).rglob("*")
            if path.is_file()
        ]
        return _import_entries(request, dataset_id, entries)


@router.get("/datasets/{dataset_id}/images")
def list_images(dataset_id: str, request: Request, review_state: str | None = None) -> list[dict]:
    return request.app.state.datasets.list_images(dataset_id, review_state)


@router.get("/datasets/{dataset_id}/publication-check")
def publication_check(dataset_id: str, request: Request) -> dict:
    return request.app.state.publisher.check(dataset_id)


@router.post("/datasets/{dataset_id}/versions", status_code=201)
def publish(dataset_id: str, payload: PublishRequest, request: Request) -> dict:
    return request.app.state.publisher.publish(dataset_id, payload.seed)


@router.get("/datasets/{dataset_id}/versions")
def versions(dataset_id: str, request: Request) -> list[dict]:
    return request.app.state.datasets.versions(dataset_id)


@router.get("/dataset-versions/{version_id}")
def version(version_id: str, request: Request) -> dict:
    return request.app.state.datasets.version(version_id)
