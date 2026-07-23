import cv2
from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.errors import AppError
from app.services.yunet import YuNetFaceDetector

router = APIRouter(prefix="/api", tags=["annotations"])


class BoxPayload(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationPayload(BaseModel):
    class_name: str
    box: BoxPayload
    source: str = "manual"


@router.get("/images/{image_id}")
def get_image(image_id: str, request: Request) -> dict:
    return request.app.state.datasets.get_image(image_id)


@router.put("/images/{image_id}/annotations")
def replace_annotations(image_id: str, payload: list[AnnotationPayload], request: Request) -> dict:
    return request.app.state.datasets.replace_annotations(
        image_id, [item.model_dump() for item in payload]
    )


@router.post("/datasets/{dataset_id}/preannotate")
def preannotate(dataset_id: str, request: Request) -> dict:
    resources = request.app.state.resources
    if not resources.valid("yunet"):
        raise AppError("PREANNOTATION_MODEL_UNAVAILABLE", "请先下载 YuNet 预标注模型", 409)
    detector = YuNetFaceDetector(resources.path("yunet"))
    updated = 0
    for image in request.app.state.datasets.list_images(dataset_id):
        if image["review_state"] != "needs_review" or not image["imported_class"]:
            continue
        frame = cv2.imread(str(request.app.state.storage.resolve(image["relative_path"])))
        if frame is None:
            continue
        candidates = detector.detect(frame)
        annotations = [
            {
                "class_name": image["imported_class"],
                "box": {"x1": c.box.x1, "y1": c.box.y1, "x2": c.box.x2, "y2": c.box.y2},
                "source": "yunet",
            }
            for c in candidates
        ]
        request.app.state.datasets.replace_annotations(image["id"], annotations)
        updated += 1
    return {"processed": updated}


@router.get("/resources")
def resource_status(request: Request) -> list[dict]:
    return request.app.state.resources.status()


@router.post("/resources/{name}/fetch")
def fetch_resource(name: str, request: Request) -> dict:
    path = request.app.state.resources.fetch(name)
    return {"name": name, "ready": True, "filename": path.name}
