from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

router = APIRouter(prefix="/api", tags=["inference"])


@router.post("/inference/image", status_code=201)
async def infer_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form(ge=0.05, le=0.95)] = 0.25,
    iou: Annotated[float, Form(ge=0.1, le=0.9)] = 0.45,
) -> dict:
    return request.app.state.inference.run(
        file.filename or "image", await file.read(), confidence, iou, "image"
    )


@router.post("/camera/snapshots", status_code=201)
async def save_camera_snapshot(
    request: Request,
    frame: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form(ge=0.05, le=0.95)] = 0.25,
) -> dict:
    return request.app.state.inference.run(
        frame.filename or "snapshot.jpg", await frame.read(), confidence, 0.45, "camera"
    )
