import hashlib

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.core.errors import AppError

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def list_models(request: Request) -> list[dict]:
    return request.app.state.models.list()


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request) -> dict:
    return request.app.state.models.get(model_id)


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str, request: Request) -> dict:
    model = request.app.state.models.get(model_id)
    if model["quality_status"] != "passed":
        raise AppError("MODEL_QUALITY_BELOW_TARGET", "模型未达到 mAP@0.5 质量门槛", 409)
    path = request.app.state.storage.resolve(model["onnx_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != model["onnx_sha256"]:
        raise AppError("MODEL_CHECKSUM_MISMATCH", "模型文件校验失败", 409)
    if path.name.endswith(".fake.onnx"):
        if not request.app.state.fake_training:
            raise AppError("MODEL_LOAD_FAILED", "模拟模型不能用于推理", 409)
    else:
        try:
            import onnxruntime as ort

            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise AppError("MODEL_LOAD_FAILED", "ONNX 模型加载失败", 409) from exc
    return request.app.state.models.activate(model_id)


@router.get("/models/{model_id}/download")
def download_model(model_id: str, request: Request) -> FileResponse:
    model = request.app.state.models.get(model_id)
    path = request.app.state.storage.resolve(model["onnx_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != model["onnx_sha256"]:
        raise AppError("MODEL_CHECKSUM_MISMATCH", "模型文件校验失败", 409)
    return FileResponse(path, filename=f"glasses-model-{model_id[:8]}.onnx")
