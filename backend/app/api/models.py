import hashlib

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.core.errors import AppError

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def list_models(request: Request) -> list[dict]:
    return request.app.state.models.list()


@router.post("/models/pretrained/install", status_code=201)
def install_pretrained_model(request: Request) -> dict:
    """Install and activate the vetted MIT-licensed local inference pipeline."""

    return request.app.state.pretrained_models.install()


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request) -> dict:
    return request.app.state.models.get(model_id)


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str, request: Request) -> dict:
    model = request.app.state.models.get(model_id)
    if model["quality_status"] not in {"passed", "pretrained"}:
        raise AppError("MODEL_QUALITY_BELOW_TARGET", "模型未达到 mAP@0.5 质量门槛", 409)
    path = request.app.state.storage.resolve(model["onnx_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != model["onnx_sha256"]:
        raise AppError("MODEL_CHECKSUM_MISMATCH", "模型文件校验失败", 409)
    if path.name.endswith(".fake.onnx"):
        # fake_training 允许快速验证训练流程，但不代表产物具有推理能力。
        # 只有后端测试显式打开 fake_inference 时，才允许激活固定输出引擎。
        if not request.app.state.fake_inference:
            raise AppError(
                "SIMULATED_MODEL_NOT_FOR_INFERENCE",
                "模拟测试模型不能用于真实识别，请训练真实 ONNX 模型",
                409,
            )
    else:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            if model["metrics"].get("engine") == "face_glasses_classifier_v1":
                if not request.app.state.resources.valid("yunet"):
                    raise AppError("FACE_MODEL_UNAVAILABLE", "YuNet 人脸模型不可用", 409)
                if session.get_outputs()[0].shape[-1] != 2:
                    raise AppError("MODEL_LOAD_FAILED", "公开分类模型输出无效", 409)
        except Exception as exc:
            if isinstance(exc, AppError):
                raise
            raise AppError("MODEL_LOAD_FAILED", "ONNX 模型加载失败", 409) from exc
    return request.app.state.models.activate(model_id)


@router.get("/models/{model_id}/download")
def download_model(model_id: str, request: Request) -> FileResponse:
    model = request.app.state.models.get(model_id)
    path = request.app.state.storage.resolve(model["onnx_path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != model["onnx_sha256"]:
        raise AppError("MODEL_CHECKSUM_MISMATCH", "模型文件校验失败", 409)
    return FileResponse(path, filename=f"glasses-model-{model_id[:8]}.onnx")
