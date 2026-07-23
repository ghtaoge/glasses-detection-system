import io
import uuid

import cv2
import numpy as np
from PIL import Image

from app.core.errors import AppError
from app.inference.fake import FakeInferenceEngine
from app.inference.onnx import OnnxInferenceEngine
from app.services.rendering import render_detections


class ImageInferenceService:
    def __init__(self, models, history, storage, validator) -> None:
        self.models = models
        self.history = history
        self.storage = storage
        self.validator = validator

    def run(
        self,
        filename: str,
        content: bytes,
        confidence: float = 0.25,
        iou: float = 0.45,
        source: str = "image",
    ) -> dict:
        validated, result = self.infer_only(filename, content, confidence, iou)
        result_data = result.as_dict()
        record_id = str(uuid.uuid4())
        extension = validated.extension
        original_relative = f"history/{record_id}/original{extension}"
        annotated_relative = f"history/{record_id}/annotated.jpg"
        annotated = render_detections(Image.open(io.BytesIO(content)), result.detections)
        self.storage.publish_bytes(original_relative, content)
        try:
            self.storage.publish_bytes(annotated_relative, annotated)
            return self.history.create(result_data, source, original_relative, annotated_relative)
        except BaseException:
            for relative in (original_relative, annotated_relative):
                path = self.storage.resolve(relative)
                if path.exists():
                    path.unlink()
            raise

    def infer_only(
        self, filename: str, content: bytes, confidence: float = 0.25, iou: float = 0.45
    ):
        validated = self.validator.validate_image(filename, content)
        model = self.models.active()
        if model is None:
            raise AppError("ACTIVE_MODEL_REQUIRED", "请先在模型库启用一个达标模型", 409)
        image_array = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            raise AppError("IMAGE_DECODE_FAILED", "图片解码失败")
        model_path = self.storage.resolve(model["onnx_path"])
        if model_path.name.endswith(".fake.onnx"):
            engine = FakeInferenceEngine(model["id"])
        else:
            try:
                import onnxruntime as ort

                session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                input_shape = session.get_inputs()[0].shape
                input_size = int(input_shape[-1]) if isinstance(input_shape[-1], int) else 640
                engine = OnnxInferenceEngine(session, model["id"], input_size)
            except Exception as exc:
                raise AppError("MODEL_LOAD_FAILED", "当前 ONNX 模型无法加载", 409) from exc
        return validated, engine.infer(image_array, confidence, iou)


class HistoryService:
    def __init__(self, repository, storage) -> None:
        self.repository = repository
        self.storage = storage

    def delete(self, record_id: str) -> None:
        record = self.repository.mark_pending(record_id)
        try:
            for key in ("original_path", "annotated_path"):
                path = self.storage.resolve(record[key])
                if path.exists():
                    path.unlink()
        except OSError as exc:
            raise AppError("HISTORY_DELETE_INCOMPLETE", "文件删除未完成，可稍后重试", 500) from exc
        self.repository.remove(record_id)
