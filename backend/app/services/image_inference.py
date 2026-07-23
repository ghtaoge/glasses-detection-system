"""图片推理、结果发布和历史删除的应用服务。"""

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
    """加载当前活动模型，执行推理，并按需原子保存结果。"""

    def __init__(self, models, history, storage, validator, allow_fake: bool = False) -> None:
        self.models = models
        self.history = history
        self.storage = storage
        self.validator = validator
        self.allow_fake = allow_fake

    def run(
        self,
        filename: str,
        content: bytes,
        confidence: float = 0.25,
        iou: float = 0.45,
        source: str = "image",
    ) -> dict:
        """推理图片并在文件全部发布后创建历史记录。"""

        validated, result = self.infer_only(filename, content, confidence, iou)
        result_data = result.as_dict()
        record_id = str(uuid.uuid4())
        extension = validated.extension
        original_relative = f"history/{record_id}/original{extension}"
        annotated_relative = f"history/{record_id}/annotated.jpg"
        annotated = render_detections(Image.open(io.BytesIO(content)), result.detections)
        # 数据库记录必须最后写入。若结果图编码或数据库提交失败，异常分支只清理
        # 本次 record_id 对应的受控路径，不会触碰其他历史记录。
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
        """只返回内存中的结果，供摄像头流使用，不写磁盘或历史表。"""

        validated = self.validator.validate_image(filename, content)
        model = self.models.active()
        if model is None:
            raise AppError("ACTIVE_MODEL_REQUIRED", "请先在模型库启用一个达标模型", 409)
        image_array = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            raise AppError("IMAGE_DECODE_FAILED", "图片解码失败")
        model_path = self.storage.resolve(model["onnx_path"])
        # FakeInferenceEngine 返回的是用于端到端测试的固定检测框，不具备识别能力。
        # 除非测试装配显式授权，否则即使旧数据库仍把模拟模型标成 active，也必须在
        # 推理边界再次拦截，避免伪造结果进入历史记录或摄像头画面。
        if model_path.name.endswith(".fake.onnx"):
            if not self.allow_fake:
                raise AppError(
                    "SIMULATED_MODEL_NOT_FOR_INFERENCE",
                    "当前启用的是模拟测试模型，不能用于真实识别；请训练并启用真实 ONNX 模型",
                    409,
                )
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
    """以可重试的两阶段流程删除历史元数据和文件。"""

    def __init__(self, repository, storage) -> None:
        self.repository = repository
        self.storage = storage

    def delete(self, record_id: str) -> None:
        # 先标记 pending，防止删除中的记录继续出现在列表中。文件删除失败时保留
        # pending 行，后续可安全重试；文件不存在视为已经删除。
        record = self.repository.mark_pending(record_id)
        try:
            for key in ("original_path", "annotated_path"):
                path = self.storage.resolve(record[key])
                if path.exists():
                    path.unlink()
        except OSError as exc:
            raise AppError("HISTORY_DELETE_INCOMPLETE", "文件删除未完成，可稍后重试", 500) from exc
        # 只有两个受控文件都处理完后，才删除检测项和主记录。
        self.repository.remove(record_id)
