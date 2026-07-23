"""Ultralytics ONNX 推理适配器。

本模块把第三方模型输出转换为应用自己的 ``Detection`` 类型。预处理和后处理都
保留原图坐标语义，因此 API、结果渲染和浏览器叠加层不需要了解模型输入尺寸。
"""

import time

import cv2
import numpy as np

from app.core.errors import AppError
from app.domain.geometry import PixelBox
from app.domain.inference import Detection, InferenceResult
from app.domain.labels import CLASS_NAMES, ClassName


class OnnxInferenceEngine:
    """执行单张图片推理，并隔离 ONNX Runtime 与业务层的数据格式。"""

    def __init__(self, session, model_id: str, input_size: int = 640) -> None:
        self.session = session
        self.model_id = model_id
        self.input_size = input_size
        self.input_name = session.get_inputs()[0].name

    def infer(self, image: np.ndarray, confidence: float, iou: float) -> InferenceResult:
        started = time.perf_counter()
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        detections = self._decode(
            raw, image.shape[1], image.shape[0], scale, pad_x, pad_y, confidence, iou
        )
        provider = self.session.get_providers()[0] if self.session.get_providers() else "unknown"
        return InferenceResult(
            image.shape[1],
            image.shape[0],
            tuple(detections),
            self.model_id,
            provider,
            (time.perf_counter() - started) * 1000,
        )

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """等比例缩放并填充到方形输入，同时返回逆变换所需参数。"""

        height, width = image.shape[:2]
        # Letterbox 不拉伸原图，否则人脸比例变化会降低模型效果。114 是 YOLO
        # 训练管线使用的中性填充值，保持训练和推理预处理一致。
        scale = min(self.input_size / width, self.input_size / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (self.input_size - resized_w) // 2, (self.input_size - resized_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        # OpenCV 解码为 BGR；导出的 YOLO 模型使用 RGB、NCHW、float32 [0, 1]。
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return (
            np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255,
            scale,
            pad_x,
            pad_y,
        )

    def _decode(
        self,
        raw: np.ndarray,
        width: int,
        height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
        confidence: float,
        iou: float,
    ) -> list[Detection]:
        """解析 YOLO 输出、恢复原图坐标，并执行按类别 NMS。"""

        if raw.ndim != 3 or raw.shape[0] != 1:
            raise AppError("MODEL_OUTPUT_INVALID", "模型输出维度无效")
        # Ultralytics 可能导出 [1, channels, anchors] 或
        # [1, anchors, channels]。按固定通道数判断布局，不能依赖哪个维度更大，
        # 因为单元测试和裁剪后的模型可能只有很少候选框。
        channels = 4 + len(CLASS_NAMES)
        if raw.shape[1] == channels:
            rows = raw[0].T
        elif raw.shape[2] == channels:
            rows = raw[0]
        else:
            raise AppError("MODEL_OUTPUT_INVALID", "模型输出类别数量无效")
        class_ids = np.argmax(rows[:, 4:], axis=1)
        scores = rows[np.arange(len(rows)), class_ids + 4]
        keep = scores >= confidence
        rows, class_ids, scores = rows[keep], class_ids[keep], scores[keep]
        candidates: list[tuple[float, int, PixelBox]] = []
        for row, class_id, score in zip(rows, class_ids, scores, strict=True):
            cx, cy, box_w, box_h = row[:4]
            # 先移除 letterbox 边距，再除以缩放比例，最后限制在原图边界内。
            box = PixelBox(
                max(0, (cx - box_w / 2 - pad_x) / scale),
                max(0, (cy - box_h / 2 - pad_y) / scale),
                min(width, (cx + box_w / 2 - pad_x) / scale),
                min(height, (cy + box_h / 2 - pad_y) / scale),
            )
            if box.x2 > box.x1 and box.y2 > box.y1:
                candidates.append((float(score), int(class_id), box))
        accepted: list[Detection] = []
        for score, class_id, box in sorted(candidates, reverse=True, key=lambda item: item[0]):
            # 三个标签描述同一张脸上的互斥状态，因此跨类别高度重叠的候选框也必须
            # 互相抑制。保留最高置信度项，避免同一张脸同时显示普通眼镜和墨镜。
            if any(
                self._box_iou(box, item.box) > iou for item in accepted
            ):
                continue
            accepted.append(Detection(box, class_id, ClassName(CLASS_NAMES[class_id]), score))
        return accepted

    @staticmethod
    def _box_iou(left: PixelBox, right: PixelBox) -> float:
        intersection = max(0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
            0, min(left.y2, right.y2) - max(left.y1, right.y1)
        )
        left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
        right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
        return intersection / max(left_area + right_area - intersection, 1e-9)
