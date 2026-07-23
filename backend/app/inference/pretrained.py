"""YuNet face detection plus mutually exclusive ONNX glasses classification."""

import time

import cv2
import numpy as np

from app.core.errors import AppError
from app.domain.geometry import PixelBox
from app.domain.inference import Detection, InferenceResult
from app.domain.labels import ClassName, class_id


class PretrainedGlassesInferenceEngine:
    """Classify every detected face exactly once as one of the three product classes."""

    def __init__(self, session, face_detector, model_id: str, input_size: int = 256) -> None:
        self.session = session
        self.face_detector = face_detector
        self.model_id = model_id
        self.input_size = input_size
        self.input_name = session.get_inputs()[0].name

    def infer(self, image: np.ndarray, confidence: float, _iou: float) -> InferenceResult:
        started = time.perf_counter()
        faces = self.face_detector.detect(image)
        detections: list[Detection] = []
        if faces:
            crops = np.stack([self._preprocess_crop(image, face.box) for face in faces])
            logits = np.asarray(self.session.run(None, {self.input_name: crops})[0])
            if logits.shape != (len(faces), 2) or not np.isfinite(logits).all():
                raise AppError("MODEL_OUTPUT_INVALID", "公开分类模型输出无效")
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            for face, (any_glasses, sunglasses) in zip(faces, probabilities, strict=True):
                class_name, class_probability = self._choose_class(
                    float(any_glasses), float(sunglasses)
                )
                # Face confidence and attribute confidence are both required for a box to be
                # trustworthy. Their product gives the UI slider one stable combined score.
                score = face.confidence * class_probability
                if score >= confidence:
                    detections.append(
                        Detection(face.box, class_id(class_name), class_name, float(score))
                    )
        providers = self.session.get_providers()
        device = f"YuNet + {providers[0] if providers else 'ONNX Runtime'}"
        return InferenceResult(
            image.shape[1],
            image.shape[0],
            tuple(detections),
            self.model_id,
            device,
            (time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _choose_class(any_glasses: float, sunglasses: float) -> tuple[ClassName, float]:
        # The published models are binary classifiers. Apply the more specific sunglasses
        # decision first, then fall back to any transparent glasses, otherwise no glasses.
        if sunglasses >= 0.5:
            return ClassName.SUNGLASSES, sunglasses
        if any_glasses >= 0.5:
            return ClassName.EYEGLASSES, any_glasses
        return ClassName.NO_GLASSES, 1.0 - any_glasses

    def _preprocess_crop(self, image: np.ndarray, box: PixelBox) -> np.ndarray:
        height, width = image.shape[:2]
        box_width, box_height = box.x2 - box.x1, box.y2 - box.y1
        margin = 0.12
        x1 = max(0, int(box.x1 - box_width * margin))
        y1 = max(0, int(box.y1 - box_height * margin))
        x2 = min(width, int(box.x2 + box_width * margin))
        y2 = min(height, int(box.y2 + box_height * margin))
        crop = image[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return np.ascontiguousarray((tensor - mean) / std, dtype=np.float32)
