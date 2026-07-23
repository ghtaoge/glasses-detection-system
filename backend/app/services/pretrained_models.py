"""Installation of the vetted public glasses classification pipeline."""

import hashlib
import os
from pathlib import Path

import numpy as np

from app.core.errors import AppError
from app.domain.labels import CLASS_NAMES


class PretrainedModelInstaller:
    """Download verified weights, export one ONNX artifact, and register it atomically."""

    ENGINE = "face_glasses_classifier_v1"
    RELATIVE_PATH = "models/pretrained-glasses-classifiers.onnx"

    def __init__(self, models, resources, storage) -> None:
        self.models = models
        self.resources = resources
        self.storage = storage

    def install(self) -> dict:
        """Build and activate the public pretrained model used for immediate local inference."""

        yunet_path = self.resources.fetch("yunet")
        any_weights = self.resources.fetch("glasses_any_classifier")
        sunglasses_weights = self.resources.fetch("glasses_sunglasses_classifier")
        output = self.storage.resolve(self.RELATIVE_PATH)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".partial.onnx")
        try:
            self._export(any_weights, sunglasses_weights, partial)
            self._validate_onnx(partial)
            os.replace(partial, output)
        finally:
            if partial.exists():
                partial.unlink()

        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        metrics = {
            # These classifiers publish F1 rather than object-detection mAP. Keep mAP fields
            # empty-valued for API compatibility instead of inventing incomparable scores.
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class": {},
            "simulated": False,
            "pretrained": True,
            "engine": self.ENGINE,
            "source": "mantasu/glasses-detector v1.0.0",
            "license": "MIT",
            "published_f1": {"anyglasses": 0.9693, "sunglasses": 0.9311},
            "face_detector": yunet_path.name,
        }
        model = self.models.upsert_pretrained(
            "公开预训练眼镜模型",
            self.RELATIVE_PATH,
            digest,
            [name.value for name in CLASS_NAMES],
            metrics,
        )
        return self.models.activate(model["id"])

    @staticmethod
    def _export(any_weights: Path, sunglasses_weights: Path, output: Path) -> None:
        try:
            import torch
            from torch import nn
            from torchvision.models import shufflenet_v2_x1_0
        except ImportError as exc:
            raise AppError("ML_RUNTIME_UNAVAILABLE", "请先安装训练依赖", 409) from exc

        def load_classifier(weights: Path):
            model = shufflenet_v2_x1_0(weights=None)
            model.fc = nn.Linear(1024, 1)
            # weights_only prevents Python objects embedded in an untrusted checkpoint from
            # executing during deserialization. The file is also pinned by size and SHA-256.
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            return model.eval()

        class CombinedClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.any_glasses = load_classifier(any_weights)
                self.sunglasses = load_classifier(sunglasses_weights)

            def forward(self, images):
                return torch.cat(
                    (self.any_glasses(images), self.sunglasses(images)), dim=1
                )

        try:
            torch.onnx.export(
                CombinedClassifier(),
                torch.zeros(1, 3, 256, 256),
                output,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        except Exception as exc:
            raise AppError("PRETRAINED_MODEL_EXPORT_FAILED", "公开模型导出 ONNX 失败", 500) from exc

    @staticmethod
    def _validate_onnx(path: Path) -> None:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output = session.run(
                None, {input_name: np.zeros((2, 3, 256, 256), dtype=np.float32)}
            )[0]
        except Exception as exc:
            raise AppError("PRETRAINED_MODEL_INVALID", "公开 ONNX 模型校验失败", 500) from exc
        if output.shape != (2, 2) or not np.isfinite(output).all():
            raise AppError("PRETRAINED_MODEL_INVALID", "公开 ONNX 模型输出无效", 500)
