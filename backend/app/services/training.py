import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from app.core.errors import AppError
from app.domain.labels import CLASS_NAMES
from app.domain.training import TrainingSettings, TrainingState


class TrainingService:
    def __init__(
        self,
        repository,
        models,
        datasets,
        storage,
        resources,
        fake: bool = False,
    ) -> None:
        self.repository = repository
        self.models = models
        self.datasets = datasets
        self.storage = storage
        self.resources = resources
        self.fake = fake
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(self, version_id: str, payload: dict) -> dict:
        version = self.datasets.version(version_id)
        try:
            settings = TrainingSettings.from_payload(payload)
        except (TypeError, ValueError) as exc:
            raise AppError("TRAINING_SETTINGS_INVALID", str(exc)) from exc
        task = self.repository.create(version_id, asdict(settings))
        thread = threading.Thread(
            target=self._run, args=(task["id"], version, settings), daemon=True
        )
        thread.start()
        return task

    def cancel(self, task_id: str) -> dict:
        task = self.repository.get(task_id)
        if task["state"] not in {"queued", "running"}:
            raise AppError("TRAINING_NOT_CANCELLABLE", "当前任务不能取消", 409)
        result = self.repository.transition(task_id, TrainingState.CANCELLING)
        marker = self.storage.resolve(f"training/{task_id}/cancel")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("cancel", encoding="utf-8")
        with self._lock:
            process = self._processes.get(task_id)
        if process and process.poll() is None:
            process.terminate()
        return result

    def resume(self, task_id: str) -> dict:
        old = self.repository.get(task_id)
        if old["state"] != "interrupted":
            raise AppError("TRAINING_NOT_RESUMABLE", "只有已中断任务可以续训", 409)
        return self.start(old["dataset_version_id"], old["settings"])

    def _run(self, task_id: str, version: dict, settings: TrainingSettings) -> None:
        try:
            self.repository.transition(task_id, TrainingState.RUNNING)
            self.repository.append_event(task_id, "started", {"device": settings.device})
            if self.fake:
                result = self._fake_run(task_id, settings)
            else:
                result = self._worker_run(task_id, version, settings)
            if self.repository.get(task_id)["state"] == "cancelling":
                self.repository.transition(
                    task_id,
                    TrainingState.FAILED,
                    error_code="TRAINING_CANCELLED",
                    error_message="训练已取消",
                )
                return
            model = self.models.create(
                task_id,
                f"眼镜检测模型 {task_id[:8]}",
                result["onnx_path"],
                result["onnx_sha256"],
                [name.value for name in CLASS_NAMES],
                result["metrics"],
            )
            result["model_id"] = model["id"]
            self.repository.append_event(task_id, "completed", result)
            self.repository.transition(task_id, TrainingState.COMPLETED, result=result)
        except BaseException as exc:
            try:
                task = self.repository.get(task_id)
                if task["state"] in {"running", "queued"}:
                    self.repository.append_event(task_id, "failed", {"message": str(exc)})
                    self.repository.transition(
                        task_id,
                        TrainingState.FAILED,
                        error_code=getattr(exc, "code", "TRAINING_FAILED"),
                        error_message=str(exc),
                    )
            except BaseException:
                pass

    def _fake_run(self, task_id: str, settings: TrainingSettings) -> dict:
        output = self.storage.resolve(f"training/{task_id}")
        output.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, settings.epochs + 1):
            if (output / "cancel").exists():
                break
            progress = epoch / settings.epochs
            self.repository.append_event(
                task_id,
                "epoch",
                {
                    "epoch": epoch,
                    "epochs": settings.epochs,
                    "loss": round(1.2 * (1 - progress) + 0.15, 4),
                    "precision": round(0.55 + 0.35 * progress, 4),
                    "recall": round(0.5 + 0.34 * progress, 4),
                    "map50": round(0.42 + 0.4 * progress, 4),
                    "map50_95": round(0.25 + 0.35 * progress, 4),
                },
            )
            time.sleep(0.01)
        artifact = output / "model.fake.onnx"
        artifact.write_bytes(b"FAKE_ONNX_FOR_TESTS")
        relative = self.storage.relative(artifact)
        metrics = {
            "map50": 0.82,
            "map50_95": 0.60,
            "precision": 0.90,
            "recall": 0.84,
            "per_class": {
                "no_glasses": {"map50": 0.84},
                "eyeglasses": {"map50": 0.83},
                "sunglasses": {"map50": 0.79},
            },
            "confusion_matrix": [[8, 1, 0], [1, 8, 1], [0, 1, 8]],
            "simulated": True,
        }
        return {
            "onnx_path": relative,
            "onnx_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "metrics": metrics,
            "device": "fake",
        }

    def _worker_run(self, task_id: str, version: dict, settings: TrainingSettings) -> dict:
        if not self.resources.valid("yolo26n"):
            raise AppError("BASE_MODEL_UNAVAILABLE", "请先下载 yolo26n 预训练权重", 409)
        output = self.storage.resolve(f"training/{task_id}")
        output.mkdir(parents=True, exist_ok=True)
        request_path = output / "request.json"
        events_path = output / "events.jsonl"
        result_path = output / "result.json"
        data_yaml = self.storage.resolve(f"{version['root_path']}/data.yaml")
        try:
            import torch
        except ImportError as exc:
            raise AppError("ML_RUNTIME_UNAVAILABLE", "请先安装训练依赖", 409) from exc
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise AppError("CUDA_UNAVAILABLE", "当前环境没有可用的 CUDA 设备", 409)
        use_cuda = settings.device in {"auto", "cuda"} and torch.cuda.is_available()
        actual_device = 0 if use_cuda else "cpu"
        request_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "model": str(self.resources.path("yolo26n")),
                    "data_yaml": str(data_yaml),
                    "output_dir": str(output),
                    **asdict(settings),
                    "device": actual_device,
                }
            ),
            encoding="utf-8",
        )
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.training.worker",
                "--request",
                str(request_path),
                "--events",
                str(events_path),
                "--result",
                str(result_path),
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            creationflags=flags,
        )
        with self._lock:
            self._processes[task_id] = process
        seen = 0
        while process.poll() is None:
            seen = self._ingest_events(task_id, events_path, seen)
            time.sleep(0.2)
        self._ingest_events(task_id, events_path, seen)
        with self._lock:
            self._processes.pop(task_id, None)
        if process.returncode != 0 or not result_path.exists():
            raise AppError("TRAINING_WORKER_FAILED", "训练进程执行失败")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["onnx_path"] = self.storage.relative(Path(result["onnx_path"]))
        return result

    def _ingest_events(self, task_id: str, path: Path, seen: int) -> int:
        if not path.exists():
            return seen
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[seen:]:
            event = json.loads(line)
            self.repository.append_event(task_id, event.pop("type"), event)
        return len(lines)
