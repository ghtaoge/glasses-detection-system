from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.annotations import router as annotations_router
from app.api.camera import router as camera_router
from app.api.datasets import router as datasets_router
from app.api.history import router as history_router
from app.api.inference import router as inference_router
from app.api.models import router as models_router
from app.api.overview import router as overview_router
from app.api.training import router as training_router
from app.core.config import load_settings
from app.core.database import Database
from app.core.errors import AppError
from app.domain.labels import CLASS_NAMES
from app.repositories.datasets import DatasetRepository
from app.repositories.history import HistoryRepository
from app.repositories.training import ModelRepository, TrainingRepository
from app.services.image_inference import HistoryService, ImageInferenceService
from app.services.publication import DatasetPublisher
from app.services.resources import ResourceService
from app.services.storage import ManagedStorage
from app.services.training import TrainingService
from app.services.uploads import UploadValidator


def create_app(
    data_dir: Path | None = None,
    min_per_class: int = 10,
    fake_training: bool | None = None,
    fake_inference: bool = False,
) -> FastAPI:
    settings = load_settings(data_dir)
    use_fake_training = settings.fake_training if fake_training is None else fake_training

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database = Database(settings.data_dir / "app.db")
        database.create_schema()
        storage = ManagedStorage(settings.data_dir)
        app.state.settings = settings
        app.state.database = database
        app.state.storage = storage
        datasets = DatasetRepository(database.session_factory)
        models = ModelRepository(database.session_factory)
        training_repository = TrainingRepository(database.session_factory)
        history = HistoryRepository(database.session_factory)
        app.state.datasets = datasets
        app.state.models = models
        app.state.training_repository = training_repository
        app.state.history = history
        app.state.fake_training = use_fake_training
        # 模拟训练只负责快速生成测试产物。模拟推理必须由测试代码单独显式开启，
        # 防止正常启动的应用把固定检测框展示成真实模型结果。
        app.state.fake_inference = fake_inference
        app.state.validator = UploadValidator(settings.max_upload_bytes, settings.max_image_pixels)
        app.state.publisher = DatasetPublisher(database.session_factory, storage, min_per_class)
        manifest = Path(__file__).resolve().parents[2] / "resources" / "manifest.json"
        app.state.resources = ResourceService(manifest, settings.data_dir / "resources")
        app.state.training_service = TrainingService(
            training_repository,
            models,
            datasets,
            storage,
            app.state.resources,
            fake=use_fake_training,
        )
        app.state.inference = ImageInferenceService(
            models,
            history,
            storage,
            app.state.validator,
            allow_fake=fake_inference,
        )
        app.state.history_service = HistoryService(history, storage)
        training_repository.interrupt_active()
        yield
        database.close()

    app = FastAPI(title="眼镜识别检测系统", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "data_dir": str(settings.data_dir.resolve()),
            "class_names": [name.value for name in CLASS_NAMES],
        }

    @app.get("/api/files/{relative_path:path}")
    def managed_file(relative_path: str) -> FileResponse:
        path = app.state.storage.resolve(relative_path)
        if not path.is_file():
            raise AppError("FILE_NOT_FOUND", "文件不存在", 404)
        return FileResponse(path)

    app.include_router(datasets_router)
    app.include_router(annotations_router)
    app.include_router(training_router)
    app.include_router(models_router)
    app.include_router(inference_router)
    app.include_router(history_router)
    app.include_router(camera_router)
    app.include_router(overview_router)
    return app


app = create_app()
