from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.annotations import router as annotations_router
from app.api.datasets import router as datasets_router
from app.core.config import load_settings
from app.core.database import Database
from app.core.errors import AppError
from app.domain.labels import CLASS_NAMES
from app.repositories.datasets import DatasetRepository
from app.services.publication import DatasetPublisher
from app.services.resources import ResourceService
from app.services.storage import ManagedStorage
from app.services.uploads import UploadValidator


def create_app(data_dir: Path | None = None, min_per_class: int = 10) -> FastAPI:
    settings = load_settings(data_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database = Database(settings.data_dir / "app.db")
        database.create_schema()
        storage = ManagedStorage(settings.data_dir)
        app.state.settings = settings
        app.state.database = database
        app.state.storage = storage
        app.state.datasets = DatasetRepository(database.session_factory)
        app.state.validator = UploadValidator(settings.max_upload_bytes, settings.max_image_pixels)
        app.state.publisher = DatasetPublisher(database.session_factory, storage, min_per_class)
        manifest = Path(__file__).resolve().parents[2] / "resources" / "manifest.json"
        app.state.resources = ResourceService(manifest, settings.data_dir / "resources")
        yield
        database.close()

    app = FastAPI(title="眼镜识别检测系统", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
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
    return app


app = create_app()
