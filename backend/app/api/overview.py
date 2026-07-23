from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview")
def overview(request: Request) -> dict:
    datasets = request.app.state.datasets.list()
    tasks = request.app.state.training_repository.list()
    models = request.app.state.models.list()
    active = next((model for model in models if model["is_active"]), None)
    storage_bytes = sum(
        path.stat().st_size for path in request.app.state.storage.root.rglob("*") if path.is_file()
    )
    return {
        "datasets": {
            "count": len(datasets),
            "image_count": sum(item["image_count"] for item in datasets),
            "pending_count": sum(item["pending_count"] for item in datasets),
        },
        "active_model": active,
        "latest_training": tasks[0] if tasks else None,
        "latest_evaluation": models[0]["metrics"] if models else None,
        "detection_counts": request.app.state.history.counts(),
        "storage_bytes": storage_bytes,
        "resources": request.app.state.resources.status(),
    }
