import asyncio
import json

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/api", tags=["training"])


class TrainingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: str
    preset: str = "quick"
    epochs: int | None = Field(None, ge=1, le=300)
    image_size: int | None = None
    batch_size: int | None = Field(None, ge=1, le=128)
    patience: int | None = Field(None, ge=0, le=50)
    device: str = "auto"


@router.post("/training", status_code=202)
def start_training(payload: TrainingCreate, request: Request) -> dict:
    values = {key: value for key, value in payload.model_dump().items() if value is not None}
    version_id = values.pop("dataset_version_id")
    return request.app.state.training_service.start(version_id, values)


@router.get("/training")
def list_training(request: Request) -> list[dict]:
    return request.app.state.training_repository.list()


@router.get("/training/{task_id}")
def get_training(task_id: str, request: Request) -> dict:
    return request.app.state.training_repository.get(task_id)


@router.post("/training/{task_id}/cancel")
def cancel_training(task_id: str, request: Request) -> dict:
    return request.app.state.training_service.cancel(task_id)


@router.post("/training/{task_id}/resume", status_code=202)
def resume_training(task_id: str, request: Request) -> dict:
    return request.app.state.training_service.resume(task_id)


@router.get("/training/{task_id}/events")
def training_events(
    task_id: str, request: Request, last_event_id: str | None = Header(None)
) -> StreamingResponse:
    repository = request.app.state.training_repository
    start = int(last_event_id or 0)

    async def stream():
        sequence = start
        heartbeat = 0
        while True:
            events = repository.events_after(task_id, sequence)
            for event in events:
                sequence = event["sequence"]
                yield (
                    f"id: {sequence}\nevent: {event['event_type']}\n"
                    f"data: {json.dumps(event['payload'], ensure_ascii=False)}\n\n"
                )
            task = repository.get(task_id)
            if task["state"] in {"completed", "failed", "interrupted"} and not events:
                break
            heartbeat += 1
            if heartbeat % 30 == 0:
                yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
