from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/history", tags=["history"])


class BulkDelete(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=100)


@router.get("")
def list_history(
    request: Request,
    source: str | None = None,
    class_name: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = 50,
) -> dict:
    return {
        "items": request.app.state.history.list(
            source, class_name, date_from, date_to, min(max(limit, 1), 100)
        ),
        "next_cursor": None,
    }


@router.get("/{record_id}")
def get_history(record_id: str, request: Request) -> dict:
    return request.app.state.history.get(record_id)


def _history_file(record_id: str, key: str, request: Request) -> FileResponse:
    record = request.app.state.history.get(record_id)
    return FileResponse(request.app.state.storage.resolve(record[key]))


@router.get("/{record_id}/original")
def original(record_id: str, request: Request) -> FileResponse:
    return _history_file(record_id, "original_path", request)


@router.get("/{record_id}/annotated")
def annotated(record_id: str, request: Request) -> FileResponse:
    return _history_file(record_id, "annotated_path", request)


@router.delete("/{record_id}", status_code=204)
def delete_history(record_id: str, request: Request) -> Response:
    request.app.state.history_service.delete(record_id)
    return Response(status_code=204)


@router.post("/bulk-delete", status_code=204)
def bulk_delete(payload: BulkDelete, request: Request) -> Response:
    for record_id in dict.fromkeys(payload.ids):
        request.app.state.history_service.delete(record_id)
    return Response(status_code=204)
