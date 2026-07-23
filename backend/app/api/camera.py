import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.core.errors import AppError

router = APIRouter(tags=["camera"])


@router.websocket("/api/camera/ws")
async def camera_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "ready", "version": 1})
    last_frame_id = -1
    try:
        while True:
            meta = await websocket.receive_json()
            if meta.get("type") != "frame_meta" or meta.get("version") != 1:
                await websocket.send_json({"type": "error", "error_code": "FRAME_META_INVALID"})
                continue
            frame_id = meta.get("frame_id")
            if not isinstance(frame_id, int) or frame_id <= last_frame_id:
                await websocket.send_json({"type": "error", "error_code": "FRAME_ID_NOT_MONOTONIC"})
                continue
            frame = await websocket.receive_bytes()
            if len(frame) > 2 * 1024 * 1024:
                await websocket.send_json({"type": "error", "error_code": "FRAME_TOO_LARGE"})
                continue
            last_frame_id = frame_id
            started = time.perf_counter()
            try:
                _, result = await run_in_threadpool(
                    websocket.app.state.inference.infer_only,
                    "camera.jpg",
                    frame,
                    float(meta.get("confidence", 0.25)),
                    0.45,
                )
                payload = result.as_dict()
                payload.update(
                    {
                        "type": "result",
                        "version": 1,
                        "frame_id": frame_id,
                        "dropped_frames": 0,
                        "round_trip_ms": (time.perf_counter() - started) * 1000,
                    }
                )
                await websocket.send_json(payload)
            except AppError as exc:
                await websocket.send_json(
                    {"type": "error", "error_code": exc.code, "message": exc.message}
                )
    except WebSocketDisconnect:
        return
