"""浏览器摄像头的版本化 WebSocket 推理协议。"""

import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.core.errors import AppError

router = APIRouter(tags=["camera"])


@router.websocket("/api/camera/ws")
async def camera_socket(websocket: WebSocket) -> None:
    """接收 ``frame_meta`` JSON 与紧随其后的 JPEG 二进制帧。"""

    await websocket.accept()
    await websocket.send_json({"type": "ready", "version": 1})
    last_frame_id = -1
    try:
        while True:
            # 每帧固定由一条元数据消息和一条二进制消息组成。先验证元数据，避免
            # 把失序客户端的数据误当成图片内容。
            meta = await websocket.receive_json()
            if meta.get("type") != "frame_meta" or meta.get("version") != 1:
                await websocket.send_json({"type": "error", "error_code": "FRAME_META_INVALID"})
                continue
            frame_id = meta.get("frame_id")
            # 单调 frame_id 让客户端能够丢弃迟到结果，且服务端无需保存帧队列。
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
                # ONNX 推理是同步 CPU 工作，放入线程池避免阻塞 FastAPI 事件循环。
                # infer_only 明确保证普通视频帧不会写入磁盘或检测历史。
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
