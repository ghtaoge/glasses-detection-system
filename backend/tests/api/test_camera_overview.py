from app.main import create_app
from fastapi.testclient import TestClient


def test_camera_protocol_rejects_old_frame_id(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        with client.websocket_connect("/api/camera/ws") as socket:
            assert socket.receive_json() == {"type": "ready", "version": 1}
            socket.send_json(
                {"type": "frame_meta", "version": 1, "frame_id": 4, "confidence": 0.25}
            )
            socket.send_bytes(b"not-an-image")
            assert socket.receive_json()["error_code"] == "IMAGE_SIGNATURE_INVALID"
            socket.send_json({"type": "frame_meta", "version": 1, "frame_id": 3})
            assert socket.receive_json()["error_code"] == "FRAME_ID_NOT_MONOTONIC"


def test_overview_has_honest_empty_state(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        body = client.get("/api/overview").json()

    assert body["active_model"] is None
    assert body["latest_evaluation"] is None
    assert body["detection_counts"] == {"total": 0, "image": 0, "camera": 0}
