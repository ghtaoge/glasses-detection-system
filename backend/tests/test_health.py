from app.main import create_app
from fastapi.testclient import TestClient


def test_health_reports_local_storage(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path, min_per_class=1)) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "data_dir": str(tmp_path.resolve()),
        "class_names": ["no_glasses", "eyeglasses", "sunglasses"],
    }
