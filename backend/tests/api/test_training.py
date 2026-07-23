import io
import time

from app.main import create_app
from fastapi.testclient import TestClient
from PIL import Image


def make_image(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 80), color).save(output, "JPEG")
    return output.getvalue()


def publish_dataset(client: TestClient) -> str:
    dataset = client.post("/api/datasets", json={"name": "训练数据"}).json()
    for class_name, color in [
        ("no_glasses", "white"),
        ("eyeglasses", "blue"),
        ("sunglasses", "black"),
    ]:
        response = client.post(
            f"/api/datasets/{dataset['id']}/imports/images",
            data={"class_name": class_name},
            files=[("files", (f"{class_name}.jpg", make_image(color), "image/jpeg"))],
        )
        assert response.status_code == 200
    for image in client.get(f"/api/datasets/{dataset['id']}/images").json():
        response = client.put(
            f"/api/images/{image['id']}/annotations",
            json=[
                {
                    "class_name": image["imported_class"],
                    "box": {"x1": 10, "y1": 5, "x2": 90, "y2": 75},
                    "source": "manual",
                }
            ],
        )
        assert response.status_code == 200
    version = client.post(f"/api/datasets/{dataset['id']}/versions", json={"seed": 7})
    assert version.status_code == 201
    return version.json()["id"]


def test_training_creates_model(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path, min_per_class=1, fake_training=True)) as client:
        version_id = publish_dataset(client)
        response = client.post(
            "/api/training",
            json={"dataset_version_id": version_id, "preset": "quick", "epochs": 2},
        )
        assert response.status_code == 202
        task_id = response.json()["id"]

        for _ in range(100):
            task = client.get(f"/api/training/{task_id}").json()
            if task["state"] == "completed":
                break
            time.sleep(0.02)

        assert task["state"] == "completed"
        models = client.get("/api/models").json()
        assert len(models) == 1
        assert models[0]["metrics"]["map50"] == 0.82
        activated = client.post(f"/api/models/{models[0]['id']}/activate")
        assert activated.status_code == 200
        assert activated.json()["is_active"] is True


def test_training_rejects_unknown_settings(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path, fake_training=True)) as client:
        response = client.post(
            "/api/training",
            json={"dataset_version_id": "missing", "unknown": True},
        )

    assert response.status_code == 422
