import io
import time

from app.main import create_app
from fastapi.testclient import TestClient
from PIL import Image


def make_image(color: str, size: tuple[int, int] = (200, 120)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "JPEG")
    return output.getvalue()


def activate_fake_model(client: TestClient) -> None:
    dataset = client.post("/api/datasets", json={"name": "推理测试"}).json()
    for class_name, color in [
        ("no_glasses", "white"),
        ("eyeglasses", "blue"),
        ("sunglasses", "black"),
    ]:
        client.post(
            f"/api/datasets/{dataset['id']}/imports/images",
            data={"class_name": class_name},
            files=[("files", (f"{class_name}.jpg", make_image(color), "image/jpeg"))],
        )
    for image in client.get(f"/api/datasets/{dataset['id']}/images").json():
        client.put(
            f"/api/images/{image['id']}/annotations",
            json=[
                {
                    "class_name": image["imported_class"],
                    "box": {"x1": 10, "y1": 10, "x2": 180, "y2": 110},
                    "source": "manual",
                }
            ],
        )
    version = client.post(f"/api/datasets/{dataset['id']}/versions", json={"seed": 3}).json()
    task = client.post(
        "/api/training",
        json={"dataset_version_id": version["id"], "preset": "quick", "epochs": 1},
    ).json()
    for _ in range(100):
        if client.get(f"/api/training/{task['id']}").json()["state"] == "completed":
            break
        time.sleep(0.02)
    model = client.get("/api/models").json()[0]
    assert client.post(f"/api/models/{model['id']}/activate").status_code == 200


def test_image_inference_saves_filterable_history(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path, min_per_class=1, fake_training=True)) as client:
        activate_fake_model(client)
        response = client.post(
            "/api/inference/image",
            data={"confidence": "0.25"},
            files={"file": ("people.jpg", make_image("gray"), "image/jpeg")},
        )

        assert response.status_code == 201
        record = response.json()
        assert [item["class_name"] for item in record["detections"]] == ["eyeglasses", "sunglasses"]
        filtered = client.get("/api/history", params={"class_name": "sunglasses"}).json()
        assert [item["id"] for item in filtered["items"]] == [record["id"]]
        assert client.get(record["annotated_url"]).status_code == 200
        assert client.delete(f"/api/history/{record['id']}").status_code == 204
        assert client.get(f"/api/history/{record['id']}").status_code == 404


def test_image_inference_requires_active_model(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        response = client.post(
            "/api/inference/image",
            files={"file": ("face.jpg", make_image("white"), "image/jpeg")},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ACTIVE_MODEL_REQUIRED"
