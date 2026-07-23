import io

from app.main import create_app
from fastapi.testclient import TestClient
from PIL import Image


def make_image(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 80), color).save(output, "JPEG")
    return output.getvalue()


def test_import_annotate_and_publish(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path, min_per_class=1)) as client:
        dataset = client.post("/api/datasets", json={"name": "教学数据"}).json()
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
            assert response.json()["imported"] == 1
        images = client.get(f"/api/datasets/{dataset['id']}/images").json()
        for image in images:
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
        check = client.get(f"/api/datasets/{dataset['id']}/publication-check").json()
        assert check["ready"] is True
        version = client.post(f"/api/datasets/{dataset['id']}/versions", json={"seed": 20260723})
        assert version.status_code == 201
        assert version.json()["class_counts"] == {"no_glasses": 1, "eyeglasses": 1, "sunglasses": 1}
        assert (tmp_path / version.json()["root_path"] / "data.yaml").exists()
