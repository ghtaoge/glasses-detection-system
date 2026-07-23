# Dataset and Annotation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a locally runnable FastAPI/Vue workbench that safely imports face images, creates and edits three-class face annotations, and publishes immutable train/validation/test dataset versions.

**Architecture:** Define application-owned geometry and dataset contracts first, then add SQLite persistence, managed storage, safe import, pluggable face pre-annotation, version publication, and the data/annotation UI. External downloads and real face models stay behind adapters so deterministic tests require no network or model files.

**Tech Stack:** Python 3.11, FastAPI 0.139.2, Pydantic 2, SQLAlchemy 2.0.51, Alembic 1.18.5, SQLite, Pillow 12.3.0, OpenCV 5.0.0.93, ImageHash 4.3.2, pytest 9.1.1, Vue 3.5.40, Vue Router 5.2.0, TypeScript 7.0.2, Vite 8.1.5, Vitest 4.1.10, Playwright 1.61.1.

**Depends on:** [Approved design](../specs/2026-07-23-glasses-detection-system-design.md)

---

## Locked File Map

```text
.env.example                              Local runtime defaults
.gitignore                                Generated/runtime exclusions
pyproject.toml                            Exact backend dependencies and tools
backend/app/main.py                       FastAPI composition root
backend/app/core/config.py                GLASSES_ environment settings
backend/app/core/database.py              SQLite lifecycle
backend/app/core/errors.py                Stable application errors
backend/app/domain/labels.py               Three-class label contract
backend/app/domain/geometry.py             Pixel and YOLO box conversion
backend/app/repositories/models.py         Dataset SQLAlchemy tables
backend/app/repositories/datasets.py       Dataset persistence operations
backend/app/services/storage.py            Managed-path and atomic-file API
backend/app/services/uploads.py            Image/archive validation
backend/app/services/data_sources.py       License-gated source adapters
backend/app/services/preannotation.py      Face detector port and fake
backend/app/services/yunet.py              OpenCV YuNet adapter
backend/app/services/publication.py        Deduplication, split, snapshot export
backend/app/api/datasets.py                Dataset/import/version endpoints
backend/app/api/annotations.py             Annotation endpoints
backend/migrations/                        Alembic schema history
backend/tests/                              Unit and integration tests
frontend/src/api/                           Typed client and contracts
frontend/src/components/AnnotationCanvas.vue Canvas editor
frontend/src/views/DatasetsView.vue         Dataset operations
frontend/src/views/AnnotationView.vue       Review queue
frontend/src/styles/base.css                 Operational visual system
frontend/tests/                              Vitest tests
frontend/e2e/data-annotation.spec.ts         Browser milestone acceptance
README.md                                    Phase-one setup and use
```

### Task 1: Repository and Application Skeleton

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `backend/app/main.py`
- Test: `backend/tests/test_health.py`

- [ ] **Step 1: Write the failing health test**

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_local_storage(tmp_path):
    with TestClient(create_app(data_dir=tmp_path)) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "data_dir": str(tmp_path.resolve()),
        "class_names": ["no_glasses", "eyeglasses", "sunglasses"],
    }
```

- [ ] **Step 2: Run the test and verify the missing package failure**

Run: `py -m pytest backend/tests/test_health.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Add exact package configuration and the minimal app**

```toml
[project]
name = "glasses-detection-system"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
  "fastapi==0.139.2", "uvicorn[standard]==0.51.0",
  "pydantic==2.13.4", "pydantic-settings==2.14.2", "sqlalchemy==2.0.51",
  "alembic==1.18.5", "aiosqlite==0.22.1",
  "python-multipart==0.0.32", "pillow==12.3.0",
  "numpy==2.5.1", "opencv-python==5.0.0.93", "imagehash==4.3.2",
  "httpx==0.28.1", "kaggle==2.2.3"
]
[project.optional-dependencies]
ml = ["ultralytics==8.4.104", "onnx==1.22.0", "onnxslim==0.1.94", "onnxruntime==1.27.0"]
dev = ["pytest==9.1.1", "pytest-asyncio==1.4.0", "ruff==0.15.22"]
[tool.pytest.ini_options]
pythonpath = ["backend"]
testpaths = ["backend/tests"]
asyncio_mode = "auto"
[tool.ruff]
line-length = 100
src = ["backend/app", "backend/tests"]
```

```python
# backend/app/core/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GLASSES_", env_file=".env")
    data_dir: Path = Path("data")
    max_upload_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 24_000_000


def load_settings(data_dir: Path | None = None) -> Settings:
    return Settings(**({"data_dir": data_dir} if data_dir else {}))
```

Create `create_app(data_dir=None)` with a lifespan that creates the managed root and the exact `/api/health` response asserted above. Put `.venv/`, `.pytest_cache/`, `__pycache__/`, `.ruff_cache/`, `frontend/node_modules/`, `frontend/dist/`, `data/`, `models/`, `.env`, and Playwright output in `.gitignore`. Put only `GLASSES_DATA_DIR=data` and documented size limits in `.env.example`.

- [ ] **Step 4: Install and verify the skeleton**

Run: `py -m venv .venv`

Run: `.venv\Scripts\python -m pip install -e ".[dev]"`

Run: `.venv\Scripts\python -m pytest backend/tests/test_health.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit the skeleton**

```powershell
git add .gitignore .env.example pyproject.toml backend
git commit -m "chore: scaffold glasses detection workbench"
```

### Task 2: Label and Geometry Contracts

**Files:**
- Create: `backend/app/domain/labels.py`
- Create: `backend/app/domain/geometry.py`
- Test: `backend/tests/domain/test_geometry.py`

- [ ] **Step 1: Write box and label tests**

```python
import pytest
from app.domain.geometry import PixelBox
from app.domain.labels import ClassName, class_id


def test_fixed_class_order():
    assert class_id(ClassName.NO_GLASSES) == 0
    assert class_id(ClassName.EYEGLASSES) == 1
    assert class_id(ClassName.SUNGLASSES) == 2


def test_pixel_box_round_trips_yolo_coordinates():
    box = PixelBox(20, 10, 120, 90)
    assert box.to_yolo(200, 100) == pytest.approx((0.35, 0.5, 0.5, 0.8))
    assert PixelBox.from_yolo((0.35, 0.5, 0.5, 0.8), 200, 100) == box


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError, match="positive area"):
        PixelBox(10, 10, 10, 30)
```

- [ ] **Step 2: Run and verify missing domain modules**

Run: `.venv\Scripts\python -m pytest backend/tests/domain/test_geometry.py -q`

Expected: FAIL because `app.domain.geometry` does not exist.

- [ ] **Step 3: Implement immutable contracts**

```python
# backend/app/domain/labels.py
from enum import StrEnum


class ClassName(StrEnum):
    NO_GLASSES = "no_glasses"
    EYEGLASSES = "eyeglasses"
    SUNGLASSES = "sunglasses"


CLASS_NAMES = tuple(ClassName)


def class_id(name: ClassName) -> int:
    return CLASS_NAMES.index(name)
```

```python
# backend/app/domain/geometry.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PixelBox:
    x1: float; y1: float; x2: float; y2: float

    def __post_init__(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive area")

    def clamp(self, width: int, height: int) -> "PixelBox":
        return PixelBox(max(0, self.x1), max(0, self.y1), min(width, self.x2), min(height, self.y2))

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        return ((self.x1+self.x2)/(2*width), (self.y1+self.y2)/(2*height),
                (self.x2-self.x1)/width, (self.y2-self.y1)/height)

    @classmethod
    def from_yolo(cls, value, width: int, height: int) -> "PixelBox":
        cx, cy, w, h = value
        return cls((cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height)
```

- [ ] **Step 4: Run the domain tests**

Run: `.venv\Scripts\python -m pytest backend/tests/domain/test_geometry.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit domain contracts**

```powershell
git add backend/app/domain backend/tests/domain
git commit -m "feat: define glasses labels and box geometry"
```

### Task 3: Dataset Persistence and Migrations

**Files:**
- Create: `backend/app/core/database.py`
- Create: `backend/app/repositories/models.py`
- Create: `backend/app/repositories/datasets.py`
- Create: `backend/alembic.ini`
- Create: `backend/migrations/env.py`
- Create: `backend/migrations/versions/0001_datasets.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/repositories/test_datasets.py`

- [ ] **Step 1: Write immutable-version repository tests**

```python
import pytest
from app.repositories.datasets import DatasetRepository


@pytest.mark.asyncio
async def test_published_version_cannot_change(database):
    repo = DatasetRepository(database.session_factory)
    dataset = await repo.create_dataset("教学数据")
    image = await repo.add_image(dataset.id, "images/a.jpg", 256, 256, "abc", "phash")
    await repo.replace_annotations(image.id, [{"class_name":"eyeglasses","box":[10,10,200,220]}])
    version = await repo.publish(dataset.id, "manifest-sha", {"train":1,"val":0,"test":0})
    with pytest.raises(ValueError, match="published"):
        await repo.replace_version_annotations(version.id, image.id, [])
```

- [ ] **Step 2: Run and verify missing persistence**

Run: `.venv\Scripts\python -m pytest backend/tests/repositories/test_datasets.py -q`

Expected: FAIL because database and repository modules do not exist.

- [ ] **Step 3: Implement schema and repository boundaries**

```python
class DatasetRow(Base):       # id, name, state(draft/published), created_at
    __tablename__ = "datasets"
class ImageRow(Base):         # id, dataset_id, relative_path, width, height, sha256, phash, review_state
    __tablename__ = "images"
class AnnotationRow(Base):    # id, image_id, class_name, x1, y1, x2, y2, source, updated_at
    __tablename__ = "annotations"
class DatasetVersionRow(Base):# id, dataset_id, number, manifest_sha256, split_counts, created_at
    __tablename__ = "dataset_versions"
class VersionImageRow(Base):  # version_id, image_id, split, frozen_annotations JSON
    __tablename__ = "version_images"
class DataSourceRow(Base):    # id, source_key, source_version, license, accepted_at, snapshot JSON
    __tablename__ = "data_sources"
```

Use SQLAlchemy typed mappings, foreign keys with explicit delete behavior, UTC timestamps, WAL mode, foreign keys, and `busy_timeout=5000`. `DatasetRepository` returns frozen dataclasses and exposes only `create_dataset`, `add_image`, `replace_annotations`, `list_review_queue`, `publish`, `get_version`, and paginated reads. The `publish` transaction copies annotation JSON into `version_images`; later draft edits cannot affect it.

- [ ] **Step 4: Add Alembic and application lifecycle**

Create migration `0001_datasets` with the six tables and indexes on dataset/state, image hashes, review state, and version/split. Store `Database` and `DatasetRepository` on `app.state`; dispose the engine on shutdown. Production startup uses `alembic upgrade head`; tests use metadata creation only.

- [ ] **Step 5: Run repository and migration tests**

Run: `.venv\Scripts\python -m pytest backend/tests/repositories/test_datasets.py -q`

Expected: immutable snapshot, unique version number, cascade, and pagination tests pass.

Run: `.venv\Scripts\alembic -c backend/alembic.ini upgrade head`

Expected: `0001_datasets` applies to `data/app.db`.

- [ ] **Step 6: Commit persistence**

```powershell
git add backend/app/core/database.py backend/app/repositories backend/migrations backend/alembic.ini backend/app/main.py backend/tests/repositories
git commit -m "feat: persist versioned face datasets"
```

### Task 4: Managed Storage and Safe Image Import

**Files:**
- Create: `backend/app/core/errors.py`
- Create: `backend/app/services/storage.py`
- Create: `backend/app/services/uploads.py`
- Test: `backend/tests/services/test_uploads.py`

- [ ] **Step 1: Write path, image-bomb, and archive tests**

```python
import io, zipfile, pytest
from PIL import Image
from app.services.uploads import UploadValidator


def image_bytes(fmt="PNG"):
    out = io.BytesIO(); Image.new("RGB", (32, 24), "white").save(out, fmt); return out.getvalue()


def test_rejects_extension_spoof(validator):
    with pytest.raises(Exception, match="IMAGE_SIGNATURE_INVALID"):
        validator.validate_image("face.jpg", b"not an image")


def test_rejects_zip_traversal(validator):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive: archive.writestr("../escape.jpg", image_bytes())
    with pytest.raises(Exception, match="ARCHIVE_PATH_INVALID"):
        validator.extract_archive(out.getvalue())
```

- [ ] **Step 2: Run and verify missing services**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_uploads.py -q`

Expected: FAIL because upload services do not exist.

- [ ] **Step 3: Implement safe managed storage**

```python
class ManagedStorage:
    def __init__(self, root: Path): self.root = root.resolve()
    def resolve(self, relative: str) -> Path:
        candidate = (self.root / PurePosixPath(relative)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise AppError("PATH_OUTSIDE_MANAGED_ROOT", 400)
        return candidate
    def publish_bytes(self, relative: str, content: bytes) -> Path:
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as handle:
            partial = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(partial, target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return target
```

Replace the ellipsis during implementation with exact `tempfile.NamedTemporaryFile(dir=target.parent, delete=False)`, `flush`, `os.fsync`, and `os.replace` calls. `UploadValidator` allows JPEG, PNG, and WebP after Pillow `verify()` and full decode, limits bytes and pixels, strips unneeded metadata on managed copies, and computes SHA-256 plus pHash. Archive extraction limits entries to 5,000, total expanded size to 2 GB, compression ratio to 100, and rejects absolute paths, `..`, drive prefixes, links, and nested archives.

- [ ] **Step 4: Run security tests**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_uploads.py -q`

Expected: valid image, spoof, decompression limits, traversal, duplicate name, and root-boundary tests pass.

- [ ] **Step 5: Commit storage and validation**

```powershell
git add backend/app/core/errors.py backend/app/services/storage.py backend/app/services/uploads.py backend/tests/services
git commit -m "feat: validate and store imported face images"
```

### Task 5: License-Gated Teaching Data Import

**Files:**
- Create: `backend/app/services/data_sources.py`
- Create: `backend/app/api/datasets.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/services/test_data_sources.py`
- Test: `backend/tests/api/test_dataset_import.py`

- [ ] **Step 1: Write source allowlist and license-confirmation tests**

```python
def test_known_source_requires_exact_license_acceptance(source_service):
    with pytest.raises(Exception, match="LICENSE_ACCEPTANCE_REQUIRED"):
        source_service.start_download("kaggle-glasses-coverings-v2", accepted_license=None)

def test_unknown_source_is_rejected(source_service):
    with pytest.raises(Exception, match="DATA_SOURCE_NOT_ALLOWED"):
        source_service.start_download("https://example.invalid/file.zip", "yes")
```

- [ ] **Step 2: Run and verify missing source service**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_data_sources.py backend/tests/api/test_dataset_import.py -q`

Expected: FAIL because source service and routes do not exist.

- [ ] **Step 3: Implement one explicit source manifest**

```python
TEACHING_SOURCE = SourceManifest(
    key="kaggle-glasses-coverings-v2",
    title="Glasses and Coverings",
    dataset_ref="mantasu/glasses-and-coverings",
    version="2",
    license_id="CC-BY-NC-4.0",
    license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    page_url="https://www.kaggle.com/datasets/mantasu/glasses-and-coverings",
    folders={"plain":"no_glasses", "glasses":"eyeglasses",
             "sunglasses":"sunglasses", "sunglasses-imagenet":"sunglasses"},
)
```

The adapter invokes the official Kaggle CLI as an argument array, never a shell string: `kaggle datasets download -d mantasu/glasses-and-coverings -p <staging> --unzip`. Before spawn, require an acceptance payload matching source key, version, license id, and `non_commercial_teaching=True`. Persist the acceptance snapshot. Stream progress as application events, import only mapped folders, and leave no database rows when download or validation fails. Unit tests inject a fake process runner and fixture archive.

- [ ] **Step 4: Expose import APIs**

```text
GET  /api/data-sources
POST /api/datasets
POST /api/datasets/{id}/imports/source
POST /api/datasets/{id}/imports/archive
GET  /api/datasets/{id}/imports/{job_id}
```

Return `202` for asynchronous source/archive imports, stable error codes, and counts for imported, duplicate, invalid, and pending-review images. Do not accept a URL field.

- [ ] **Step 5: Run source and API tests**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_data_sources.py backend/tests/api/test_dataset_import.py -q`

Expected: all license, allowlist, rollback, mapping, and response-contract tests pass.

- [ ] **Step 6: Commit data import**

```powershell
git add backend/app/services/data_sources.py backend/app/api/datasets.py backend/app/main.py backend/tests
git commit -m "feat: import license-approved teaching data"
```

### Task 6: Face Pre-Annotation and Review Queue

**Files:**
- Create: `resources/manifest.json`
- Create: `backend/app/services/resources.py`
- Create: `backend/app/services/preannotation.py`
- Create: `backend/app/services/yunet.py`
- Create: `backend/app/api/annotations.py`
- Test: `backend/tests/services/test_resources.py`
- Test: `backend/tests/services/test_preannotation.py`
- Test: `backend/tests/api/test_annotations.py`

- [ ] **Step 1: Write review-routing tests**

```python
def test_one_confident_face_creates_draft_annotation(service, one_face_detector):
    result = service.annotate(image_record, "eyeglasses")
    assert result.review_state == "auto_annotated"
    assert result.annotations[0].source == "yunet"

def test_multiple_faces_require_review(service, two_face_detector):
    result = service.annotate(image_record, "sunglasses")
    assert result.review_state == "needs_review"
    assert len(result.annotations) == 2
```

- [ ] **Step 2: Run and verify missing pre-annotation**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_preannotation.py backend/tests/api/test_annotations.py -q`

Expected: FAIL because detector ports and routes do not exist.

- [ ] **Step 3: Implement the port and deterministic policy**

```python
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> tuple[FaceCandidate, ...]:
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class FaceCandidate:
    box: PixelBox
    confidence: float

def review_state(candidates, width, height):
    valid = [c for c in candidates if c.confidence >= 0.80]
    if len(valid) != 1: return "needs_review"
    area = ((valid[0].box.x2-valid[0].box.x1)*(valid[0].box.y2-valid[0].box.y1))/(width*height)
    return "auto_annotated" if 0.08 <= area <= 0.90 else "needs_review"
```

Add a resource manifest with these exact entries:

```json
{
  "yunet": {
    "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "bytes": 232589,
    "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "license": "Apache-2.0"
  },
  "yolo26n": {
    "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "bytes": 5544453,
    "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "license": "AGPL-3.0"
  }
}
```

`ResourceService.fetch(name)` accepts only manifest keys, downloads to a sibling partial file, enforces declared size while streaming, verifies SHA-256, and atomically moves into `data/resources/`. An existing valid file is reused; an invalid file is quarantined with a timestamp and reported. `YuNetFaceDetector` uses `cv2.FaceDetectorYN`, the verified local ONNX path, input-size updates per image, score threshold `0.80`, NMS threshold `0.3`, and top-k `5000`. Missing or checksum-invalid model produces `PREANNOTATION_MODEL_UNAVAILABLE`; it never downloads at inference time. A fake detector drives default tests.

- [ ] **Step 4: Add annotation APIs**

```text
POST /api/datasets/{id}/preannotate
GET  /api/datasets/{id}/images?review_state=&cursor=
GET  /api/images/{id}
PUT  /api/images/{id}/annotations
POST /api/images/{id}/review
```

`PUT` replaces all boxes in one transaction after class, positive-area, image-boundary, and finite-number validation. Return original pixel coordinates and image dimensions.

- [ ] **Step 5: Run pre-annotation tests**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_resources.py backend/tests/services/test_preannotation.py backend/tests/api/test_annotations.py -q`

Expected: single, none, multiple, low-score, abnormal-area, box-validation, and replace-transaction tests pass.

- [ ] **Step 6: Commit pre-annotation**

```powershell
git add resources backend/app/services/resources.py backend/app/services/preannotation.py backend/app/services/yunet.py backend/app/api/annotations.py backend/tests
git commit -m "feat: preannotate and review face boxes"
```

### Task 7: Deduplicated Dataset Publication

**Files:**
- Create: `backend/app/services/publication.py`
- Modify: `backend/app/api/datasets.py`
- Test: `backend/tests/services/test_publication.py`

- [ ] **Step 1: Write split and leakage tests**

```python
def test_duplicate_group_never_crosses_splits(publisher, balanced_records):
    version = publisher.publish(balanced_records, seed=20260723)
    by_group = {}
    for item in version.items: by_group.setdefault(item.duplicate_group, set()).add(item.split)
    assert all(len(splits) == 1 for splits in by_group.values())

def test_unreviewed_image_blocks_publication(publisher, balanced_records):
    balanced_records[0].review_state = "needs_review"
    with pytest.raises(Exception, match="DATASET_REVIEW_INCOMPLETE"):
        publisher.publish(balanced_records, seed=20260723)
```

- [ ] **Step 2: Run and verify missing publisher**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_publication.py -q`

Expected: FAIL because publication service does not exist.

- [ ] **Step 3: Implement deterministic publication**

```python
class DatasetPublisher:
    SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
    def publish(self, records, seed: int = 20260723) -> PublishedSnapshot:
        checked = self.validate_records(records)
        groups = self.build_duplicate_groups(checked, max_phash_distance=4)
        assignments = self.allocate_whole_groups(groups, ratios=self.SPLITS, seed=seed)
        with self.storage.snapshot_staging() as staging:
            manifest = self.write_yolo_snapshot(staging, checked, assignments)
            digest = hashlib.sha256(manifest.canonical_json()).hexdigest()
            final_root = self.storage.commit_snapshot(staging, digest)
        return self.repository.create_version(checked.dataset_id, digest, manifest, final_root)
```

Implement the five named helpers in the same class. `validate_records` rejects zero-count classes, fewer than 10 reviewed faces per class, invalid boxes, and missing managed source files. `build_duplicate_groups` uses union-find over equal SHA-256 or pHash Hamming distance at most four. `allocate_whole_groups` sorts seeded groups by class and size, then assigns each whole group to the split with the largest remaining per-class deficit; it never splits a group to improve ratios. `write_yolo_snapshot` writes YOLO label lines as `<class_id> <cx> <cy> <width> <height>` with six decimals and returns a manifest sorted by relative path. `commit_snapshot` uses an atomic directory rename on the same volume. `data.yaml` contains absolute snapshot paths only at training launch; the persisted portable manifest uses relative paths and `names: [no_glasses, eyeglasses, sunglasses]`. Record `identity_leakage_checked=false` because the seed data lacks identity labels.

- [ ] **Step 4: Expose validation and publication**

```text
GET  /api/datasets/{id}/publication-check
POST /api/datasets/{id}/versions {"seed": 20260723}
GET  /api/dataset-versions/{version_id}
```

The check returns blocking errors, warnings, per-class counts, duplicate groups, proposed split counts, and identity-leakage status. Publication re-runs the checks transactionally.

- [ ] **Step 5: Run publication tests**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_publication.py -q`

Expected: deterministic split, pHash grouping, class minimum, review gate, YOLO conversion, immutable snapshot, and rollback tests pass.

- [ ] **Step 6: Commit publication**

```powershell
git add backend/app/services/publication.py backend/app/api/datasets.py backend/tests/services/test_publication.py
git commit -m "feat: publish immutable YOLO datasets"
```

### Task 8: Vue Dataset and Annotation Workbench

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.ts`
- Create: `frontend/src/router.ts`
- Create: `frontend/src/App.vue`
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/components/AnnotationCanvas.vue`
- Create: `frontend/src/views/DatasetsView.vue`
- Create: `frontend/src/views/AnnotationView.vue`
- Create: `frontend/src/styles/base.css`
- Test: `frontend/tests/annotation-canvas.test.ts`
- Test: `frontend/e2e/data-annotation.spec.ts`

- [ ] **Step 1: Scaffold exact frontend dependencies and write canvas tests**

```json
{"scripts":{"dev":"vite","build":"vue-tsc -b && vite build","test":"vitest run","test:e2e":"playwright test"},"dependencies":{"vue":"3.5.40","vue-router":"5.2.0","lucide-vue-next":"1.0.0"},"devDependencies":{"@vitejs/plugin-vue":"6.0.8","@vue/test-utils":"2.4.11","@playwright/test":"1.61.1","typescript":"7.0.2","vite":"8.1.5","vitest":"4.1.10","vue-tsc":"3.2.5"}}
```

```typescript
it('converts display drag to original pixel coordinates', async () => {
  const wrapper = mount(AnnotationCanvas, { props: { imageWidth: 1000, imageHeight: 500, displayWidth: 500, displayHeight: 250, annotations: [] } })
  await wrapper.find('canvas').trigger('pointerdown', { offsetX: 50, offsetY: 25 })
  await wrapper.find('canvas').trigger('pointerup', { offsetX: 150, offsetY: 125 })
  expect(wrapper.emitted('create')?.[0]).toEqual([{ x1: 100, y1: 50, x2: 300, y2: 250 }])
})
```

- [ ] **Step 2: Run and verify the missing frontend**

Run: `npm --prefix frontend install`

Run: `npm --prefix frontend test`

Expected: FAIL because `AnnotationCanvas.vue` does not exist.

- [ ] **Step 3: Implement typed API and stable canvas state**

```typescript
export type ClassName = 'no_glasses' | 'eyeglasses' | 'sunglasses'
export interface PixelBox { x1:number; y1:number; x2:number; y2:number }
export interface Annotation { id:string; className:ClassName; box:PixelBox; source:'manual'|'yunet' }
export interface ImageRecord { id:string; url:string; width:number; height:number; reviewState:string; annotations:Annotation[] }
```

Use one coordinate conversion module for draw, hit-test, move, and resize. Canvas CSS size may respond, but backing dimensions follow `devicePixelRatio`; annotation values remain original pixels. Add keyboard delete, visible focus, pointer capture, and class selection with labeled swatches.

- [ ] **Step 4: Implement the two operational views**

`DatasetsView` covers create, source-license modal, source/archive import status, counts, publication check, and version list. `AnnotationView` uses a left filterable review queue, central canvas, right annotation list, save/review actions, and automatic navigation to the next pending image. Use full-width work areas, no nested cards, 8px-or-less radii, Lucide icons with tooltips, and responsive tracks that become queue/canvas/details tabs below 800px.

- [ ] **Step 5: Add browser acceptance**

```typescript
test('imports, reviews, and publishes a dataset', async ({ page }) => {
  await page.goto('/datasets')
  await page.getByRole('button', { name: '新建数据集' }).click()
  await page.getByLabel('名称').fill('教学数据')
  await page.getByRole('button', { name: '创建' }).click()
  await page.getByRole('link', { name: '标注工作台' }).click()
  await expect(page.getByText('待复核')).toBeVisible()
  await page.getByRole('button', { name: '确认并下一张' }).click()
  await page.goto('/datasets')
  await page.getByRole('button', { name: '发布版本' }).click()
  await expect(page.getByText('版本 1')).toBeVisible()
})
```

Use deterministic backend fixtures; also assert no console error, failed request, horizontal overflow, or canvas with zero dimensions at 1440x900 and 390x844.

- [ ] **Step 6: Run frontend and full phase tests**

Run: `npm --prefix frontend test`

Run: `npm --prefix frontend run build`

Run: `npm --prefix frontend run test:e2e -- data-annotation.spec.ts`

Run: `.venv\Scripts\python -m pytest -q`

Expected: all tests and builds pass.

- [ ] **Step 7: Document and commit phase one**

Update `README.md` with Python/Node prerequisites, setup, migrations, backend/frontend commands, local storage layout, Kaggle CLI authentication, license warning, and phase-one workflow.

```powershell
git add frontend README.md
git commit -m "feat: add dataset annotation workbench"
```

## Phase One Exit Gate

Run: `.venv\Scripts\python -m ruff check backend`

Run: `.venv\Scripts\python -m pytest -q`

Run: `npm --prefix frontend test`

Run: `npm --prefix frontend run build`

Run: `npm --prefix frontend run test:e2e -- data-annotation.spec.ts`

Expected: every command passes; a fixture archive can be imported, reviewed, and published into an immutable YOLO snapshot without network or real model access.
