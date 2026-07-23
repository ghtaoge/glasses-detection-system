# Image Inference and History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver verified multi-face image inference with the active ONNX model, annotated result rendering, durable searchable history, and reliable record/file deletion.

**Architecture:** Keep preprocessing, ONNX output decoding, non-maximum suppression, rendering, and persistence as separate units. The inference service snapshots the active model reference per request, returns application-owned detections, atomically publishes artifacts, and writes history only after successful inference.

**Tech Stack:** Phase-two stack plus NumPy, Pillow, OpenCV, ONNX Runtime 1.27.0, Vue canvas overlays, Vitest, Playwright.

**Depends on:** [Phase two](2026-07-23-glasses-phase-2-training-models.md)

---

## Locked File Map

```text
backend/app/domain/inference.py            Detection response contracts
backend/app/inference/base.py              Inference session port
backend/app/inference/onnx.py              Preprocess/decode/NMS adapter
backend/app/inference/fake.py              Deterministic test engine
backend/app/services/rendering.py          Annotated image generation
backend/app/services/image_inference.py    Image use case
backend/app/repositories/history.py        Detection history persistence
backend/app/services/history.py            Artifact deletion workflow
backend/app/api/inference.py               Image inference endpoint
backend/app/api/history.py                 History endpoints
backend/migrations/versions/0003_history.py History schema
frontend/src/components/DetectionCanvas.vue Result overlay
frontend/src/views/RecognitionView.vue      Image recognition tab
frontend/src/views/HistoryView.vue          History list/detail
frontend/e2e/image-history.spec.ts          Browser acceptance
```

### Task 1: Inference Contracts and ONNX Adapter

**Files:**
- Create: `backend/app/domain/inference.py`
- Create: `backend/app/inference/base.py`
- Create: `backend/app/inference/onnx.py`
- Create: `backend/app/inference/fake.py`
- Test: `backend/tests/inference/test_onnx.py`

- [ ] **Step 1: Write preprocessing and decoding tests**

```python
def test_letterbox_round_trip(adapter, fake_session):
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    fake_session.output = fixture_yolo_output(box=(160,240,480,400), class_id=1, score=.9)
    result = adapter.infer(image, confidence=.25, iou=.45)
    assert result[0].class_name == "eyeglasses"
    assert result[0].box == pytest.approx(PixelBox(150,75,450,225), abs=1)

def test_nms_is_class_aware(adapter, fake_session):
    fake_session.output = overlapping_fixture(classes=(1,2), scores=(.9,.8))
    assert len(adapter.infer(np.zeros((640,640,3),np.uint8), .25, .45)) == 2
```

- [ ] **Step 2: Run and verify missing inference modules**

Run: `.venv\Scripts\python -m pytest backend/tests/inference/test_onnx.py -q`

Expected: FAIL because inference modules do not exist.

- [ ] **Step 3: Define application-owned output**

```python
@dataclass(frozen=True, slots=True)
class Detection:
    box: PixelBox; class_id: int; class_name: ClassName; confidence: float

@dataclass(frozen=True, slots=True)
class InferenceResult:
    width: int; height: int; detections: tuple[Detection, ...]
    model_id: str; device: str; duration_ms: float

class InferenceEngine(Protocol):
    def infer(self, image: np.ndarray, confidence: float, iou: float) -> InferenceResult:
        raise NotImplementedError
```

- [ ] **Step 4: Implement ONNX preprocessing and output isolation**

`OnnxInferenceEngine` accepts an injected `InferenceSession`, exact class map, and input size. Convert BGR to RGB, letterbox with scale and padding, normalize float32 to `[0,1]`, transpose NCHW, and add batch. Decode the verified Ultralytics detection output layout recorded during phase-two export validation, reject unexpected ranks/class counts, apply confidence filtering and class-aware NMS, reverse padding/scale, clamp to original dimensions, discard zero-area boxes, and sort by descending confidence. Record `session.get_providers()[0]` as actual device.

- [ ] **Step 5: Run adapter tests and CPU smoke**

Run: `.venv\Scripts\python -m pytest backend/tests/inference/test_onnx.py -q`

Expected: color, letterbox, decode-layout, confidence, class-aware NMS, coordinate clamp, invalid output, and device tests pass.

- [ ] **Step 6: Commit inference adapter**

```powershell
git add backend/app/domain/inference.py backend/app/inference backend/tests/inference
git commit -m "feat: run verified ONNX face inference"
```

### Task 2: Safe Image Inference and Annotation Rendering

**Files:**
- Create: `backend/app/services/rendering.py`
- Create: `backend/app/services/image_inference.py`
- Test: `backend/tests/services/test_image_inference.py`

- [ ] **Step 1: Write result and failure-order tests**

```python
@pytest.mark.asyncio
async def test_inference_renders_original_pixel_boxes(service, valid_upload):
    result = await service.run(valid_upload, confidence=.25)
    assert result.width == 640 and result.height == 480
    assert result.detections[0].box.x2 <= 640
    assert service.storage.resolve(result.annotated_path).exists()

@pytest.mark.asyncio
async def test_engine_failure_publishes_no_artifacts(service, failing_engine, valid_upload):
    with pytest.raises(Exception, match="INFERENCE_FAILED"):
        await service.run(valid_upload, confidence=.25)
    assert list(service.storage.inference_root.glob("**/*")) == []
```

- [ ] **Step 2: Run and verify missing services**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_image_inference.py -q`

Expected: FAIL because services do not exist.

- [ ] **Step 3: Implement rendering with stable category colors**

```python
COLORS = {"no_glasses":(52,168,83), "eyeglasses":(45,116,218), "sunglasses":(220,92,58)}
LABELS_ZH = {"no_glasses":"未戴眼镜", "eyeglasses":"普通眼镜", "sunglasses":"墨镜"}
```

Render on a copy of the decoded image. Scale line width and label text to image dimensions with bounded sizes, keep labels inside the image, and encode JPEG quality 90. The service validates with the phase-one validator, snapshots the active model/session under the registry read lock, runs inference outside the lock, renders, writes original and annotated files into task staging, and returns their temporary managed paths. No history row is created in this task.

- [ ] **Step 4: Run service tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_image_inference.py -q`

Expected: input validation, no-active-model, decode, multiple faces, zero faces, rendering bounds, engine failure, and cleanup tests pass.

```powershell
git add backend/app/services/rendering.py backend/app/services/image_inference.py backend/tests/services/test_image_inference.py
git commit -m "feat: render multi-face image detections"
```

### Task 3: Detection History Schema and Deletion State Machine

**Files:**
- Create: `backend/app/repositories/history.py`
- Create: `backend/app/services/history.py`
- Create: `backend/migrations/versions/0003_history.py`
- Test: `backend/tests/services/test_history.py`

- [ ] **Step 1: Write persistence and retry tests**

```python
@pytest.mark.asyncio
async def test_history_preserves_each_face(history, completed_inference):
    record = await history.save(completed_inference, source="image")
    loaded = await history.get(record.id)
    assert [d.class_name for d in loaded.detections] == ["eyeglasses","sunglasses"]

@pytest.mark.asyncio
async def test_file_failure_keeps_pending_delete(history, record, failing_storage):
    with pytest.raises(Exception, match="HISTORY_DELETE_INCOMPLETE"):
        await history.delete(record.id)
    assert (await history.get(record.id)).delete_state == "pending"
```

- [ ] **Step 2: Run and verify missing history**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_history.py -q`

Expected: FAIL because history modules do not exist.

- [ ] **Step 3: Add schema and repository**

```python
class DetectionRecordRow(Base):
    # id, source(image/camera), model_id, original_path, annotated_path,
    # width, height, duration_ms, device, delete_state, created_at
class DetectionItemRow(Base):
    # id, record_id, class_name, confidence, x1, y1, x2, y2
```

Migration `0003_history` adds foreign keys, indexes on created time/source/model/delete state, and an item class-name index. Save copies staged files atomically to `history/<record_id>/` before inserting one record and all item rows in one transaction. If insert fails, delete only that new directory.

- [ ] **Step 4: Implement recoverable deletion**

Deletion first marks selected rows `pending`, commits, deletes only resolved managed files, then deletes item/record rows in a second transaction. A file failure leaves `pending` for retry. Missing files count as already removed. A startup cleanup retries at most 100 pending records and never scans/deletes unrelated paths.

- [ ] **Step 5: Run migration and history tests**

Run: `.venv\Scripts\alembic -c backend/alembic.ini upgrade head`

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_history.py -q`

Expected: save atomicity, item ordering, filters, pagination, pending delete, retry, missing file, and path boundary tests pass.

- [ ] **Step 6: Commit history persistence**

```powershell
git add backend/app/repositories/history.py backend/app/services/history.py backend/migrations backend/tests/services/test_history.py
git commit -m "feat: persist and delete detection history"
```

### Task 4: Image Inference and History APIs

**Files:**
- Create: `backend/app/api/inference.py`
- Create: `backend/app/api/history.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/api/test_inference.py`
- Test: `backend/tests/api/test_history.py`

- [ ] **Step 1: Write API contract tests**

```python
def test_image_detection_returns_original_coordinates(client, active_fake_model, image_file):
    response = client.post("/api/inference/image", files={"file":image_file}, data={"confidence":"0.25"})
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "image" and body["detections"][0]["class_name"] == "eyeglasses"

def test_history_filters_any_matching_face(client, seeded_history):
    response = client.get("/api/history", params={"class_name":"sunglasses"})
    assert all(any(d["class_name"]=="sunglasses" for d in row["detections"]) for row in response.json()["items"])
```

- [ ] **Step 2: Run and verify `404` failures**

Run: `.venv\Scripts\python -m pytest backend/tests/api/test_inference.py backend/tests/api/test_history.py -q`

Expected: FAIL because routes do not exist.

- [ ] **Step 3: Add exact endpoints**

```text
POST   /api/inference/image
GET    /api/history?date_from=&date_to=&source=&class_name=&cursor=&limit=
GET    /api/history/{id}
GET    /api/history/{id}/original
GET    /api/history/{id}/annotated
DELETE /api/history/{id}
POST   /api/history/bulk-delete {"ids":["record-id-1","record-id-2"]}
```

Inference accepts multipart file, confidence `0.05..0.95`, and IoU `0.1..0.9`; it returns `201` only after history commit. File endpoints stream only files referenced by the record. Pagination is cursor-based and stable by `(created_at,id)`. Bulk deletion is limited to 100 explicit ids.

- [ ] **Step 4: Run API tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/api/test_inference.py backend/tests/api/test_history.py -q`

Expected: upload, zero/multiple detections, no model, filters, stable pagination, file access, single/bulk delete, and retry response tests pass.

```powershell
git add backend/app/api backend/app/main.py backend/tests/api
git commit -m "feat: expose image detection history APIs"
```

### Task 5: Recognition and History Views

**Files:**
- Create: `frontend/src/components/DetectionCanvas.vue`
- Create: `frontend/src/views/RecognitionView.vue`
- Create: `frontend/src/views/HistoryView.vue`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/router.ts`
- Modify: `frontend/src/App.vue`
- Test: `frontend/tests/detection-canvas.test.ts`
- Test: `frontend/tests/history-view.test.ts`
- Test: `frontend/e2e/image-history.spec.ts`

- [ ] **Step 1: Write overlay and filter tests**

```typescript
it('scales original boxes without mutating results', () => {
  const boxes = scaleDetections([{ box:{x1:100,y1:50,x2:300,y2:250}, className:'eyeglasses', confidence:.9 }], 1000, 500, 500, 250)
  expect(boxes[0].box).toEqual({ x1:50,y1:25,x2:150,y2:125 })
})
```

- [ ] **Step 2: Run and verify missing views**

Run: `npm --prefix frontend test`

Expected: FAIL because detection/history components do not exist.

- [ ] **Step 3: Implement image recognition**

Recognition page uses `图片检测` and disabled-until-phase-four `实时摄像头` tabs. Validate file type/size client-side, show local preview immediately, submit with progress and abort support, then render boxes from original coordinates in an aspect-ratio-stable media stage. Show per-face class, confidence, model, device, and duration. Preserve the selected image after recoverable server errors and expose retry.

- [ ] **Step 4: Implement history operations**

History uses date/source/class filters reflected in query parameters, cursor pagination, explicit row selection, delete confirmation, and pending-delete status. Detail view shows original/result toggles, all detections, model, device, and duration. Do not nest the media viewer inside multiple cards; use an unframed split layout on desktop and stacked layout on mobile.

- [ ] **Step 5: Add browser acceptance and run phase tests**

```typescript
test('detects multiple faces and deletes the saved record', async ({ page }) => {
  await page.goto('/recognition')
  await page.getByLabel('选择图片').setInputFiles('frontend/e2e/fixtures/two-faces.jpg')
  await expect(page.getByText('普通眼镜')).toBeVisible()
  await expect(page.getByText('墨镜')).toBeVisible()
  await page.goto('/history')
  await page.getByRole('link', { name: /查看详情/ }).first().click()
  await page.getByRole('button', { name: '删除记录' }).click()
  await page.getByRole('button', { name: '确认删除' }).click()
  await expect(page.getByText('记录已删除')).toBeVisible()
})
```

Run: `.venv\Scripts\python -m pytest -q`

Run: `npm --prefix frontend test`

Run: `npm --prefix frontend run build`

Run: `npm --prefix frontend run test:e2e -- image-history.spec.ts`

Expected: all checks pass with deterministic fake inference.

- [ ] **Step 6: Commit image workflow**

```powershell
git add frontend
git commit -m "feat: add image recognition and history views"
```

## Phase Three Exit Gate

Run the deterministic suite plus the opt-in active ONNX smoke test:

```powershell
.venv\Scripts\python -m ruff check backend
.venv\Scripts\python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e -- image-history.spec.ts
$env:GLASSES_REAL_ONNX='1'
.venv\Scripts\python -m pytest backend/tests/real/test_onnx_inference.py -q
```

Expected: a real registered ONNX model loads on CPU and returns a schema-valid result; deterministic browser flow saves, filters, views, and deletes a multi-face result.
