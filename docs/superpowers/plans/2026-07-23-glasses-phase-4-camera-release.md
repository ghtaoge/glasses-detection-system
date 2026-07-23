# Realtime Camera and Release Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete realtime browser-camera inference, explicit snapshot history, resource coordination, dashboard visibility, recovery, security checks, reproducible local scripts, and final responsive browser acceptance.

**Architecture:** Use a versioned binary WebSocket protocol with one replaceable pending frame per connection, never an unbounded queue. Coordinate inference provider choice with training state, keep camera frames ephemeral unless the user saves a snapshot, then finish with startup reconciliation, bounded cleanup, full browser QA, and local run/test scripts.

**Tech Stack:** Existing stack, browser `getUserMedia`, FastAPI WebSockets, ONNX Runtime CPU/CUDA providers, Vue 3, Canvas, Vitest, Playwright.

**Depends on:** [Phase three](2026-07-23-glasses-phase-3-inference-history.md)

---

## Locked File Map

```text
backend/app/domain/camera.py                Frame/result protocol contracts
backend/app/services/resource_coordinator.py Training/inference device policy
backend/app/services/camera_sessions.py     Latest-frame connection service
backend/app/api/camera.py                   WebSocket and snapshot endpoints
backend/app/services/reconciliation.py      Startup state/artifact repair
backend/app/services/retention.py           Bounded cleanup retries
backend/app/api/overview.py                 Dashboard aggregate endpoint
frontend/src/composables/useCamera.ts        Browser media lifecycle
frontend/src/composables/useDetectionSocket.ts WebSocket sequencing
frontend/src/components/CameraStage.vue      Video/canvas stage
frontend/src/views/RecognitionView.vue       Camera tab
frontend/src/views/OverviewView.vue          System summary
frontend/e2e/camera.spec.ts                  Deterministic media acceptance
frontend/e2e/responsive.spec.ts              Desktop/mobile layout checks
frontend/e2e/accessibility.spec.ts           Keyboard and a11y checks
scripts/setup.ps1                            Idempotent local setup
scripts/start.ps1                            Local backend/frontend start
scripts/test.ps1                             Deterministic release gate
.github/workflows/ci.yml                     CPU-only CI
docs/OPERATIONS.md                           Local operating guide
docs/PRIVACY.md                              Face-data and license guidance
```

### Task 1: Versioned Camera Protocol and Latest-Frame Service

**Files:**
- Create: `backend/app/domain/camera.py`
- Create: `backend/app/services/camera_sessions.py`
- Create: `backend/app/api/camera.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/services/test_camera_sessions.py`
- Test: `backend/tests/api/test_camera_websocket.py`

- [ ] **Step 1: Write stale-frame and response-order tests**

```python
@pytest.mark.asyncio
async def test_pending_frame_is_replaced_not_queued(camera_session, slow_engine):
    await camera_session.submit(frame(1)); await camera_session.submit(frame(2)); await camera_session.submit(frame(3))
    slow_engine.release_first()
    assert [r.frame_id for r in await camera_session.collect(2)] == [1, 3]
    assert camera_session.metrics.dropped_frames == 1

def test_old_frame_id_is_rejected(websocket_client, active_fake_model):
    websocket_client.send_json({"type":"frame_meta","version":1,"frame_id":4,"content_type":"image/jpeg"})
    websocket_client.send_bytes(valid_jpeg)
    websocket_client.send_json({"type":"frame_meta","version":1,"frame_id":3,"content_type":"image/jpeg"})
    assert websocket_client.receive_json()["error_code"] == "FRAME_ID_NOT_MONOTONIC"
```

- [ ] **Step 2: Run and verify missing camera modules**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_camera_sessions.py backend/tests/api/test_camera_websocket.py -q`

Expected: FAIL because camera modules do not exist.

- [ ] **Step 3: Define exact protocol**

```python
@dataclass(frozen=True, slots=True)
class FrameMeta:
    version: Literal[1]; frame_id: int; captured_at_ms: int
    width: int; height: int; content_type: Literal["image/jpeg"]

@dataclass(frozen=True, slots=True)
class CameraResult:
    version: Literal[1]; frame_id: int; detections: tuple[Detection, ...]
    model_id: str; device: str; duration_ms: float; dropped_frames: int
```

Client sends one JSON `frame_meta` followed by one binary JPEG. Limit frame bytes to 2 MB, dimensions to 1920x1080, rate to 15 accepted frames/second, and idle time to 30 seconds. Server sends `ready`, `result`, `warning`, or `error` JSON. Close with policy error after malformed sequencing, non-monotonic ids, repeated rate violations, or invalid images.

- [ ] **Step 4: Implement one running and one replaceable frame**

Each connection owns exactly one inference task and one `pending: Frame | None`. `submit` replaces pending and increments dropped count. The processing loop takes pending under a lock, clears it, executes inference, sends a result, and repeats; cancellation closes the loop and awaits the exact task. Never place raw frames in SQLite or logs. WebSocket errors use stable codes but omit image bytes and local paths.

- [ ] **Step 5: Run protocol tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_camera_sessions.py backend/tests/api/test_camera_websocket.py -q`

Expected: latest-frame, disconnect cancellation, byte/dimension/rate limits, monotonic ids, malformed pair, no-model, and error-redaction tests pass.

```powershell
git add backend/app/domain/camera.py backend/app/services/camera_sessions.py backend/app/api/camera.py backend/app/main.py backend/tests
git commit -m "feat: stream latest camera frames"
```

### Task 2: Resource Coordination and Snapshot Persistence

**Files:**
- Create: `backend/app/services/resource_coordinator.py`
- Modify: `backend/app/services/camera_sessions.py`
- Modify: `backend/app/api/camera.py`
- Test: `backend/tests/services/test_resource_coordinator.py`
- Test: `backend/tests/api/test_camera_snapshot.py`

- [ ] **Step 1: Write training/device and snapshot tests**

```python
def test_training_on_cuda_routes_new_camera_session_to_cpu(coordinator):
    coordinator.training_started("cuda")
    lease = coordinator.acquire_inference()
    assert lease.device == "cpu" and lease.warning == "GPU_TRAINING_ACTIVE"

def test_snapshot_is_reinferred_before_history_save(client, camera_frame, forged_result):
    response = client.post("/api/camera/snapshots", files={"frame":camera_frame}, data={"frame_id":"8","client_result":json.dumps(forged_result)})
    assert response.status_code == 201
    assert response.json()["detections"] != forged_result["detections"]
```

- [ ] **Step 2: Run and verify missing coordinator/snapshot**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_resource_coordinator.py backend/tests/api/test_camera_snapshot.py -q`

Expected: FAIL because resource coordinator and snapshot route do not exist.

- [ ] **Step 3: Implement provider leases**

The coordinator tracks training device and counts active inference leases. When CUDA training is active, new image/camera inference uses a separately loaded CPU session and returns warning `GPU_TRAINING_ACTIVE`; existing CUDA requests finish. When training ends, new leases prefer the active CUDA session again. If CPU session loading fails, reject new inference with `INFERENCE_RESOURCE_UNAVAILABLE`; do not compete for the training GPU. Locks protect selection only, never the full inference duration.

- [ ] **Step 4: Implement explicit snapshots**

Add `POST /api/camera/snapshots` accepting the current JPEG and frame id. Client results are diagnostic only; the server validates and re-runs the current active model, renders, and saves via the phase-three history service with source `camera`. Limit to one snapshot request per second per connection token. No camera history is written by the WebSocket itself.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_resource_coordinator.py backend/tests/api/test_camera_snapshot.py -q`

Expected: training start/end, CPU selection, load failure, concurrent leases, forged client result, rate limit, and saved source tests pass.

```powershell
git add backend/app/services/resource_coordinator.py backend/app/services/camera_sessions.py backend/app/api/camera.py backend/tests
git commit -m "feat: coordinate inference and save camera snapshots"
```

### Task 3: Browser Camera Lifecycle and Detection Overlay

**Files:**
- Create: `frontend/src/composables/useCamera.ts`
- Create: `frontend/src/composables/useDetectionSocket.ts`
- Create: `frontend/src/components/CameraStage.vue`
- Modify: `frontend/src/views/RecognitionView.vue`
- Test: `frontend/tests/use-camera.test.ts`
- Test: `frontend/tests/detection-socket.test.ts`
- Test: `frontend/tests/camera-stage.test.ts`

- [ ] **Step 1: Write lifecycle and stale-response tests**

```typescript
it('stops every media track on stop and unmount', async () => {
  const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }]
  mockGetUserMedia(tracks)
  const camera = useCamera()
  await camera.start('device-1'); camera.stop()
  expect(tracks.every(track => track.stop.mock.calls.length === 1)).toBe(true)
})

it('ignores a response older than the displayed frame', () => {
  const socket = useDetectionSocket()
  socket.accept(result(12)); socket.accept(result(11))
  expect(socket.latest.value?.frameId).toBe(12)
})
```

- [ ] **Step 2: Run and verify missing composables**

Run: `npm --prefix frontend test`

Expected: FAIL because camera composables do not exist.

- [ ] **Step 3: Implement camera ownership**

`useCamera` enumerates devices after permission, starts exact `deviceId` with ideal 1280x720, exposes permission/device errors, and stops all tracks on stop, device change, route leave, page hide, and unmount. It does not auto-request permission on page load. `useDetectionSocket` reconnects only while the user remains started, uses capped backoff `0.5,1,2,4,5` seconds, resets frame sequence per connection, and never replays frames.

- [ ] **Step 4: Implement bounded frame capture and display smoothing**

`CameraStage` uses an unframed video with an absolute canvas of identical aspect ratio. At target FPS `1..15`, capture only when the socket is ready and no unsent frame exists, JPEG quality `0.75`, then send metadata and bytes. Draw only results with `frameId >= displayedFrameId`. Retain the last valid boxes for 500 ms, render them with reduced opacity, and clear at timeout; the UI labels retained boxes as stale internally for tests but adds no instructional prose.

- [ ] **Step 5: Implement controls and snapshot action**

The camera tab provides device menu, start/stop icon buttons with tooltips, confidence slider, target-FPS stepper, actual device/latency/drop status, and save-snapshot command. Disable snapshot without a decoded current frame. Permission denied, device busy, unsupported API, socket reconnect, no model, and CPU-during-training states have distinct visible messages.

- [ ] **Step 6: Run component tests and commit**

Run: `npm --prefix frontend test`

Expected: permissions, device switch, track cleanup, capture backpressure, stale response, 500 ms clear, reconnect, controls, and snapshot tests pass.

```powershell
git add frontend/src/composables frontend/src/components/CameraStage.vue frontend/src/views/RecognitionView.vue frontend/tests
git commit -m "feat: add realtime camera detection UI"
```

### Task 4: Overview, Startup Reconciliation, and Retention

**Files:**
- Create: `backend/app/services/reconciliation.py`
- Create: `backend/app/services/retention.py`
- Create: `backend/app/api/overview.py`
- Modify: `backend/app/main.py`
- Create: `frontend/src/views/OverviewView.vue`
- Modify: `frontend/src/router.ts`
- Test: `backend/tests/services/test_reconciliation.py`
- Test: `backend/tests/services/test_retention.py`
- Test: `backend/tests/api/test_overview.py`
- Test: `frontend/tests/overview-view.test.ts`

- [ ] **Step 1: Write recovery and overview tests**

```python
@pytest.mark.asyncio
async def test_restart_marks_only_nonterminal_training_interrupted(reconciler, running_task, completed_task):
    await reconciler.run()
    assert (await reconciler.training.get(running_task.id)).state == "interrupted"
    assert (await reconciler.training.get(completed_task.id)).state == "completed"

def test_overview_reports_honest_empty_state(client):
    body = client.get("/api/overview").json()
    assert body["active_model"] is None
    assert body["latest_evaluation"] is None
    assert body["detection_counts"] == {"image":0,"camera":0}
```

- [ ] **Step 2: Run and verify missing recovery/overview**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_reconciliation.py backend/tests/services/test_retention.py backend/tests/api/test_overview.py -q`

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement bounded reconciliation and retention**

At startup, mark nonterminal training interrupted, remove task-local `.partial` files older than one hour, retry at most 100 pending history deletes, verify the active model file/checksum/class map, and expose a warning instead of silently selecting another model. Retention runs once at startup and every six hours, handles at most 100 explicitly expired or pending records, resolves every file under managed roots, preserves dataset versions/models/training metrics, and records failures for retry. Shutdown cancels and awaits the timer.

- [ ] **Step 4: Add dashboard aggregate endpoint and view**

`GET /api/overview` returns dataset image/class counts, latest published version, active model and quality status, latest training/evaluation, history counts by source/class for the selected 30-day window, storage bytes, actual runtime providers, and warnings. `OverviewView` presents compact metric bands, class distribution, latest run, current model, recent detections, and clear empty/error states. It is the default route and links to operational pages without marketing copy.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_reconciliation.py backend/tests/services/test_retention.py backend/tests/api/test_overview.py -q`

Run: `npm --prefix frontend test`

Expected: recovery bounds, active-model failure, deletion retry, timer shutdown, aggregate, empty, and warning tests pass.

```powershell
git add backend/app/services backend/app/api/overview.py backend/app/main.py backend/tests frontend/src/views/OverviewView.vue frontend/src/router.ts frontend/tests
git commit -m "feat: add recovery and system overview"
```

### Task 5: Browser, Responsive, and Accessibility Acceptance

**Files:**
- Create: `frontend/e2e/camera.spec.ts`
- Create: `frontend/e2e/responsive.spec.ts`
- Create: `frontend/e2e/accessibility.spec.ts`
- Modify: `frontend/playwright.config.ts`
- Modify: `frontend/package.json`
- Test: `frontend/tests/layout.test.ts`

- [ ] **Step 1: Add deterministic media and camera journey**

```typescript
test('detects camera frames and saves one snapshot', async ({ page }) => {
  await page.addInitScript(mockCameraWithFixtureVideo)
  await page.goto('/recognition?tab=camera')
  await page.getByRole('button', { name: '开始摄像头' }).click()
  await expect(page.getByText('普通眼镜')).toBeVisible()
  await page.getByRole('button', { name: '保存快照' }).click()
  await page.goto('/history?source=camera')
  await expect(page.getByText('摄像头快照')).toBeVisible()
})
```

Use a deterministic canvas-backed fake `MediaStreamTrack` and fixture frames; assert only one history record is created and ordinary streamed frames are not persisted.

- [ ] **Step 2: Add desktop/mobile geometry checks**

At 1440x900 and 390x844 visit all seven routes. Assert no horizontal page overflow; toolbar text fits; media/canvas dimensions are nonzero; annotation, detection, and camera canvases align with their media; navigation is usable; tables become scroll regions rather than overflowing the viewport; and no panel overlaps adjacent content.

- [ ] **Step 3: Add keyboard and accessibility checks**

Install `@axe-core/playwright==4.12.1` with Playwright 1.61.1 and lock it. Use role/name queries for primary flows. Assert icon-only buttons have accessible names and tooltips, fields have visible labels, dialogs trap and restore focus, tabs/segmented controls expose selection, status messages use restrained live regions, canvas operations have a keyboard-accessible annotation list alternative, and no serious/critical axe violations exist.

- [ ] **Step 4: Capture and inspect visual evidence**

Capture every route at both viewports plus camera running, annotation editing, training running, model failure, and empty states. Add pixel-variance assertions for video and canvases so blank media cannot pass. Review screenshots for overlap, truncation, nested cards, excessive rounding, one-hue dominance, and stale overlays; change implementation rather than accepting incorrect snapshots.

- [ ] **Step 5: Run browser acceptance and commit**

Run: `npm --prefix frontend run test:e2e -- camera.spec.ts responsive.spec.ts accessibility.spec.ts`

Expected: workflows, geometry, pixel, console, network, keyboard, and axe checks pass at both viewports.

```powershell
git add frontend/e2e frontend/playwright.config.ts frontend/package.json frontend/package-lock.json frontend/tests
git commit -m "test: verify realtime responsive experience"
```

### Task 6: Reproducible Setup, CI, and Operations Documentation

**Files:**
- Create: `scripts/setup.ps1`
- Create: `scripts/start.ps1`
- Create: `scripts/test.ps1`
- Create: `.github/workflows/ci.yml`
- Create: `docs/OPERATIONS.md`
- Create: `docs/PRIVACY.md`
- Modify: `README.md`
- Test: `backend/tests/deploy/test_scripts.py`

- [ ] **Step 1: Write script safety tests**

```python
def test_scripts_use_repo_relative_paths_and_one_uvicorn_worker():
    setup = Path("scripts/setup.ps1").read_text()
    start = Path("scripts/start.ps1").read_text()
    all_text = setup + start + Path("scripts/test.ps1").read_text()
    assert "$PSScriptRoot" in all_text
    assert "$HOME" not in all_text and "$home" not in all_text
    assert "Remove-Item -Recurse" not in all_text
    assert "--workers 1" in start
```

- [ ] **Step 2: Implement idempotent local scripts**

`setup.ps1` verifies Python 3.11-3.13 and Node 24+, creates `.venv` only if absent, installs `.[dev,ml]`, runs `npm ci`, creates managed directories, copies `.env.example` only if `.env` is absent, runs Alembic, and prints explicit Kaggle/YuNet/base-weight preparation status. It does not guess a CUDA PyTorch wheel; document the official selector and re-run setup after PyTorch installation. `start.ps1` runs migrations, starts one Uvicorn worker and Vite in hidden child windows, tracks exact process objects, and stops them in `finally`. `test.ps1` runs Ruff, pytest, Vitest, build, and Playwright, preserving the first failing exit code; `-RealModel` and `-Cuda` explicitly enable optional checks after prerequisite validation.

- [ ] **Step 3: Add deterministic CPU CI**

CI on Windows and Linux installs locked dependencies, runs Alembic against a temporary database, Ruff, backend tests with fake adapters, Vitest, frontend build, and Playwright with deterministic media. It never downloads Kaggle data or large weights, never marks CUDA/real-training tests passed when skipped, and uploads browser traces/screenshots only on failure.

- [ ] **Step 4: Write operator and privacy docs**

`OPERATIONS.md` gives exact setup/start/test commands, source-license acceptance, data import/review/publication, quick vs standard training, cancellation/resume, metric interpretation, model activation/rollback, image/camera use, snapshot/history deletion, storage backup, interrupted-task recovery, CPU fallback, and log locations. `PRIVACY.md` explains local face-image storage, no cloud inference, deletion behavior, CC BY-NC teaching-data restriction, no identity inference, and prohibited high-risk uses. README summarizes the seven pages and links both documents.

- [ ] **Step 5: Run release scripts and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/deploy/test_scripts.py -q`

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1`

Expected: safety and complete deterministic release checks pass.

```powershell
git add scripts .github README.md docs backend/tests/deploy
git commit -m "chore: complete local release workflow"
```

## Final Release Gate

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1 -RealModel
git status --short
```

Expected: deterministic suite and real CPU ONNX smoke pass; Git status is clean. On a CUDA host also run `scripts/test.ps1 -Cuda` and record provider/device output. Manually train the standard profile on a reviewed published dataset and verify the UI reports real overall/per-class metrics, confusion matrix, errors, and an honest pass/fail against `mAP@0.5 >= 0.80`. Hardware-unavailable checks remain explicitly unverified.
