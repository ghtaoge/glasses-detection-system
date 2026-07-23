# Training, Evaluation, and Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Web-managed YOLO training, persistent progress events, frozen-test evaluation, verified ONNX export, and an explicit model registry and activation workflow.

**Architecture:** Persist task state and append-only events before introducing subprocesses. Run a narrow JSON request/result worker protocol in a supervised child process, keep Ultralytics behind an adapter, evaluate against the versioned test split, then register and atomically activate validated ONNX artifacts.

**Tech Stack:** Phase-one stack plus Ultralytics 8.4.104, `yolo26n.pt`, PyTorch installed according to the official CPU/CUDA selector, ONNX Runtime 1.27.0, SSE, ECharts 6.1.0.

**Depends on:** [Phase one](2026-07-23-glasses-phase-1-data-annotation.md)

---

## Locked File Map

```text
backend/app/domain/training.py             Training settings, states, events
backend/app/repositories/training.py       Task/event persistence
backend/app/repositories/models_registry.py Model persistence
backend/app/services/training_commands.py  Whitelisted request builder
backend/app/services/training_supervisor.py Child lifecycle and cancellation
backend/app/training/worker.py              JSON worker entry point
backend/app/training/ultralytics_adapter.py YOLO train/val/export isolation
backend/app/services/evaluation.py          Frozen-test result normalization
backend/app/services/model_registry.py      Artifact checks and activation
backend/app/api/training.py                 Task and SSE endpoints
backend/app/api/models.py                   Model endpoints
backend/migrations/versions/0002_training.py Training/model schema
frontend/src/views/TrainingView.vue         Training configuration and metrics
frontend/src/views/ModelsView.vue           Registry comparison and activation
frontend/src/components/MetricChart.vue     Stable ECharts wrapper
frontend/e2e/training-models.spec.ts         Milestone browser acceptance
```

### Task 1: Training Domain and Persistent State Machine

**Files:**
- Create: `backend/app/domain/training.py`
- Create: `backend/app/repositories/training.py`
- Create: `backend/app/repositories/models_registry.py`
- Create: `backend/migrations/versions/0002_training.py`
- Test: `backend/tests/repositories/test_training.py`

- [ ] **Step 1: Write transition and event-order tests**

```python
@pytest.mark.asyncio
async def test_training_transitions_and_events_are_durable(training_repo, version_id):
    task = await training_repo.create(version_id, {"preset":"quick"})
    await training_repo.transition(task.id, "running")
    first = await training_repo.append_event(task.id, "epoch", {"epoch":1,"map50":0.42})
    second = await training_repo.append_event(task.id, "epoch", {"epoch":2,"map50":0.55})
    assert second.sequence == first.sequence + 1
    assert [e.sequence for e in await training_repo.events_after(task.id, first.sequence)] == [second.sequence]

@pytest.mark.asyncio
async def test_completed_task_cannot_restart(training_repo, version_id):
    task = await training_repo.create(version_id, {})
    await training_repo.transition(task.id, "running")
    await training_repo.transition(task.id, "completed")
    with pytest.raises(ValueError, match="completed -> running"):
        await training_repo.transition(task.id, "running")
```

- [ ] **Step 2: Run and verify missing training repository**

Run: `.venv\Scripts\python -m pytest backend/tests/repositories/test_training.py -q`

Expected: FAIL because training modules do not exist.

- [ ] **Step 3: Implement exact domain contracts**

```python
class TrainingState(StrEnum):
    QUEUED="queued"; RUNNING="running"; CANCELLING="cancelling"
    COMPLETED="completed"; FAILED="failed"; INTERRUPTED="interrupted"

ALLOWED_TRANSITIONS = {
    TrainingState.QUEUED:{TrainingState.RUNNING,TrainingState.CANCELLING,TrainingState.INTERRUPTED},
    TrainingState.RUNNING:{TrainingState.CANCELLING,TrainingState.COMPLETED,TrainingState.FAILED,TrainingState.INTERRUPTED},
    TrainingState.CANCELLING:{TrainingState.FAILED,TrainingState.INTERRUPTED},
    TrainingState.COMPLETED:set(), TrainingState.FAILED:set(), TrainingState.INTERRUPTED:set(),
}

@dataclass(frozen=True, slots=True)
class TrainingSettings:
    preset: Literal["quick","standard"]
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    device: Literal["auto","cpu","cuda"]
```

Validate quick defaults `(5, 416, 4, 3)` and standard defaults `(80, 640, 16, 15)`, with ranges epochs `1..300`, image size `{320,416,512,640,768}`, batch `1..128`, and patience `0..50`. Persist `TrainingTaskRow`, unique `(task_id, sequence)` `TrainingEventRow`, and `ModelRow` with source task, dataset version, artifact paths/checksums, class map, metrics JSON, status, active flag, and timestamps.

- [ ] **Step 4: Apply migration and run tests**

Run: `.venv\Scripts\alembic -c backend/alembic.ini upgrade head`

Run: `.venv\Scripts\python -m pytest backend/tests/repositories/test_training.py -q`

Expected: migration applies and all transition, sequence, single-running-task, and active-model uniqueness tests pass.

- [ ] **Step 5: Commit training persistence**

```powershell
git add backend/app/domain/training.py backend/app/repositories backend/migrations backend/tests/repositories
git commit -m "feat: persist training and model state"
```

### Task 2: Whitelisted Training Requests and YOLO Snapshot Launch

**Files:**
- Create: `backend/app/services/training_commands.py`
- Test: `backend/tests/services/test_training_commands.py`

- [ ] **Step 1: Write command-boundary tests**

```python
def test_builds_worker_request_without_shell_text(builder, published_version):
    request = builder.build(published_version, TrainingSettings("standard",80,640,16,15,"auto"))
    assert request["model"] == "yolo26n.pt"
    assert request["data_yaml"].endswith("data.yaml")
    assert request["classes"] == ["no_glasses","eyeglasses","sunglasses"]
    assert "command" not in request

def test_rejects_unpublished_version(builder, draft_version):
    with pytest.raises(Exception, match="DATASET_VERSION_NOT_PUBLISHED"):
        builder.build(draft_version, TrainingSettings("quick",5,416,4,3,"cpu"))
```

- [ ] **Step 2: Run and verify missing builder**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_training_commands.py -q`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement a serializable request, not arbitrary CLI**

```python
@dataclass(frozen=True, slots=True)
class WorkerRequest:
    task_id: str; model: str; data_yaml: str; output_dir: str
    epochs: int; image_size: int; batch_size: int; patience: int; device: str
    seed: int = 20260723; deterministic: bool = True; workers: int = 0
```

Resolve all paths under managed dataset/task roots, verify manifest checksum immediately before launch, map `auto` to `0` only when CUDA is available and otherwise `cpu`, and serialize with `dataclasses.asdict`. Never accept model names, project paths, extra arguments, callbacks, or shell fragments from the API.

- [ ] **Step 4: Run builder tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_training_commands.py -q`

Expected: path, checksum, class map, device, range, and injection tests pass.

```powershell
git add backend/app/services/training_commands.py backend/tests/services/test_training_commands.py
git commit -m "feat: build reproducible training requests"
```

### Task 3: Supervised Worker and Cancellation

**Files:**
- Create: `backend/app/services/training_supervisor.py`
- Create: `backend/app/training/worker.py`
- Test: `backend/tests/services/test_training_supervisor.py`

- [ ] **Step 1: Write fake-worker lifecycle tests**

```python
@pytest.mark.asyncio
async def test_supervisor_persists_worker_events(supervisor, fake_worker, queued_task):
    fake_worker.events = [{"type":"started"},{"type":"epoch","epoch":1,"map50":0.4},{"type":"completed","result":"result.json"}]
    await supervisor.run(queued_task.id)
    assert (await supervisor.repository.get(queued_task.id)).state == "completed"

@pytest.mark.asyncio
async def test_cancel_escalates_after_grace_period(supervisor, hanging_worker, running_task):
    await supervisor.cancel(running_task.id, grace_seconds=0.01)
    assert hanging_worker.terminate_called
    assert (await supervisor.repository.get(running_task.id)).state == "failed"
```

- [ ] **Step 2: Run and verify missing supervisor**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_training_supervisor.py -q`

Expected: FAIL because the supervisor does not exist.

- [ ] **Step 3: Implement a one-task process supervisor**

```python
argv = [sys.executable, "-m", "app.training.worker", "--request", str(request_path), "--events", str(events_path)]
process = await asyncio.create_subprocess_exec(*argv, stdout=PIPE, stderr=PIPE, creationflags=hidden_window_flag())
```

Use an `asyncio.Lock` plus repository uniqueness check to allow one active task. The worker reads one JSON file, emits one JSON object per line to its events file using flush+fsync, writes `result.json` atomically, and exits `0` only after all artifacts exist. The supervisor tails complete lines, validates event schemas, persists them, caps displayed log lines at 5,000, and records stderr in a managed log. Cancellation writes a task-local cancel marker checked between epochs, waits 10 seconds, then terminates the exact process object. Startup reconciliation changes leftover `queued`, `running`, or `cancelling` tasks to `interrupted` without deleting checkpoints.

- [ ] **Step 4: Run lifecycle tests**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_training_supervisor.py -q`

Expected: completion, malformed event, nonzero exit, cancellation, exact-process termination, and startup reconciliation tests pass.

- [ ] **Step 5: Commit supervision**

```powershell
git add backend/app/services/training_supervisor.py backend/app/training/worker.py backend/tests/services/test_training_supervisor.py
git commit -m "feat: supervise local training jobs"
```

### Task 4: Ultralytics Train, Evaluate, and Export Adapter

**Files:**
- Create: `backend/app/training/ultralytics_adapter.py`
- Create: `backend/app/services/evaluation.py`
- Test: `backend/tests/training/test_ultralytics_adapter.py`
- Test: `backend/tests/services/test_evaluation.py`

- [ ] **Step 1: Write adapter isolation and metric tests**

```python
def test_adapter_passes_locked_training_arguments(fake_yolo, request):
    UltralyticsAdapter(fake_yolo).train(request)
    assert fake_yolo.train_kwargs == {"data":request.data_yaml,"epochs":request.epochs,
      "imgsz":request.image_size,"batch":request.batch_size,"patience":request.patience,
      "device":request.device,"seed":20260723,"deterministic":True,"workers":0,
      "project":request.output_dir,"name":"train","exist_ok":False}

def test_evaluation_uses_frozen_test_split(normalizer, fake_metrics):
    report = normalizer.normalize(fake_metrics, split="test")
    assert report.split == "test"
    assert report.passed is (report.map50 >= 0.80)
    assert set(report.per_class) == {"no_glasses","eyeglasses","sunglasses"}
```

- [ ] **Step 2: Run and verify missing ML adapters**

Run: `.venv\Scripts\python -m pytest backend/tests/training/test_ultralytics_adapter.py backend/tests/services/test_evaluation.py -q`

Expected: FAIL because adapters do not exist.

- [ ] **Step 3: Implement exact adapter calls**

Load the checksum-verified `data/resources/yolo26n.pt`, register an epoch-end callback that emits epoch, losses, precision, recall, map50, and map50_95, then call `train` with only the asserted arguments. After training, load `best.pt`, call `val(data=data_yaml, split="test", plots=True, project=output_dir, name="test", exist_ok=False)`, normalize overall and per-class values, copy confusion matrix and error-sample indexes, and call `export(format="onnx", dynamic=True, simplify=True, opset=17)`. Treat callback exceptions, missing best weights, missing three-class metrics, NaN/Inf, export failure, or cancel marker as explicit worker failure codes.

- [ ] **Step 4: Verify ONNX before reporting completion**

Create an `onnxruntime.InferenceSession` with CPU provider, assert one image input with four dimensions, run a zero tensor at the exported input size, and assert finite output arrays. Compute SHA-256 for `best.pt`, ONNX, metrics JSON, and confusion matrix. Write all hashes and exact Ultralytics/PyTorch/device versions into `result.json`.

- [ ] **Step 5: Run adapter tests**

Run: `.venv\Scripts\python -m pytest backend/tests/training/test_ultralytics_adapter.py backend/tests/services/test_evaluation.py -q`

Expected: locked calls, callback normalization, frozen test split, threshold, class coverage, ONNX validation, cancellation, and nonfinite metric tests pass with fakes.

- [ ] **Step 6: Commit ML adapter**

```powershell
git add backend/app/training/ultralytics_adapter.py backend/app/services/evaluation.py backend/tests
git commit -m "feat: train evaluate and export YOLO models"
```

### Task 5: Model Registry and Atomic Activation

**Files:**
- Create: `backend/app/services/model_registry.py`
- Test: `backend/tests/services/test_model_registry.py`

- [ ] **Step 1: Write registration and activation tests**

```python
@pytest.mark.asyncio
async def test_registration_preserves_unmet_threshold(registry, completed_result):
    completed_result.metrics["map50"] = 0.79
    model = await registry.register(completed_result)
    assert model.quality_status == "below_target"
    assert not model.active

@pytest.mark.asyncio
async def test_activation_validates_before_switch(registry, active_model, corrupt_candidate):
    with pytest.raises(Exception, match="MODEL_CHECKSUM_MISMATCH"):
        await registry.activate(corrupt_candidate.id)
    assert (await registry.active()).id == active_model.id
```

- [ ] **Step 2: Run and verify missing registry**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_model_registry.py -q`

Expected: FAIL because model registry does not exist.

- [ ] **Step 3: Implement validation and transaction order**

Registration resolves artifacts under the task output root, recomputes checksums, loads metrics, verifies exact classes, copies artifacts atomically into `models/<model_id>/`, and writes the model row last. Activation takes an application lock, recomputes ONNX checksum, creates and smoke-tests a candidate session through an injected loader, then updates old/new active flags in one transaction and swaps the in-memory session only after commit. A failed swap restores database flags and retains the previous session. Active models cannot be deleted.

- [ ] **Step 4: Run registry tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/services/test_model_registry.py -q`

Expected: threshold, artifact boundary, checksum, class mismatch, load failure, rollback, active-delete, and successful switch tests pass.

```powershell
git add backend/app/services/model_registry.py backend/tests/services/test_model_registry.py
git commit -m "feat: register and activate verified models"
```

### Task 6: Training and Model APIs with Replayable SSE

**Files:**
- Create: `backend/app/api/training.py`
- Create: `backend/app/api/models.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/api/test_training.py`
- Test: `backend/tests/api/test_models.py`

- [ ] **Step 1: Write API and reconnect tests**

```python
def test_sse_replays_after_last_event_id(client, task_with_events):
    with client.stream("GET", f"/api/training/{task_with_events.id}/events", headers={"Last-Event-ID":"1"}) as response:
        text = next(response.iter_text())
    assert "id: 2" in text and "id: 1" not in text

def test_arbitrary_training_key_is_rejected(client, version_id):
    response = client.post("/api/training", json={"dataset_version_id":version_id,"preset":"quick","shell":"calc.exe"})
    assert response.status_code == 422
```

- [ ] **Step 2: Run and verify missing routes**

Run: `.venv\Scripts\python -m pytest backend/tests/api/test_training.py backend/tests/api/test_models.py -q`

Expected: FAIL with `404`.

- [ ] **Step 3: Add exact endpoints**

```text
POST /api/training
GET  /api/training
GET  /api/training/{id}
POST /api/training/{id}/cancel
POST /api/training/{id}/resume
GET  /api/training/{id}/events
GET  /api/models
GET  /api/models/{id}
POST /api/models/{id}/activate
POST /api/models/{id}/sample-test
GET  /api/models/{id}/download
DELETE /api/models/{id}
```

Resume is permitted only for `interrupted` tasks with an existing checkpoint and creates a new task linked by `resumed_from_id`; it never mutates the old task. SSE uses event `id`, named event type, JSON data, heartbeat comments every 15 seconds, and `Cache-Control: no-cache`. Downloads set attachment name and verify checksum before streaming.

- [ ] **Step 4: Run API tests and commit**

Run: `.venv\Scripts\python -m pytest backend/tests/api/test_training.py backend/tests/api/test_models.py -q`

Expected: schema, one-active-task, cancel, resume, replay, heartbeat, activation, download, and protected-delete tests pass.

```powershell
git add backend/app/api backend/app/main.py backend/tests/api
git commit -m "feat: expose training and model APIs"
```

### Task 7: Training and Model Workbench

**Files:**
- Create: `frontend/src/views/TrainingView.vue`
- Create: `frontend/src/views/ModelsView.vue`
- Create: `frontend/src/components/MetricChart.vue`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/router.ts`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/package.json`
- Test: `frontend/tests/training-view.test.ts`
- Test: `frontend/tests/models-view.test.ts`
- Test: `frontend/e2e/training-models.spec.ts`

- [ ] **Step 1: Write state and activation tests**

```typescript
it('does not label a sub-threshold model as passed', async () => {
  mockApi.model({ map50: 0.79, qualityStatus: 'below_target', active: false })
  const wrapper = mount(ModelsView)
  await flushPromises()
  expect(wrapper.get('[data-testid="quality-status"]').text()).toContain('未达标')
  expect(wrapper.text()).toContain('79.0%')
})
```

- [ ] **Step 2: Run and verify missing views**

Run: `npm --prefix frontend test`

Expected: FAIL because training/model views do not exist.

- [ ] **Step 3: Implement training state and event reducer**

Use typed `TrainingTask`, `TrainingEvent`, `EvaluationReport`, and `ModelRecord`. Open SSE with persisted `lastEventId`, deduplicate by sequence, sort epoch metrics by epoch, and refetch task state after reconnect. The form exposes only preset, epochs, image size, batch, patience, and device. Disable edits while submitting; cancellation requires confirmation and displays `cancelling` until the server decides the terminal state.

- [ ] **Step 4: Implement charts and model comparison**

`MetricChart` owns one ECharts instance, uses `ResizeObserver`, disposes on unmount, and has fixed minimum height. Training page shows status, actual device, progress, log tail, loss/mAP curves, per-class metrics, confusion matrix, and error samples. Models page shows dataset version, map values, quality status, artifact checksum, active marker, sample test, explicit activation, download, and protected deletion. Do not imply that activation means quality passed.

- [ ] **Step 5: Add browser acceptance and run phase tests**

```typescript
test('runs a fake training job and activates its model', async ({ page }) => {
  await page.goto('/training')
  await page.getByRole('button', { name: '开始训练' }).click()
  await expect(page.getByText('训练完成')).toBeVisible()
  await expect(page.getByText('mAP@0.5')).toBeVisible()
  await page.goto('/models')
  await page.getByRole('button', { name: '启用模型' }).click()
  await expect(page.getByText('当前模型')).toBeVisible()
})
```

Run: `.venv\Scripts\python -m pytest -q`

Run: `npm --prefix frontend test`

Run: `npm --prefix frontend run build`

Run: `npm --prefix frontend run test:e2e -- training-models.spec.ts`

Expected: all backend, frontend, build, and browser checks pass using fake workers.

- [ ] **Step 6: Commit the workbench**

```powershell
git add frontend
git commit -m "feat: add training and model workbench"
```

## Phase Two Exit Gate

Run the full deterministic suite, then opt in to one real quick run:

```powershell
.venv\Scripts\python -m ruff check backend
.venv\Scripts\python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e -- training-models.spec.ts
$env:GLASSES_REAL_TRAINING='1'
.venv\Scripts\python -m pytest backend/tests/real/test_quick_training.py -q
```

Expected: deterministic checks pass; the real check trains one epoch on a tiny fixture, evaluates the frozen test split, exports a CPU-loadable ONNX model, and records real metrics without claiming the `0.80` quality target.
