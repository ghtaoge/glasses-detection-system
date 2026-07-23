# Glasses Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved local teaching system for dataset preparation, YOLO training, model evaluation, multi-face image inference, realtime camera inference, and auditable local history.

**Architecture:** Deliver four dependency-ordered milestones. Each milestone keeps domain contracts stable, ends with a runnable application and its own acceptance gate, and is committed before the next milestone starts.

**Tech Stack:** Python 3.11, FastAPI 0.139.2, SQLAlchemy 2.0.51, SQLite, Ultralytics 8.4.104 with `yolo26n.pt`, ONNX Runtime 1.27.0, OpenCV, Vue 3.5.40, TypeScript, Vite, Vitest, Playwright.

**Approved design:** [2026-07-23-glasses-detection-system-design.md](../specs/2026-07-23-glasses-detection-system-design.md)

---

## Milestones

1. [Foundation, dataset, and annotation](2026-07-23-glasses-phase-1-data-annotation.md)
2. [Training, evaluation, and model registry](2026-07-23-glasses-phase-2-training-models.md)
3. [Image inference and history](2026-07-23-glasses-phase-3-inference-history.md)
4. [Realtime camera and release acceptance](2026-07-23-glasses-phase-4-camera-release.md)

## Locked Cross-Milestone Contracts

```python
CLASS_NAMES = ("no_glasses", "eyeglasses", "sunglasses")

class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(TypedDict):
    box: BoundingBox
    class_id: int
    class_name: Literal["no_glasses", "eyeglasses", "sunglasses"]
    confidence: float
```

All persisted datetimes are UTC. All persisted paths are POSIX-style paths relative to the configured managed root. Bounding boxes use original-image pixels at API and persistence boundaries, and YOLO-normalized coordinates only inside exported training snapshots.

Training task states are `queued`, `running`, `cancelling`, `completed`, `failed`, and `interrupted`. A model becomes active only through an explicit activation request after ONNX load validation and exact class-map validation. Camera messages always carry monotonically increasing `frame_id` values.

## Execution Order

- Execute milestones strictly in order; later plans assume earlier commits exist.
- Use the test-first step order written in each task.
- Commit after every task using the exact commit scope listed in the plan.
- Do not download datasets or model weights in default unit tests.
- Keep real-model, Kaggle, and CUDA checks opt-in and report skipped hardware checks as unverified, never passed.

## Final Exit Gate

The project is complete only after all four milestone exit gates pass, the working tree is clean, the CPU ONNX smoke test passes, desktop and mobile browser screenshots have no overlap or blank canvas, and a real trained model's frozen-test evaluation is displayed honestly against `mAP@0.5 >= 0.80`.

## Spec Coverage Self-Review

| Design requirement | Implementation tasks |
| --- | --- |
| Three fixed classes and whole-face boxes | Phase 1 Tasks 2, 6-8 |
| Public seed data, license gate, local import | Phase 1 Tasks 4-5 |
| Pre-annotation and manual multi-face review | Phase 1 Tasks 6, 8 |
| Deduplication, immutable versions, 70/15/15 splits | Phase 1 Tasks 3, 7 |
| Web training, cancellation, interruption, resume | Phase 2 Tasks 1-3, 6-7 |
| Frozen-test metrics, confusion matrix, honest 0.80 gate | Phase 2 Tasks 4-5, 7 |
| Verified ONNX model registry and manual activation | Phase 2 Tasks 4-7 |
| Multi-face image detection and annotated history | Phase 3 Tasks 1-5 |
| Recoverable record/file deletion | Phase 3 Tasks 3-5; Phase 4 Task 4 |
| Latest-frame camera detection and explicit snapshots | Phase 4 Tasks 1-3 |
| CPU/CUDA selection and training resource policy | Phase 2 Tasks 2-4; Phase 4 Task 2 |
| Seven-page operational UI | Phase 1 Task 8; Phase 2 Task 7; Phase 3 Task 5; Phase 4 Tasks 3-4 |
| Input safety, path safety, privacy, and no cloud inference | Phase 1 Tasks 4-6; Phase 3 Tasks 2-4; Phase 4 Tasks 1, 4, 6 |
| Unit, integration, browser, responsive, and accessibility tests | Every task; Phase 4 Task 5 and final gate |
| Windows local setup and offline-after-preparation operation | Phase 4 Task 6 |

Self-review found no uncovered design requirement. Authentication, distributed training, video-file analysis, fine-grained frame attributes, native mobile apps, and high-risk automated decisions remain explicitly excluded.
