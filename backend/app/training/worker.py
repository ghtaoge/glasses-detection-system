import argparse
import hashlib
import json
import os
from pathlib import Path


def write_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    events_path = Path(args.events)
    cancel_path = Path(request["output_dir"]) / "cancel"
    from ultralytics import YOLO

    model = YOLO(request["model"])

    def epoch_callback(trainer) -> None:
        if cancel_path.exists():
            raise KeyboardInterrupt("training cancelled")
        metrics = trainer.metrics or {}
        write_event(
            events_path,
            {
                "type": "epoch",
                "epoch": int(trainer.epoch + 1),
                "epochs": request["epochs"],
                "loss": float(trainer.loss.sum()) if trainer.loss is not None else 0.0,
                "precision": float(metrics.get("metrics/precision(B)", 0)),
                "recall": float(metrics.get("metrics/recall(B)", 0)),
                "map50": float(metrics.get("metrics/mAP50(B)", 0)),
                "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0)),
            },
        )

    model.add_callback("on_fit_epoch_end", epoch_callback)
    train_result = model.train(
        data=request["data_yaml"],
        epochs=request["epochs"],
        imgsz=request["image_size"],
        batch=request["batch_size"],
        patience=request["patience"],
        device=request["device"],
        seed=20260723,
        deterministic=True,
        workers=0,
        project=request["output_dir"],
        name="train",
        exist_ok=False,
    )
    best = Path(train_result.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(
        data=request["data_yaml"],
        split="test",
        plots=True,
        project=request["output_dir"],
        name="test",
    )
    onnx_path = Path(trained.export(format="onnx", dynamic=True, simplify=True, opset=17))
    names = [trained.names[index] for index in sorted(trained.names)]
    if names != ["no_glasses", "eyeglasses", "sunglasses"]:
        raise RuntimeError("exported class map does not match")
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = [1, 3, request["image_size"], request["image_size"]]
    outputs = session.run(None, {model_input.name: np.zeros(shape, dtype=np.float32)})
    if not outputs or not all(np.isfinite(output).all() for output in outputs):
        raise RuntimeError("exported ONNX smoke test failed")
    per_class = {
        names[index]: {"map50": float(metrics.box.maps[index])} for index in range(len(names))
    }
    result = {
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "metrics": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class": per_class,
            "simulated": False,
        },
        "device": request["device"],
    }
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
