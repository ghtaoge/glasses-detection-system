"""隔离 Ultralytics 的训练子进程入口。

worker 只读取父进程生成的 JSON 请求，通过 JSONL 追加事件，并在所有模型产物验证
完成后写入 result.json。它不接收任意 shell 参数或用户提供的输出路径。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def write_event(path: Path, event: dict) -> None:
    """写入一条完整事件并刷盘，父进程只会读取完整 JSON 行。"""

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
        # 取消标记在 epoch 回调中检查，让 Ultralytics 有机会完成当前批次并释放资源。
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

    # 注册模型前必须在 CPU provider 上实际执行一次。导出成功不等于运行时可以
    # 加载；该检查可提前发现不支持的算子、损坏文件和非有限输出。
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
