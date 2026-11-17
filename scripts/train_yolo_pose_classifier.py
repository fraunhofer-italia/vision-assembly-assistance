#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO image classifier for object-specific pose classes.")
    parser.add_argument("--data", required=True, help="Dataset directory containing train/val/test class folders.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Classification checkpoint, e.g. yolov8n-cls.pt or yolov8s-cls.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="data/classifier_dataset_v1/runs")
    parser.add_argument("--name", default="front_cover_pose_yolov8n_cls")
    parser.add_argument("--degrees", type=float, default=3.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--fliplr", type=float, default=0.0, help="Usually keep 0: horizontal flips can change pose semantics.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Usually keep 0: vertical flips can change pose semantics.")
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("Ultralytics is not installed. Install it first with: `python3 -m pip install ultralytics`") from exc

    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    started = time.time()
    model = YOLO(str(args.model))
    results = model.train(
        data=str(data_dir),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        degrees=float(args.degrees),
        translate=float(args.translate),
        scale=float(args.scale),
        fliplr=float(args.fliplr),
        flipud=float(args.flipud),
    )
    ended = time.time()

    run_dir = Path(args.project) / args.name
    best_weights = run_dir / "weights" / "best.pt"
    report = {
        "model": str(args.model),
        "dataset": str(data_dir),
        "run_dir": str(run_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "started_at_unix": started,
        "ended_at_unix": ended,
        "duration_sec": ended - started,
    }
    report_path = Path(args.report_path) if args.report_path else run_dir / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(results)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
