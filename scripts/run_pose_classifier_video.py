#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assembly_assistant.detection_types.label_taxonomy import canonical_label
from assembly_assistant.detectors.rfdetr_object_detector import RFDETRObjectDetector


DEFAULT_CHECKPOINT = Path("data/detector_dataset_v1/runs/piston_detector_rfdetr_small_120e_run00/checkpoint_best_total.pth")
DEFAULT_DATASET_DIR = Path("data/detector_dataset_v1/Assembly_parts_detection.v2i.coco")


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required for pose-classifier video inference. Install cv2/opencv-python.")
    return cv2


def _clip_padded_bbox(bbox: list[float], width: int, height: int, pad_ratio: float) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * max(0.0, float(pad_ratio))
    return [
        int(max(0.0, np.floor(x1 - pad))),
        int(max(0.0, np.floor(y1 - pad))),
        int(min(float(width), np.ceil(x2 + pad))),
        int(min(float(height), np.ceil(y2 + pad))),
    ]


def _best_detection(detections: list[dict], part_label: str) -> dict | None:
    target = canonical_label(part_label)
    candidates = [det for det in detections if canonical_label(str(det.get("label", ""))) == target]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector + YOLO pose classifier on a video and save overlays/summary.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--classifier", required=True, help="YOLO classification weights, e.g. best.pt")
    parser.add_argument("--part-label", required=True)
    parser.add_argument("--output-dir", default="data/classifier_dataset_v1/inference_outputs/latest")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--detector-model-size", default="small", choices=["nano", "small", "medium", "large"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--bbox-pad", type=float, default=0.20)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    cv = _require_cv2()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("Ultralytics is not installed. Install it first with: `python3 -m pip install ultralytics`") from exc

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    output_dir = Path(args.output_dir).resolve()
    overlay_dir = output_dir / "overlays"
    crop_dir = output_dir / "crops"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    detector = RFDETRObjectDetector(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        model_size=args.detector_model_size,
        device=args.device,
        threshold=float(args.det_threshold),
    )
    classifier = YOLO(str(args.classifier))

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None
    if args.save_video:
        writer = cv.VideoWriter(
            str(output_dir / "pose_classifier_overlay.mp4"),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    records = []
    processed = 0
    frame_id = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_id += 1
        if frame_id % max(1, int(args.frame_step)) != 0:
            continue
        if args.max_frames is not None and processed >= int(args.max_frames):
            break
        processed += 1
        overlay = frame.copy()
        det = _best_detection(detector.detect(frame), args.part_label)
        if det is None:
            records.append({"frame_id": int(frame_id), "detection": None, "prediction": None})
            if writer is not None:
                writer.write(overlay)
            continue
        bbox = _clip_padded_bbox(det["bbox"], width, height, float(args.bbox_pad))
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        crop_path = crop_dir / f"frame_{frame_id:06d}.jpg"
        cv.imwrite(str(crop_path), crop)
        cls_results = classifier.predict(source=crop, device=args.device, verbose=False)
        probs = cls_results[0].probs
        top1 = int(probs.top1)
        top1_conf = float(probs.top1conf)
        names = cls_results[0].names or {}
        pred_name = str(names.get(top1, top1))
        cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(
            overlay,
            f"{pred_name} {top1_conf:.2f}",
            (x1, max(18, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        overlay_path = overlay_dir / f"frame_{frame_id:06d}.jpg"
        cv.imwrite(str(overlay_path), overlay)
        if writer is not None:
            writer.write(overlay)
        records.append(
            {
                "frame_id": int(frame_id),
                "crop_path": str(crop_path),
                "overlay_path": str(overlay_path),
                "detection": {
                    "bbox_xyxy_padded": bbox,
                    "bbox_xyxy_detector": [float(v) for v in det["bbox"]],
                    "confidence": float(det.get("confidence", 0.0)),
                },
                "prediction": {
                    "class_id": top1,
                    "class_name": pred_name,
                    "confidence": top1_conf,
                },
            }
        )

    cap.release()
    if writer is not None:
        writer.release()
    summary = {
        "video": str(video_path),
        "classifier": str(args.classifier),
        "part_label": canonical_label(args.part_label),
        "records": records,
    }
    (output_dir / "pose_classifier_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
