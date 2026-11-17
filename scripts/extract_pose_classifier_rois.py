#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

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


LOGGER = logging.getLogger("extract_pose_classifier_rois")
POSE_LABELS = ("correct_pose", "side_pose", "flipped_pose")
DEFAULT_CHECKPOINT = Path("data/detector_dataset_v1/runs/piston_detector_rfdetr_small_120e_run00/checkpoint_best_total.pth")
DEFAULT_DATASET_DIR = Path("data/detector_dataset_v1/Assembly_parts_detection.v2i.coco")


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required for ROI extraction. Install cv2/opencv-python.")
    return cv2


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    value = value.strip("_")
    return value or "sequence"


def _iter_frame_ids(total_frames: int, frame_step: int, max_frames: int | None) -> Iterable[int]:
    count = 0
    for frame_id in range(0, total_frames, max(1, int(frame_step))):
        if max_frames is not None and count >= int(max_frames):
            break
        yield frame_id
        count += 1


def _best_detection(detections: list[dict], part_label: str) -> dict | None:
    target = canonical_label(part_label)
    candidates = [det for det in detections if canonical_label(str(det.get("label", ""))) == target]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence", 0.0)))


def _clip_padded_bbox(bbox: list[float], width: int, height: int, pad_ratio: float) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * max(0.0, float(pad_ratio))
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(width), x2 + pad)
    y2 = min(float(height), y2 + pad)
    xi1, yi1 = int(np.floor(x1)), int(np.floor(y1))
    xi2, yi2 = int(np.ceil(x2)), int(np.ceil(y2))
    return [xi1, yi1, max(xi1 + 1, xi2), max(yi1 + 1, yi2)]


def _draw_overlay(frame: np.ndarray, bbox: list[int], label: str, score: float) -> np.ndarray:
    cv = _require_cv2()
    out = frame.copy()
    x1, y1, x2, y2 = bbox
    cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(
        out,
        f"{label} {score:.2f}",
        (x1, max(18, y1 - 6)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract object-specific pose-classifier ROI crops from a single-pose video using the RF-DETR object detector."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--part-label", required=True)
    parser.add_argument("--pose-label", required=True, choices=POSE_LABELS)
    parser.add_argument("--output-root", default="data/classifier_dataset_v1/raw_sequences")
    parser.add_argument("--sequence-name", default=None, help="Unique name for this single-pose video sequence.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-size", default="small", choices=["nano", "small", "medium", "large"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--bbox-pad", type=float, default=0.20)
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    cv = _require_cv2()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    part_label = canonical_label(args.part_label)
    pose_label = str(args.pose_label)
    sequence_name = _slugify(
        args.sequence_name
        or f"{video_path.stem}_{pose_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    sequence_dir = Path(args.output_root).resolve() / part_label / pose_label / sequence_name
    images_dir = sequence_dir / "images"
    overlays_dir = sequence_dir / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    detector = RFDETRObjectDetector(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        model_size=args.model_size,
        device=args.device,
        threshold=float(args.threshold),
    )

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)

    records: list[dict] = []
    requested_ids = set(_iter_frame_ids(total_frames, args.frame_step, args.max_frames))
    frame_id = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_id += 1
        if frame_id not in requested_ids:
            continue
        detections = detector.detect(frame)
        det = _best_detection(detections, part_label)
        if det is None:
            LOGGER.debug("No %s detection on frame %d", part_label, frame_id)
            continue
        bbox = _clip_padded_bbox(det["bbox"], width, height, float(args.bbox_pad))
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < int(args.min_crop_size) or crop.shape[1] < int(args.min_crop_size):
            LOGGER.debug("Skipping tiny crop on frame %d: %s", frame_id, crop.shape[:2])
            continue
        crop_name = f"{sequence_name}_frame_{frame_id:06d}.jpg"
        crop_path = images_dir / crop_name
        cv.imwrite(str(crop_path), crop)
        overlay_path = None
        if args.save_overlays:
            overlay_path = overlays_dir / crop_name
            cv.imwrite(str(overlay_path), _draw_overlay(frame, bbox, part_label, float(det.get("confidence", 0.0))))
        records.append(
            {
                "video_path": str(video_path),
                "sequence_name": sequence_name,
                "part_label": part_label,
                "pose_label": pose_label,
                "frame_id": int(frame_id),
                "timestamp": float(frame_id / fps) if fps > 0 else None,
                "crop_path": str(crop_path),
                "overlay_path": str(overlay_path) if overlay_path else None,
                "bbox_xyxy_padded": bbox,
                "bbox_xyxy_detector": [float(v) for v in det["bbox"]],
                "detection_confidence": float(det.get("confidence", 0.0)),
            }
        )

    cap.release()
    manifest = {
        "video_path": str(video_path),
        "sequence_name": sequence_name,
        "part_label": part_label,
        "pose_label": pose_label,
        "single_pose_sequence_required": True,
        "frame_step": int(args.frame_step),
        "bbox_pad": float(args.bbox_pad),
        "num_crops": len(records),
        "records": records,
    }
    (sequence_dir / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved %d crop(s) under %s", len(records), sequence_dir)
    print(sequence_dir)


if __name__ == "__main__":
    main()
