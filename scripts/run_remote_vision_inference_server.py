#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.detectors.rfdetr_object_detector import (  # noqa: E402
    RFDETRObjectDetector,
)
from assembly_assistant.pipeline.video_extractors import (  # noqa: E402
    RFDETR_CHECKPOINT_DEFAULT,
    RFDETR_DATASET_DIR_DEFAULT,
)
from assembly_assistant.pose.pose_classifier import (  # noqa: E402
    PoseClassifierManager,
    parse_pose_classifier_model_map,
)
from assembly_assistant.utils.logger import get_logger  # noqa: E402


LOGGER = get_logger("remote_vision_inference_server")


def _decode_frame(image_b64: str):
    raw = base64.b64decode(str(image_b64).encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG frame.")
    return frame


def _filter_by_confidence(items: list[dict], threshold: float) -> list[dict]:
    out = []
    for item in items or []:
        try:
            conf = float(item.get("confidence", item.get("score", 0.0)))
        except Exception:
            conf = 0.0
        if conf >= float(threshold):
            out.append(dict(item))
    return out


class InferenceState:
    def __init__(self, args: argparse.Namespace):
        self.detector = None
        checkpoint = Path(args.detector_checkpoint)
        if checkpoint.exists():
            self.detector = RFDETRObjectDetector(
                checkpoint_path=checkpoint,
                dataset_dir=Path(args.detector_dataset_dir),
                model_size=args.detector_model_size,
                device=args.detector_device,
                threshold=float(args.detector_threshold),
            )
            LOGGER.info("Loaded RF-DETR detector: %s", checkpoint)
        else:
            LOGGER.warning("Detector checkpoint not found; object detection disabled: %s", checkpoint)

        self.pose_classifier = None
        if not args.disable_pose_classifier:
            self.pose_classifier = PoseClassifierManager.auto(
                model_map=parse_pose_classifier_model_map(args.pose_classifier_models),
                device=args.pose_classifier_device,
                bbox_pad=float(args.pose_classifier_bbox_pad),
                min_detection_confidence=float(args.pose_min_detection_confidence),
            )
            LOGGER.info("Pose classifier manager ready.")


def _make_handler(state: InferenceState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AssemblyRemoteVision/1.0"

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "detector_enabled": state.detector is not None,
                        "pose_classifier_enabled": state.pose_classifier is not None,
                    },
                )
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/infer":
                self._send_json(404, {"ok": False, "error": "not found"})
                return
            started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                frame = _decode_frame(payload.get("image_b64", ""))
                det_threshold = float(payload.get("detection_min_confidence", 0.0))
                pose_threshold = float(payload.get("pose_min_confidence", 0.0))
                expected_poses = payload.get("expected_poses") or []

                detections = state.detector.detect(frame) if state.detector is not None else []
                detections = _filter_by_confidence(detections, det_threshold)

                pose_predictions = []
                if state.pose_classifier is not None:
                    pose_predictions = state.pose_classifier.predict(
                        frame_bgr=frame,
                        detections=detections,
                        expected_poses=expected_poses,
                    )
                    pose_predictions = _filter_by_confidence(pose_predictions, pose_threshold)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "detections": detections,
                        "pose_predictions": pose_predictions,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    },
                )
            except Exception as exc:
                LOGGER.exception("Inference request failed")
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            LOGGER.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU-side RF-DETR + pose-classifier HTTP inference service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--detector-checkpoint", default=str(RFDETR_CHECKPOINT_DEFAULT))
    parser.add_argument("--detector-dataset-dir", default=str(RFDETR_DATASET_DIR_DEFAULT))
    parser.add_argument("--detector-model-size", choices=["nano", "small", "medium", "large"], default="small")
    parser.add_argument("--detector-device", default="cuda")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--disable-pose-classifier", action="store_true")
    parser.add_argument("--pose-classifier-models", default=None)
    parser.add_argument("--pose-classifier-device", default="cuda")
    parser.add_argument("--pose-classifier-bbox-pad", type=float, default=0.20)
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.0)
    args = parser.parse_args()

    state = InferenceState(args)
    server = ThreadingHTTPServer((args.host, int(args.port)), _make_handler(state))
    LOGGER.info("Remote vision inference server listening on http://%s:%d", args.host, int(args.port))
    server.serve_forever()


if __name__ == "__main__":
    main()
