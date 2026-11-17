from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2


@dataclass
class RemoteInferenceResult:
    detections: list[dict]
    pose_predictions: list[dict]
    latency_ms: float
    remote_latency_ms: float | None = None


class RemoteVisionInferenceClient:
    """HTTP JSON client for offloading object and pose inference to a GPU host."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float = 2.0,
        jpeg_quality: int = 80,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.jpeg_quality = int(max(30, min(100, jpeg_quality)))

    def _encode_frame(self, frame_bgr) -> str:
        ok, buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame for remote inference.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def infer(
        self,
        frame_bgr,
        *,
        expected_poses: list[dict] | None = None,
        detection_min_confidence: float = 0.0,
        pose_min_confidence: float = 0.0,
        frame_id: int | None = None,
        stage_name: str | None = None,
    ) -> RemoteInferenceResult:
        payload: dict[str, Any] = {
            "image_b64": self._encode_frame(frame_bgr),
            "expected_poses": expected_poses or [],
            "detection_min_confidence": float(detection_min_confidence),
            "pose_min_confidence": float(pose_min_confidence),
            "frame_id": frame_id,
            "stage_name": stage_name,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote inference request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not bool(data.get("ok", False)):
            raise RuntimeError(f"Remote inference server returned an error: {data.get('error')}")
        return RemoteInferenceResult(
            detections=[dict(x) for x in (data.get("detections") or []) if isinstance(x, dict)],
            pose_predictions=[dict(x) for x in (data.get("pose_predictions") or []) if isinstance(x, dict)],
            latency_ms=float(latency_ms),
            remote_latency_ms=(
                float(data["latency_ms"])
                if data.get("latency_ms") is not None
                else None
            ),
        )
