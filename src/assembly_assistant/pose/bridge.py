from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

PILOT_PARTS = {"front_cover", "rear_cover", "piston_rod"}


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class PosePilotDecision:
    stage_name: str
    part_label: str
    bbox_xyxy: list[float]
    confidence: float
    frame_id: int
    timestamp: float
    rgb_frame_path: str | None
    approx_camera_k: list[list[float]]
    source: str = "real_offline_rgb"
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "timestamp": float(self.timestamp),
            "stage_name": self.stage_name,
            "part_label": self.part_label,
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "rgb_frame_path": self.rgb_frame_path,
            "camera_intrinsics_used": {"K": self.approx_camera_k},
            "source": self.source,
            "notes": self.notes,
        }


class PosePilotBridge:
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.stages = dict(self.config.get("stages") or {})
        self.min_confidence = float(self.config.get("min_bbox_confidence", 0.25))
        self.min_box_size_px = float(self.config.get("min_box_size_px", 24.0))
        self.approx_intrinsics = self.config.get("approx_camera_intrinsics") or {}

    @classmethod
    def from_json(cls, path: str | Path) -> "PosePilotBridge":
        return cls(_load_json(path))

    def is_relevant_stage(self, stage_name: str | None) -> bool:
        return bool(stage_name and str(stage_name).strip() in self.stages)

    def _approx_k(self, frame_shape) -> list[list[float]]:
        if self.approx_intrinsics:
            fx = float(self.approx_intrinsics.get("fx"))
            fy = float(self.approx_intrinsics.get("fy"))
            cx = float(self.approx_intrinsics.get("cx"))
            cy = float(self.approx_intrinsics.get("cy"))
            return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        h, w = frame_shape[:2]
        f = float(max(h, w))
        return [[f, 0.0, float(w) / 2.0], [0.0, f, float(h) / 2.0], [0.0, 0.0, 1.0]]

    def build_decision(
        self,
        *,
        frame_data: dict,
        stage_name: str | None,
        frame_id: int,
        timestamp: float,
        frame_shape,
        rgb_frame_path: str | None = None,
    ) -> Optional[PosePilotDecision]:
        stage_name = str(stage_name or "").strip()
        stage_cfg = self.stages.get(stage_name)
        if not stage_cfg:
            return None
        target_part = str(stage_cfg.get("part_label", "")).strip()
        if target_part not in PILOT_PARTS:
            return None
        active_part = frame_data.get("active_part") or {}
        active_label = str(active_part.get("label") or "").strip()
        bbox = active_part.get("bbox")
        conf = float(active_part.get("confidence") or 0.0)
        if active_label != target_part:
            return None
        if bbox is None or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if (x2 - x1) < self.min_box_size_px or (y2 - y1) < self.min_box_size_px:
            return None
        if conf < self.min_confidence:
            return None
        return PosePilotDecision(
            stage_name=stage_name,
            part_label=target_part,
            bbox_xyxy=[x1, y1, x2, y2],
            confidence=conf,
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            rgb_frame_path=rgb_frame_path,
            approx_camera_k=self._approx_k(frame_shape),
            notes=str(stage_cfg.get("notes", "")),
        )


class PosePilotExporter:
    def __init__(self, output_dir: str | Path, *, write_every_n_frames: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "pose_pilot_manifest.jsonl"
        self.write_every_n_frames = max(1, int(write_every_n_frames))
        self._seen = set()

    def export(self, *, frame: np.ndarray, decision: PosePilotDecision) -> Optional[Path]:
        key = (decision.stage_name, decision.part_label, int(decision.frame_id))
        if key in self._seen:
            return None
        if int(decision.frame_id) % self.write_every_n_frames != 0:
            return None
        self._seen.add(key)
        img_name = f"{decision.stage_name}_{decision.part_label}_{decision.frame_id:06d}.png"
        img_path = self.images_dir / img_name
        cv2.imwrite(str(img_path), frame)
        record = decision.to_record()
        record["rgb_frame_path"] = str(img_path)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return img_path
