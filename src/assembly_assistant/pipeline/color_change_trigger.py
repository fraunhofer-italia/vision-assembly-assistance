from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class ColorTriggerResult:
    region_id: str
    baseline_hsv: np.ndarray
    current_hsv: np.ndarray
    delta: float
    score: float
    active: bool


class ColorChangeTrigger:
    """
    Color-change trigger per ROI.

    - Builds baseline HSV mean per ROI from first N frames.
    - Computes delta in HSV mean per ROI.
    - score = clamp((delta - threshold) / max(range, eps), 0, 1)
    """

    def __init__(
        self,
        *,
        roi_masks: Dict[str, np.ndarray],
        baseline_frames: int = 30,
        delta_threshold: float = 18.0,
        delta_range: float = 40.0,
        hue_weight: float = 0.5,
        sat_weight: float = 1.0,
        val_weight: float = 1.0,
    ):
        self.roi_masks = roi_masks
        self.baseline_frames = int(max(1, baseline_frames))
        self.delta_threshold = float(delta_threshold)
        self.delta_range = float(max(1e-6, delta_range))
        self.hue_weight = float(hue_weight)
        self.sat_weight = float(sat_weight)
        self.val_weight = float(val_weight)
        self._baseline_samples: Dict[str, list] = {rid: [] for rid in roi_masks}
        self._baseline_hsv: Dict[str, np.ndarray] = {}
        self._frames_seen = 0

    @property
    def ready(self) -> bool:
        return bool(self._baseline_hsv)

    def update(self, frame_bgr: np.ndarray) -> Optional[Dict[str, ColorTriggerResult]]:
        if frame_bgr is None:
            return None
        self._frames_seen += 1

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        if not self.ready:
            self._accumulate_baseline(hsv)
            if not self.ready:
                return None

        out: Dict[str, ColorTriggerResult] = {}
        for rid, mask in self.roi_masks.items():
            vals = hsv[mask > 0]
            if vals.size == 0:
                continue
            current = np.median(vals.reshape(-1, 3), axis=0)
            baseline = self._baseline_hsv.get(rid)
            if baseline is None:
                continue
            # Hue is circular: use shortest distance on [0, 180).
            dh = float(abs(current[0] - baseline[0]))
            dh = min(dh, 180.0 - dh)
            ds = float(current[1] - baseline[1])
            dv = float(current[2] - baseline[2])
            delta = float(
                ((self.hue_weight * dh) ** 2 + (self.sat_weight * ds) ** 2 + (self.val_weight * dv) ** 2) ** 0.5
            )
            score = max(0.0, min(1.0, (delta - self.delta_threshold) / self.delta_range))
            out[rid] = ColorTriggerResult(
                region_id=rid,
                baseline_hsv=baseline,
                current_hsv=current,
                delta=delta,
                score=score,
                active=bool(score > 0.0),
            )
        return out

    def _accumulate_baseline(self, hsv: np.ndarray) -> None:
        if self._frames_seen > self.baseline_frames:
            self._finalize_baseline()
            return
        for rid, mask in self.roi_masks.items():
            vals = hsv[mask > 0]
            if vals.size == 0:
                continue
            med = np.median(vals.reshape(-1, 3), axis=0)
            self._baseline_samples[rid].append(med)
        if self._frames_seen >= self.baseline_frames:
            self._finalize_baseline()

    def _finalize_baseline(self) -> None:
        if self._baseline_hsv:
            return
        for rid, vals in self._baseline_samples.items():
            if not vals:
                continue
            self._baseline_hsv[rid] = np.median(np.stack(vals, axis=0), axis=0)
        self._baseline_samples = {rid: [] for rid in self.roi_masks}
