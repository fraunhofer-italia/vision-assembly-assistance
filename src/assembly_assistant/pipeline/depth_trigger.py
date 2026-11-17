from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DepthTriggerResult:
    region_id: str
    baseline_mm: float
    current_mm: float
    delta_mm: float
    dwell_frames: int
    active: bool


class DepthTriggerExtractor:
    """
    Depth-based ROI trigger helper.

    Usage:
      - call update(depth_frame) every frame
      - baseline is built from the first baseline_frames frames
      - once ready, update returns per-region trigger info
    """

    def __init__(
        self,
        *,
        roi_masks: Dict[str, np.ndarray],
        baseline_frames: int = 30,
        delta_mm: float = 25.0,
        dwell_frames: int = 6,
    ):
        self.roi_masks = roi_masks
        self.baseline_frames = int(max(1, baseline_frames))
        self.delta_mm = float(delta_mm)
        self.dwell_frames = int(max(1, dwell_frames))

        self._baseline_samples: Dict[str, list] = {rid: [] for rid in roi_masks}
        self._baseline_mm: Dict[str, float] = {}
        self._dwell_counts: Dict[str, int] = {rid: 0 for rid in roi_masks}
        self._frames_seen = 0

    @property
    def ready(self) -> bool:
        return bool(self._baseline_mm)

    def baseline(self) -> Dict[str, float]:
        return dict(self._baseline_mm)

    def update(self, depth_mm: np.ndarray) -> Optional[Dict[str, DepthTriggerResult]]:
        if depth_mm is None:
            return None
        self._frames_seen += 1

        if not self.ready:
            self._accumulate_baseline(depth_mm)
            if not self.ready:
                return None

        out: Dict[str, DepthTriggerResult] = {}
        for rid, mask in self.roi_masks.items():
            roi = depth_mm[mask > 0]
            if roi.size == 0:
                continue
            current = float(np.median(roi))
            baseline = float(self._baseline_mm.get(rid, 0.0))
            delta = float(baseline - current)
            if delta >= self.delta_mm:
                self._dwell_counts[rid] = self._dwell_counts.get(rid, 0) + 1
            else:
                self._dwell_counts[rid] = 0
            active = self._dwell_counts[rid] >= self.dwell_frames
            out[rid] = DepthTriggerResult(
                region_id=rid,
                baseline_mm=baseline,
                current_mm=current,
                delta_mm=delta,
                dwell_frames=int(self._dwell_counts[rid]),
                active=bool(active),
            )
        return out

    def _accumulate_baseline(self, depth_mm: np.ndarray) -> None:
        if self._frames_seen > self.baseline_frames:
            self._finalize_baseline()
            return
        for rid, mask in self.roi_masks.items():
            roi = depth_mm[mask > 0]
            if roi.size == 0:
                continue
            self._baseline_samples[rid].append(float(np.median(roi)))
        if self._frames_seen >= self.baseline_frames:
            self._finalize_baseline()

    def _finalize_baseline(self) -> None:
        if self._baseline_mm:
            return
        for rid, vals in self._baseline_samples.items():
            if not vals:
                continue
            self._baseline_mm[rid] = float(np.median(vals))
        # Release memory.
        self._baseline_samples = {rid: [] for rid in self.roi_masks}
