from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class ROIMotionResult:
    region_id: str
    motion_score: float
    active: bool
    mean_history: float


class ROIMotionTrigger:
    """
    Simple ROI motion-energy trigger using background subtraction.

    Background is fixed as the median of the first N gray frames.

    Two extra robustness guards make this less video-sensitive when whole-frame
    motion rises from arm sweeps, shadows, or exposure shifts:
    - relative_threshold_cap prevents the global-motion gate from becoming
      unrealistically strict in busy frames.
    - local_advantage_margin allows a region to activate when it is clearly more
      active than the rest of the frame, even if the pure ratio test is harsh.
    """

    def __init__(
        self,
        *,
        roi_masks: Dict[str, np.ndarray],
        baseline_frames: int = 30,
        ignore_frames: int = 0,
        diff_threshold: int = 18,
        score_threshold: float = 0.02,
        dwell_frames: int = 6,
        history_frames: int = 10,
        blur_kernel: tuple[int, int] = (5, 5),
        global_suppress_threshold: float = 0.03,
        global_relative_factor: float = 2.0,
        strong_score_threshold: float = 0.20,
        strong_relative_factor: float = 1.5,
        relative_threshold_cap: float = 0.35,
        local_advantage_margin: float = 0.04,
        strong_local_advantage_margin: float = 0.02,
        illumination_compensation: bool = True,
        global_lighting_lockout_threshold: float = 0.45,
    ):
        self.roi_masks = roi_masks
        self.baseline_frames = int(max(1, baseline_frames))
        self.ignore_frames = int(max(0, ignore_frames))
        self.diff_threshold = int(max(1, diff_threshold))
        self.score_threshold = float(max(0.0, score_threshold))
        self.dwell_frames = int(max(1, dwell_frames))
        self.history_frames = int(max(self.dwell_frames, history_frames))
        self.blur_kernel = tuple(int(max(1, k)) for k in blur_kernel)
        self.global_suppress_threshold = float(max(0.0, global_suppress_threshold))
        self.global_relative_factor = float(max(0.0, global_relative_factor))
        self.strong_score_threshold = float(max(0.0, strong_score_threshold))
        self.strong_relative_factor = float(max(0.0, strong_relative_factor))
        self.relative_threshold_cap = float(max(0.0, relative_threshold_cap))
        self.local_advantage_margin = float(max(0.0, local_advantage_margin))
        self.strong_local_advantage_margin = float(max(0.0, strong_local_advantage_margin))
        self.illumination_compensation = bool(illumination_compensation)
        self.global_lighting_lockout_threshold = float(max(0.0, global_lighting_lockout_threshold))
        self._frames_seen = 0
        self._baseline_seen = 0
        self._baseline_stack: list[np.ndarray] = []
        self._background: Optional[np.ndarray] = None
        self._history: Dict[str, deque] = {rid: deque(maxlen=self.history_frames) for rid in roi_masks}
        self._roi_area: Dict[str, int] = {
            rid: int(np.count_nonzero(mask)) for rid, mask in roi_masks.items()
        }
        self.last_global_motion_score: float = 0.0
        self.last_illumination_scale: float = 1.0
        self.last_illumination_offset: float = 0.0

    @property
    def ready(self) -> bool:
        return self._background is not None

    def reset(self) -> None:
        self._frames_seen = 0
        self._baseline_seen = 0
        self._baseline_stack = []
        self._background = None
        self._history = {rid: deque(maxlen=self.history_frames) for rid in self.roi_masks}

    def update(self, frame_bgr: np.ndarray) -> Optional[Dict[str, ROIMotionResult]]:
        if frame_bgr is None:
            return None
        self._frames_seen += 1
        if self._frames_seen <= self.ignore_frames:
            if self._frames_seen == self.ignore_frames:
                self._baseline_stack = []
                self._background = None
                self._baseline_seen = 0
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if not self.ready:
            self._accumulate_baseline(gray)
            if not self.ready:
                return None

        if self.illumination_compensation:
            gray_f = gray.astype(np.float32)
            bg_f = self._background.astype(np.float32)
            frame_mean = float(np.mean(gray_f))
            bg_mean = float(np.mean(bg_f))
            frame_std = float(np.std(gray_f))
            bg_std = float(np.std(bg_f))
            scale = bg_std / max(frame_std, 1e-6)
            scale = float(np.clip(scale, 0.50, 2.00))
            offset = bg_mean - (frame_mean * scale)
            corrected = np.clip((gray_f * scale) + offset, 0.0, 255.0)
            diff = np.abs(corrected - bg_f).astype(np.uint8)
            self.last_illumination_scale = scale
            self.last_illumination_offset = offset
        else:
            diff = cv2.absdiff(gray, self._background)
            self.last_illumination_scale = 1.0
            self.last_illumination_offset = 0.0
        blur = cv2.GaussianBlur(diff, self.blur_kernel, 0)
        _, motion_mask = cv2.threshold(blur, self.diff_threshold, 1, cv2.THRESH_BINARY)
        global_motion_score = float(np.sum(motion_mask)) / float(motion_mask.size)
        self.last_global_motion_score = global_motion_score

        out: Dict[str, ROIMotionResult] = {}
        for rid, mask in self.roi_masks.items():
            area = self._roi_area.get(rid, 0)
            if area <= 0:
                continue
            roi_motion = motion_mask[mask > 0]
            motion_score = float(np.sum(roi_motion)) / float(area)
            hist = self._history[rid]
            hist.append(motion_score)
            if len(hist) >= self.dwell_frames:
                recent = list(hist)[-self.dwell_frames :]
                mean_recent = float(np.mean(recent)) if recent else 0.0
            else:
                mean_recent = float(np.mean(hist)) if hist else 0.0
            rel_threshold = self.score_threshold
            if self.global_relative_factor > 0:
                rel_threshold = max(rel_threshold, global_motion_score * self.global_relative_factor)
            if self.relative_threshold_cap > 0:
                rel_threshold = min(rel_threshold, self.relative_threshold_cap)

            strong_rel_threshold = self.score_threshold
            if self.strong_relative_factor > 0:
                strong_rel_threshold = max(strong_rel_threshold, global_motion_score * self.strong_relative_factor)
            if self.relative_threshold_cap > 0:
                strong_rel_threshold = min(strong_rel_threshold, self.relative_threshold_cap)

            has_local_advantage = mean_recent >= (global_motion_score + self.local_advantage_margin)
            has_strong_local_advantage = mean_recent >= (
                global_motion_score + self.strong_local_advantage_margin
            )
            has_strong_local_evidence = (
                len(hist) >= self.dwell_frames
                and mean_recent >= self.strong_score_threshold
                and (mean_recent >= strong_rel_threshold or has_strong_local_advantage)
            )
            if (
                self.global_lighting_lockout_threshold > 0
                and global_motion_score >= self.global_lighting_lockout_threshold
                and not has_local_advantage
                and not has_strong_local_advantage
            ):
                active = False
            elif self.global_suppress_threshold > 0 and global_motion_score >= self.global_suppress_threshold:
                if mean_recent < rel_threshold and not has_local_advantage:
                    active = bool(has_strong_local_evidence)
                else:
                    active = bool(len(hist) >= self.dwell_frames)
            else:
                active = bool(mean_recent >= rel_threshold and len(hist) >= self.dwell_frames)
            out[rid] = ROIMotionResult(
                region_id=rid,
                motion_score=motion_score,
                active=active,
                mean_history=mean_recent,
            )
        return out

    def _accumulate_baseline(self, gray: np.ndarray) -> None:
        if self._baseline_seen >= self.baseline_frames:
            self._finalize_baseline()
            return
        self._baseline_stack.append(gray)
        self._baseline_seen += 1
        if self._baseline_seen >= self.baseline_frames:
            self._finalize_baseline()

    def _finalize_baseline(self) -> None:
        if self._background is not None or not self._baseline_stack:
            return
        stack = np.stack(self._baseline_stack, axis=0)
        self._background = np.median(stack, axis=0).astype(np.uint8)
        self._baseline_stack = []
