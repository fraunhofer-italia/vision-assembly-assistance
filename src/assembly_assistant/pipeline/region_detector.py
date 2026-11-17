from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from assembly_assistant.pipeline.region_registry import RegionRegistry
from assembly_assistant.trajectory.frame_processor import FramePreprocessor


@dataclass
class RegionDetection:
    region_id: str
    source: str  # "detected" | "fallback"
    confidence: float
    coordinate_frame: str
    polygon_px: Optional[List[List[float]]]
    polygon_norm: Optional[List[List[float]]]
    detector_debug: Dict[str, float]


class RegionDetector:
    """
    Classical-CV colored-region detector.

    - Uses HSV masking from registry detector configs.
    - Applies contour + geometric filtering.
    - Produces both pixel polygons and homography-normalized polygons.
    - Falls back to manual polygons from registry when confidence is low.
    """

    def __init__(
        self,
        region_registry: RegionRegistry,
        preprocessor: Optional[FramePreprocessor] = None,
        *,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.35,
        min_fill_ratio: float = 0.35,
        min_confidence: float = 0.55,
        kernel_size: int = 5,
    ):
        self.registry = region_registry
        self.pre = preprocessor
        self.min_area_ratio = float(max(0.0, min_area_ratio))
        self.max_area_ratio = float(max(self.min_area_ratio, max_area_ratio))
        self.min_fill_ratio = float(max(0.0, min_fill_ratio))
        self.min_confidence = float(max(0.0, min_confidence))
        self.kernel_size = int(max(1, kernel_size))

    def detect_regions(self, frame_bgr: np.ndarray) -> Dict[str, RegionDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("detect_regions requires non-empty BGR frame")

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        out: Dict[str, RegionDetection] = {}

        for region_id, region in self.registry.regions.items():
            detector = region.detector or {}
            hsv_range = detector.get("hsv_range")
            if not self._valid_hsv_range(hsv_range):
                out[region_id] = self._fallback_detection(region_id, region, reason="missing_hsv_range")
                continue

            mask = self._make_mask(hsv, hsv_range)
            contour, metrics = self._best_contour(mask=mask, frame_area=frame_area)
            if contour is None:
                out[region_id] = self._fallback_detection(
                    region_id,
                    region,
                    reason="no_valid_contour",
                    detector_debug=metrics,
                )
                continue

            polygon_px = self._contour_polygon(contour)
            polygon_norm = self._to_norm_polygon(polygon_px)
            confidence = self._confidence(metrics)
            if confidence < self.min_confidence:
                out[region_id] = self._fallback_detection(
                    region_id,
                    region,
                    reason="low_confidence",
                    detector_debug={**metrics, "confidence": confidence},
                )
                continue

            out[region_id] = RegionDetection(
                region_id=region_id,
                source="detected",
                confidence=confidence,
                coordinate_frame="image_px",
                polygon_px=polygon_px,
                polygon_norm=polygon_norm,
                detector_debug={**metrics, "confidence": confidence},
            )
        return out

    def _make_mask(self, hsv: np.ndarray, hsv_range: list) -> np.ndarray:
        lower = np.array(hsv_range[0], dtype=np.uint8)
        upper = np.array(hsv_range[1], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _best_contour(self, *, mask: np.ndarray, frame_area: float) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        best_metrics: Dict[str, float] = {"contours_found": float(len(contours))}

        for c in contours:
            area = float(cv2.contourArea(c))
            if area <= 1.0:
                continue
            area_ratio = area / max(1.0, frame_area)
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            bbox_area = float(max(1, bw) * max(1, bh))
            fill_ratio = area / bbox_area
            if fill_ratio < self.min_fill_ratio:
                continue
            perimeter = float(max(1e-6, cv2.arcLength(c, True)))
            compactness = float(4.0 * np.pi * area / (perimeter * perimeter))

            # Prefer larger, box-like, compact blobs.
            score = (
                min(1.0, area_ratio / max(self.min_area_ratio, 1e-6)) * 0.45
                + min(1.0, fill_ratio) * 0.35
                + min(1.0, compactness) * 0.20
            )
            if score > best_score:
                best = c
                best_score = score
                best_metrics = {
                    "contours_found": float(len(contours)),
                    "area": area,
                    "area_ratio": area_ratio,
                    "fill_ratio": fill_ratio,
                    "compactness": compactness,
                    "score_raw": score,
                }
        return best, best_metrics

    @staticmethod
    def _contour_polygon(contour: np.ndarray) -> List[List[float]]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return [[float(p[0]), float(p[1])] for p in box]

    def _to_norm_polygon(self, polygon_px: List[List[float]]) -> Optional[List[List[float]]]:
        if self.pre is None:
            return None
        out = []
        for p in polygon_px:
            wp = self.pre.warp_point((float(p[0]), float(p[1])))
            out.append([float(wp[0]), float(wp[1])])
        return out

    @staticmethod
    def _valid_hsv_range(hsv_range) -> bool:
        if not isinstance(hsv_range, list) or len(hsv_range) != 2:
            return False
        for arr in hsv_range:
            if not isinstance(arr, list) or len(arr) != 3:
                return False
        return True

    def _confidence(self, metrics: Dict[str, float]) -> float:
        area_ratio = float(metrics.get("area_ratio", 0.0))
        fill_ratio = float(metrics.get("fill_ratio", 0.0))
        compactness = float(metrics.get("compactness", 0.0))
        c = 0.45 * min(1.0, area_ratio / max(self.min_area_ratio * 6.0, 1e-6))
        c += 0.35 * min(1.0, fill_ratio)
        c += 0.20 * min(1.0, compactness)
        return float(max(0.0, min(1.0, c)))

    def _fallback_detection(
        self,
        region_id: str,
        region,
        *,
        reason: str,
        detector_debug: Optional[Dict[str, float]] = None,
    ) -> RegionDetection:
        detector_debug = detector_debug or {}
        detector_debug = {**detector_debug, "fallback_reason": 1.0, "fallback_reason_text": 0.0}
        polygon_px = None
        polygon_norm = None
        if region.coordinate_frame == "image_px":
            polygon_px = [[float(p[0]), float(p[1])] for p in region.polygon]
            polygon_norm = self._to_norm_polygon(polygon_px)
        elif region.coordinate_frame == "workspace_norm":
            polygon_norm = [[float(p[0]), float(p[1])] for p in region.polygon]
        return RegionDetection(
            region_id=region_id,
            source="fallback",
            confidence=0.0,
            coordinate_frame=region.coordinate_frame,
            polygon_px=polygon_px,
            polygon_norm=polygon_norm,
            detector_debug={**detector_debug, "fallback_reason_code": float(hash(reason) % 1000)},
        )

    @staticmethod
    def stability_score(
        detections_by_frame: List[Dict[str, RegionDetection]],
        *,
        region_ids: Optional[List[str]] = None,
    ) -> float:
        """
        Percentage of frame-region pairs where detector source is 'detected'
        (vs fallback). Useful for quick lighting robustness checks.
        """
        if not detections_by_frame:
            return 0.0
        if region_ids is None:
            region_ids = list(detections_by_frame[0].keys())
        total = 0
        detected = 0
        for frame_det in detections_by_frame:
            for rid in region_ids:
                if rid not in frame_det:
                    continue
                total += 1
                if frame_det[rid].source == "detected":
                    detected += 1
        if total == 0:
            return 0.0
        return float(detected / total)
