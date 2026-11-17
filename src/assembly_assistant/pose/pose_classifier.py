from __future__ import annotations

from pathlib import Path
from typing import Iterable

from assembly_assistant.detection_types.label_taxonomy import canonical_label
from assembly_assistant.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_pose_classifier_model_map(value: str | None) -> dict[str, Path]:
    """
    Parse a comma-separated mapping: part_label=/path/to/best.pt,other=/path.pt.
    """
    if not value:
        return {}
    out: dict[str, Path] = {}
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "--pose-classifier-models entries must be part_label=/path/to/best.pt"
            )
        raw_label, raw_path = token.split("=", 1)
        label = canonical_label(raw_label.strip())
        path = Path(raw_path.strip()).expanduser()
        if label and label != "NONE" and str(path):
            out[label] = path
    return out


class PoseClassifierManager:
    """
    Object-specific YOLO classification wrapper.

    It consumes object detector boxes, crops the requested part with padding, and
    predicts the pose class using the corresponding per-part classification model.
    """

    def __init__(
        self,
        *,
        model_map: dict[str, str | Path] | None = None,
        search_roots: Iterable[str | Path] | None = None,
        device: str | None = None,
        bbox_pad: float = 0.20,
        min_detection_confidence: float = 0.0,
    ):
        self.device = device
        self.bbox_pad = float(bbox_pad)
        self.min_detection_confidence = float(min_detection_confidence)
        self.search_roots = [Path(p) for p in (search_roots or [Path("."), Path("runs"), Path("data/classifier_dataset_v1/runs")])]
        self.model_map: dict[str, Path] = {
            canonical_label(k): Path(v).expanduser() for k, v in (model_map or {}).items()
        }
        self.models: dict[str, object] = {}
        self._yolo_cls = None

    @classmethod
    def auto(
        cls,
        *,
        model_map: dict[str, str | Path] | None = None,
        device: str | None = None,
        bbox_pad: float = 0.20,
        min_detection_confidence: float = 0.0,
    ) -> "PoseClassifierManager":
        return cls(
            model_map=model_map,
            device=device,
            bbox_pad=bbox_pad,
            min_detection_confidence=min_detection_confidence,
        )

    def available_model_path(self, part_label: str) -> Path | None:
        label = canonical_label(part_label)
        explicit = self.model_map.get(label)
        if explicit is not None and explicit.exists():
            return explicit

        patterns = [
            f"**/{label}_pose_*/weights/best.pt",
            f"**/{label}*pose*/weights/best.pt",
        ]
        for root in self.search_roots:
            if not root.exists():
                continue
            for pattern in patterns:
                matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    self.model_map[label] = matches[0]
                    return matches[0]
        return None

    def _load_model(self, part_label: str):
        label = canonical_label(part_label)
        if label in self.models:
            return self.models[label]

        model_path = self.available_model_path(label)
        if model_path is None:
            LOGGER.warning("No pose classifier weights found for part=%s", label)
            return None

        if self._yolo_cls is None:
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as exc:
                LOGGER.warning("Pose classifier disabled: ultralytics unavailable (%s)", exc)
                return None
            self._yolo_cls = YOLO

        model = self._yolo_cls(str(model_path))
        self.models[label] = model
        LOGGER.info("Loaded pose classifier part=%s weights=%s", label, model_path)
        return model

    @staticmethod
    def _bbox_xyxy(det: dict) -> list[float] | None:
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _padded_bbox(bbox: list[float], width: int, height: int, pad_ratio: float) -> list[int]:
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad = max(bw, bh) * max(0.0, float(pad_ratio))
        xi1 = int(max(0.0, x1 - pad))
        yi1 = int(max(0.0, y1 - pad))
        xi2 = int(min(float(width), x2 + pad))
        yi2 = int(min(float(height), y2 + pad))
        if xi2 <= xi1:
            xi2 = min(width, xi1 + 1)
        if yi2 <= yi1:
            yi2 = min(height, yi1 + 1)
        return [xi1, yi1, xi2, yi2]

    def _best_detection(self, detections: list[dict], part_label: str) -> dict | None:
        label = canonical_label(part_label)
        best = None
        best_conf = float("-inf")
        for det in detections or []:
            det_label = canonical_label(str(det.get("label", det.get("class", ""))))
            if det_label != label:
                continue
            try:
                conf = float(det.get("confidence", det.get("score", 0.0)))
            except Exception:
                conf = 0.0
            if conf < self.min_detection_confidence:
                continue
            if conf > best_conf:
                best_conf = conf
                best = det
        return best

    @staticmethod
    def _needed_part_labels(expected_poses: list[dict] | None) -> list[str]:
        labels = []
        for item in expected_poses or []:
            label = canonical_label(item.get("part_label", item.get("part", "")))
            if label and label != "NONE" and label not in labels:
                labels.append(label)
        return labels

    def predict(
        self,
        *,
        frame_bgr,
        detections: list[dict] | None,
        expected_poses: list[dict] | None = None,
    ) -> list[dict]:
        if frame_bgr is None or detections is None:
            return []
        part_labels = self._needed_part_labels(expected_poses)
        if not part_labels:
            return []

        height, width = frame_bgr.shape[:2]
        predictions: list[dict] = []
        for part_label in part_labels:
            det = self._best_detection(detections, part_label)
            if det is None:
                continue
            bbox = self._bbox_xyxy(det)
            if bbox is None:
                continue
            crop_box = self._padded_bbox(bbox, width, height, self.bbox_pad)
            x1, y1, x2, y2 = crop_box
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            model = self._load_model(part_label)
            if model is None:
                continue

            kwargs = {"verbose": False}
            if self.device:
                kwargs["device"] = self.device
            try:
                results = model.predict(crop, **kwargs)
            except Exception as exc:
                LOGGER.warning("Pose classifier inference failed part=%s: %s", part_label, exc)
                continue
            if not results:
                continue
            probs = getattr(results[0], "probs", None)
            if probs is None:
                continue
            top1 = int(getattr(probs, "top1", -1))
            conf = float(getattr(probs, "top1conf", 0.0))
            names = getattr(results[0], "names", None) or getattr(model, "names", {}) or {}
            class_name = str(names.get(top1, top1))
            predictions.append(
                {
                    "part_label": canonical_label(part_label),
                    "class_name": class_name,
                    "class_id": top1,
                    "confidence": conf,
                    "bbox_xyxy": [float(v) for v in bbox],
                    "bbox_xyxy_padded": [int(v) for v in crop_box],
                    "detector_confidence": float(det.get("confidence", det.get("score", 0.0))),
                    "model_path": str(self.available_model_path(part_label) or ""),
                }
            )
        return predictions
