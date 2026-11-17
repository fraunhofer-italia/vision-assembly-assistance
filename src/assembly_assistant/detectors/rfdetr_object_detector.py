from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from assembly_assistant.detection_types.detector_outputs import GenericDetection, detections_to_dict
from assembly_assistant.detection_types.label_taxonomy import canonical_label


class RFDETRDependencyError(RuntimeError):
    pass


def _infer_coco_category_names(dataset_dir: Path, split: str, ann_filename: str) -> dict[int, str]:
    ann_path = dataset_dir / split / ann_filename
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")
    payload = json.loads(ann_path.read_text())
    categories = payload.get("categories", []) or []
    annotations = payload.get("annotations", []) or []
    if not categories or not annotations:
        raise ValueError(f"Invalid COCO annotations (empty categories/annotations): {ann_path}")
    used_category_ids = {int(a.get('category_id')) for a in annotations if a.get("category_id") is not None}
    id_to_name = {int(c.get("id")): str(c.get("name")) for c in categories if c.get("id") is not None and c.get("name")}
    ordered_ids = sorted(cid for cid in used_category_ids if cid in id_to_name)
    return {cid: canonical_label(id_to_name[cid]) for cid in ordered_ids}


class RFDETRObjectDetector:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        dataset_dir: str | Path = "data/detector_dataset_v1/Assembly_parts_detection.v2i.coco",
        class_split: str = "train",
        ann_filename: str = "_annotations.coco.json",
        model_size: str = "small",
        device: str = "cpu",
        threshold: float = 0.25,
        optimize: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {self.checkpoint_path}")
        self.dataset_dir = Path(dataset_dir)
        self.threshold = float(threshold)
        self.category_names = _infer_coco_category_names(self.dataset_dir, class_split, ann_filename)
        self._model = self._build_model(model_size, str(device), bool(optimize))

    def _build_model(self, model_size: str, device: str, optimize: bool):
        try:
            from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
        except Exception as exc:
            raise RFDETRDependencyError(
                "RF-DETR runtime is not installed. Activate the RF-DETR environment before using this detector."
            ) from exc

        cls_map = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        if model_size not in cls_map:
            raise ValueError(f"Unsupported RF-DETR model size: {model_size}")
        model_device = "cpu" if str(device).lower() == "cpu" else "cuda"
        model = cls_map[model_size](
            pretrain_weights=str(self.checkpoint_path),
            device=model_device,
            num_classes=len(self.category_names),
        )
        if optimize:
            try:
                model.optimize_for_inference()
            except Exception:
                pass
        return model

    def _resolve_label(self, cls_id: int) -> str:
        if cls_id in self.category_names:
            return self.category_names[cls_id]
        if (cls_id + 1) in self.category_names:
            return self.category_names[cls_id + 1]
        if cls_id == 0:
            return "NONE"
        return str(cls_id)

    def detect(self, frame: np.ndarray) -> List[dict]:
        det = self._model.predict(frame, threshold=float(self.threshold))
        xyxy = det.xyxy if hasattr(det, "xyxy") else []
        conf = det.confidence if hasattr(det, "confidence") else []
        cls = det.class_id if hasattr(det, "class_id") else []
        out: List[GenericDetection] = []
        for bbox, score, cls_id in zip(xyxy, conf, cls):
            label = self._resolve_label(int(cls_id))
            out.append(
                GenericDetection(
                    label=canonical_label(label),
                    bbox=tuple(float(v) for v in bbox),
                    confidence=float(score),
                    source_model="rfdetr",
                    metadata={"class_id": int(cls_id)},
                )
            )
        return detections_to_dict(out)
