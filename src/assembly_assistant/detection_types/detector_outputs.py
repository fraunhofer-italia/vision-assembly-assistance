from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class HandDetection:
    keypoints: List[Tuple[float, float]]
    velocity: float


@dataclass
class ToolDetection:
    tool_class: str
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float


@dataclass
class GenericDetection:
    label: str
    bbox: Optional[Tuple[float, float, float, float]]
    confidence: float
    source_model: str
    mask_path: Optional[str] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def hand_to_dict(hand: HandDetection) -> dict:
    return {
        "keypoints": hand.keypoints,
        "velocity": hand.velocity,
    }


def tool_to_dict(tool: ToolDetection) -> dict:
    return {
        "class": tool.tool_class,
        "bbox": tool.bbox,
        "confidence": tool.confidence,
    }


def detection_to_dict(det: GenericDetection) -> dict:
    return {
        "label": det.label,
        "bbox": list(det.bbox) if det.bbox is not None else None,
        "confidence": float(det.confidence),
        "source_model": det.source_model,
        "mask_path": det.mask_path,
        "polygon": [list(pt) for pt in det.polygon] if det.polygon is not None else None,
        "metadata": dict(det.metadata or {}),
    }


def detections_to_dict(detections: List[GenericDetection]) -> List[dict]:
    return [detection_to_dict(det) for det in detections]
