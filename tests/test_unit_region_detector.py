import cv2
import numpy as np

from assembly_assistant.pipeline.region_detector import RegionDetector
from assembly_assistant.pipeline.region_registry import RegionRegistry


def _hsv_to_bgr(h: int, s: int, v: int):
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _registry():
    return RegionRegistry.from_dict(
        {
            "registry_id": "rtest",
            "version": "1.0",
            "modes": ["assembly", "dismantle"],
            "regions": [
                {
                    "region_id": "box_tools_pair",
                    "coordinate_frame": "image_px",
                    "polygon_px": [[40, 50], [200, 50], [200, 170], [40, 170]],
                    "roles": ["tool_pick", "tool_drop"],
                    "detector": {"hsv_range": [[18, 80, 80], [35, 255, 255]]},
                },
                {
                    "region_id": "part_box",
                    "coordinate_frame": "image_px",
                    "polygon_px": [[240, 50], [400, 50], [400, 170], [240, 170]],
                    "roles": ["part_pick", "part_drop"],
                    "detector": {"hsv_range": [[90, 70, 70], [130, 255, 255]]},
                    "mode_semantics": {
                        "assembly": {"part_pick": True, "part_drop": False},
                        "dismantle": {"part_pick": False, "part_drop": True},
                    },
                },
            ],
            "stage_bindings": {},
        }
    )


def _base_frame():
    img = np.zeros((260, 480, 3), dtype=np.uint8)
    yellow = _hsv_to_bgr(26, 220, 220)
    blue = _hsv_to_bgr(110, 220, 220)
    cv2.rectangle(img, (40, 50), (200, 170), yellow, thickness=-1)
    cv2.rectangle(img, (240, 50), (400, 170), blue, thickness=-1)
    return img


def test_region_detector_detects_with_hsv_and_geometry_filters():
    reg = _registry()
    det = RegionDetector(reg, preprocessor=None, min_confidence=0.35)
    out = det.detect_regions(_base_frame())
    assert out["box_tools_pair"].source == "detected"
    assert out["part_box"].source == "detected"
    assert out["box_tools_pair"].polygon_px is not None
    assert out["part_box"].polygon_px is not None
    assert out["box_tools_pair"].confidence > 0.35
    assert out["part_box"].confidence > 0.35


def test_region_detector_fallback_when_color_missing():
    reg = _registry()
    det = RegionDetector(reg, preprocessor=None, min_confidence=0.35)
    frame = np.zeros((260, 480, 3), dtype=np.uint8)
    out = det.detect_regions(frame)
    assert out["box_tools_pair"].source == "fallback"
    assert out["part_box"].source == "fallback"
    assert out["box_tools_pair"].polygon_px is not None


def test_region_detector_stability_over_lighting_variations():
    reg = _registry()
    det = RegionDetector(reg, preprocessor=None, min_confidence=0.35)
    base = _base_frame().astype(np.float32)
    runs = []
    for gain in [0.85, 0.95, 1.0, 1.05, 1.15]:
        frame = np.clip(base * gain, 0, 255).astype(np.uint8)
        runs.append(det.detect_regions(frame))
    score = RegionDetector.stability_score(runs, region_ids=["box_tools_pair", "part_box"])
    assert score >= 0.95
