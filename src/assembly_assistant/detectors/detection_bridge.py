from __future__ import annotations

from typing import Iterable, List, Optional

from assembly_assistant.detection_types.label_taxonomy import (
    build_inventory,
    canonical_label,
    is_part_label,
    is_tool_label,
    label_category,
)

SCREWDRIVER_LABELS = {"screwdriver_large_l", "screwdriver_small_l"}
HAND_DISTANCE_WEIGHT = 0.0025
PREVIOUS_DISTANCE_WEIGHT = 0.0025
PREFERRED_LABEL_BOOST = 1.0
PREVIOUS_LABEL_BOOST = 0.75
MAX_ACTIVE_HAND_DISTANCE = 0.18
MAX_ACTIVE_HAND_DISTANCE_PX = 140.0
MAX_ACTIVE_KEYPOINT_DISTANCE = 0.12
MAX_ACTIVE_KEYPOINT_DISTANCE_PX = 90.0


def _bbox_center(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _dist(a, b) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def _within_hand_distance(bbox, wrist, max_distance: float = MAX_ACTIVE_HAND_DISTANCE) -> bool:
    center = _bbox_center(bbox)
    if wrist is None or center is None:
        return False
    return _dist(wrist, center) <= float(max_distance)


def _within_any_hand_point_distance(bbox, hand_points, max_distance: float) -> bool:
    center = _bbox_center(bbox)
    if center is None or hand_points is None:
        return False
    try:
        return min(_dist(center, pt) for pt in hand_points) <= float(max_distance)
    except Exception:
        return False


def _bbox_is_pixel_space(bbox) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    try:
        return max(abs(float(v)) for v in bbox) > 2.0
    except Exception:
        return False


def _resolve_hand_point_for_bbox(hand_block: Optional[dict], bbox):
    if not isinstance(hand_block, dict):
        return None, None, MAX_ACTIVE_HAND_DISTANCE
    if _bbox_is_pixel_space(bbox):
        keypoints_px = hand_block.get("keypoints_px")
        wrist_px = hand_block.get("wrist_px")
        if keypoints_px is not None:
            return wrist_px, keypoints_px, MAX_ACTIVE_HAND_DISTANCE_PX
        if wrist_px is not None:
            return wrist_px, None, MAX_ACTIVE_HAND_DISTANCE_PX
    keypoints = hand_block.get("keypoints")
    return hand_block.get("wrist"), keypoints, MAX_ACTIVE_HAND_DISTANCE


def _normalized_label_set(values) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {canonical_label(v) for v in values if str(v).strip()}


def _candidate_score(
    det: dict,
    *,
    wrist=None,
    preferred_labels=None,
    previous_active: Optional[dict] = None,
) -> float:
    item = normalize_detection(det)
    label = canonical_label(item.get("label", item.get("class", "NONE")))
    center = _bbox_center(item.get("bbox"))
    score = float(item.get("confidence", 0.0))

    if wrist is not None and center is not None:
        score -= HAND_DISTANCE_WEIGHT * _dist(wrist, center)

    preferred = _normalized_label_set(preferred_labels)
    if preferred and label in preferred:
        score += PREFERRED_LABEL_BOOST

    previous = normalize_detection(previous_active or {})
    prev_label = canonical_label(previous.get("label", previous.get("class", "NONE")))
    prev_center = _bbox_center(previous.get("bbox"))
    if prev_label != "NONE" and label == prev_label:
        score += PREVIOUS_LABEL_BOOST
    if prev_center is not None and center is not None:
        score -= PREVIOUS_DISTANCE_WEIGHT * _dist(prev_center, center)

    return score


def select_primary_detection(detections: Iterable[dict] | None, hand_block: Optional[dict] = None) -> dict:
    items = [dict(d) for d in (detections or []) if isinstance(d, dict)]
    if not items:
        return {"class": "NONE", "bbox": None, "confidence": 0.0}

    screwdrivers = [
        d for d in items if canonical_label(d.get("label", d.get("class", ""))) in SCREWDRIVER_LABELS
    ]
    if screwdrivers:
        best = max(screwdrivers, key=lambda d: float(d.get("confidence", 0.0)))
        return {
            "class": canonical_label(best.get("label", best.get("class", "NONE"))),
            "bbox": best.get("bbox"),
            "confidence": float(best.get("confidence", 0.0)),
        }

    wrist = None
    if isinstance(hand_block, dict) and hand_block.get("visible"):
        wrist = hand_block.get("wrist")
    if wrist is not None:
        with_centers: List[tuple[float, dict]] = []
        for det in items:
            center = _bbox_center(det.get("bbox"))
            if center is None:
                continue
            score = float(det.get("confidence", 0.0)) - 0.25 * _dist(wrist, center)
            with_centers.append((score, det))
        if with_centers:
            best = max(with_centers, key=lambda item: item[0])[1]
            return {
                "class": canonical_label(best.get("label", best.get("class", "NONE"))),
                "bbox": best.get("bbox"),
                "confidence": float(best.get("confidence", 0.0)),
            }

    best = max(items, key=lambda d: float(d.get("confidence", 0.0)))
    return {
        "class": canonical_label(best.get("label", best.get("class", "NONE"))),
        "bbox": best.get("bbox"),
        "confidence": float(best.get("confidence", 0.0)),
    }


def normalize_detection(det: dict) -> dict:
    item = dict(det or {})
    label = canonical_label(item.get("label", item.get("class", "NONE")))
    item["label"] = label
    item["class"] = label
    item["category"] = label_category(label)
    return item


def partition_detections(detections: Iterable[dict] | None) -> tuple[List[dict], List[dict], List[dict]]:
    objects: List[dict] = []
    tools: List[dict] = []
    parts: List[dict] = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        item = normalize_detection(det)
        objects.append(item)
        if is_tool_label(item.get("label")):
            tools.append(item)
        elif is_part_label(item.get("label")):
            parts.append(item)
    return tools, parts, objects


def filter_workspace_detections(
    detections: Iterable[dict] | None,
    workspace_filter=None,
) -> List[dict]:
    items = [dict(d) for d in (detections or []) if isinstance(d, dict)]
    if workspace_filter is None:
        return items
    out: List[dict] = []
    for det in items:
        try:
            keep = bool(workspace_filter(det))
        except Exception:
            keep = False
        if keep:
            out.append(det)
    return out


def select_active_part(
    detections: Iterable[dict] | None,
    hand_block: Optional[dict] = None,
    *,
    preferred_labels=None,
    previous_active: Optional[dict] = None,
    max_hand_distance: float = MAX_ACTIVE_HAND_DISTANCE,
) -> dict | None:
    parts = [dict(d) for d in (detections or []) if isinstance(d, dict)]
    if not parts:
        return None

    if not (isinstance(hand_block, dict) and hand_block.get("visible")):
        return None
    close_parts = []
    for det in parts:
        wrist, hand_points, resolved_distance = _resolve_hand_point_for_bbox(hand_block, det.get("bbox"))
        is_px = _bbox_is_pixel_space(det.get("bbox"))
        distance_limit = max_hand_distance if not is_px else MAX_ACTIVE_HAND_DISTANCE_PX
        keypoint_limit = MAX_ACTIVE_KEYPOINT_DISTANCE if not is_px else MAX_ACTIVE_KEYPOINT_DISTANCE_PX
        hand_close = _within_hand_distance(
            det.get("bbox"),
            wrist,
            resolved_distance if resolved_distance is not None else distance_limit,
        )
        point_close = _within_any_hand_point_distance(det.get("bbox"), hand_points, keypoint_limit)
        if hand_close or point_close:
            close_parts.append(det)
    if not close_parts:
        return None
    scored = [
        (
            _candidate_score(
                det,
                wrist=_resolve_hand_point_for_bbox(hand_block, det.get("bbox"))[0],
                preferred_labels=preferred_labels,
                previous_active=previous_active,
            ),
            det,
        )
        for det in close_parts
    ]
    if not scored:
        return None
    return normalize_detection(max(scored, key=lambda item: item[0])[1])


def select_active_tool(
    detections: Iterable[dict] | None,
    hand_block: Optional[dict] = None,
    *,
    preferred_labels=None,
    previous_active: Optional[dict] = None,
    max_hand_distance: float = MAX_ACTIVE_HAND_DISTANCE,
) -> dict | None:
    tools = [dict(d) for d in (detections or []) if isinstance(d, dict)]
    if not tools:
        return None
    if not (isinstance(hand_block, dict) and hand_block.get("visible")):
        return None
    close_tools = []
    for det in tools:
        wrist, hand_points, resolved_distance = _resolve_hand_point_for_bbox(hand_block, det.get("bbox"))
        is_px = _bbox_is_pixel_space(det.get("bbox"))
        distance_limit = max_hand_distance if not is_px else MAX_ACTIVE_HAND_DISTANCE_PX
        keypoint_limit = MAX_ACTIVE_KEYPOINT_DISTANCE if not is_px else MAX_ACTIVE_KEYPOINT_DISTANCE_PX
        hand_close = _within_hand_distance(
            det.get("bbox"),
            wrist,
            resolved_distance if resolved_distance is not None else distance_limit,
        )
        point_close = _within_any_hand_point_distance(det.get("bbox"), hand_points, keypoint_limit)
        if hand_close or point_close:
            close_tools.append(det)
    if not close_tools:
        return None
    scored = [
        (
            _candidate_score(
                det,
                wrist=_resolve_hand_point_for_bbox(hand_block, det.get("bbox"))[0],
                preferred_labels=preferred_labels,
                previous_active=previous_active,
            ),
            det,
        )
        for det in close_tools
    ]
    if not scored:
        return None
    return normalize_detection(max(scored, key=lambda item: item[0])[1])
