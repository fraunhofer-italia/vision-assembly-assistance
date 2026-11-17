from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from assembly_assistant.detection_types.label_taxonomy import canonical_label


def _as_label_list(value, *, default=None) -> list[str]:
    if value is None:
        value = default or []
    if isinstance(value, str):
        value = [value]
    return [canonical_label(v) for v in value if str(v).strip()]


def _as_inventory_map(value) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for raw_key, raw_value in value.items():
        try:
            out[canonical_label(raw_key)] = int(raw_value)
        except Exception:
            continue
    return out


def _as_expected_pose_list(value) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [
            {
                "part_label": raw_label,
                "expected_pose": raw_pose,
            }
            for raw_label, raw_pose in value.items()
        ]
    if not isinstance(value, list):
        return []

    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        part_label = canonical_label(item.get("part_label", item.get("part", "")))
        expected_pose = str(item.get("expected_pose", item.get("pose", ""))).strip()
        if not part_label or part_label == "NONE" or not expected_pose:
            continue
        temporal_window = str(item.get("temporal_window", item.get("window", "late"))).strip().lower()
        if temporal_window not in {"early", "middle", "late", "all"}:
            temporal_window = "late"
        try:
            min_confidence = float(item.get("min_confidence", 0.50))
        except Exception:
            min_confidence = 0.50
        try:
            min_match_ratio = float(item.get("min_match_ratio", 0.50))
        except Exception:
            min_match_ratio = 0.50
        out.append(
            {
                "part_label": part_label,
                "expected_pose": expected_pose,
                "temporal_window": temporal_window,
                "min_confidence": max(0.0, min(1.0, min_confidence)),
                "min_match_ratio": max(0.0, min(1.0, min_match_ratio)),
            }
        )
    return out


def _normalize_stage_descriptor(stage: dict) -> dict:
    s = dict(stage or {})
    s["expected_parts"] = _as_label_list(s.get("expected_parts"), default=[])
    s["expected_tools"] = _as_label_list(s.get("expected_tools"), default=["NONE"])
    s["expected_inventory"] = _as_inventory_map(s.get("expected_inventory"))
    s["required_part_counts"] = _as_inventory_map(s.get("required_part_counts"))
    s["required_tool_counts"] = _as_inventory_map(s.get("required_tool_counts"))
    s["active_part_candidates"] = _as_label_list(s.get("active_part_candidates"), default=s["expected_parts"])
    s["active_tool_candidates"] = _as_label_list(s.get("active_tool_candidates"), default=s["expected_tools"])
    s["forbidden_parts"] = _as_label_list(s.get("forbidden_parts"), default=[])
    s["forbidden_tools"] = _as_label_list(s.get("forbidden_tools"), default=[])
    s["expected_poses"] = _as_expected_pose_list(s.get("expected_poses"))
    default_tool_gate = any(label != "NONE" for label in s["active_tool_candidates"])
    s["enforce_active_tool_match"] = bool(s.get("enforce_active_tool_match", default_tool_gate))
    s["enforce_active_part_match"] = bool(s.get("enforce_active_part_match", False))
    s["enforce_expected_pose_match"] = bool(
        s.get("enforce_expected_pose_match", bool(s["expected_poses"]))
    )
    return s


def load_stage_descriptors(
    *,
    config_path: str | Path,
    operation_mode: str,
    stage_names: List[str],
) -> List[dict]:
    p = Path(config_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    modes = data.get("modes") or {}
    mode_block = modes.get(str(operation_mode).lower())
    if not isinstance(mode_block, dict):
        raise ValueError(f"Stage descriptor config missing mode block for '{operation_mode}'.")
    raw_stages = mode_block.get("stages") or []
    if not isinstance(raw_stages, list):
        raise ValueError(f"Stage descriptor config mode '{operation_mode}' field 'stages' must be a list.")

    by_name: Dict[str, dict] = {}
    for s in raw_stages:
        name = str(s.get("stage_name", "")).strip()
        if not name:
            raise ValueError(f"Mode '{operation_mode}' has a stage descriptor with missing 'stage_name'.")
        by_name[name] = dict(s)

    out = []
    missing = []
    for n in stage_names:
        if n not in by_name:
            missing.append(n)
        else:
            out.append(_normalize_stage_descriptor(by_name[n]))
    if missing:
        raise ValueError(
            f"Stage descriptor config mode '{operation_mode}' missing descriptors for stages: {missing}"
        )
    return out
