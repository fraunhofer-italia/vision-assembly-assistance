from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# Pragmatic identifier policy for reproducible experiment tracking.
# Keeps names shell/git/filesystem friendly.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_identifier(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")
    if not _ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name}='{value}' is invalid. Use 1-64 chars: [A-Za-z0-9_.-], "
            "starting with alphanumeric."
        )
    return value


def validate_stage_names(stage_names: Iterable[str]) -> List[str]:
    normalized = [validate_identifier(s, "stage_name") for s in stage_names]
    seen = set()
    duplicates = []
    for s in normalized:
        if s in seen and s not in duplicates:
            duplicates.append(s)
        seen.add(s)
    if duplicates:
        raise ValueError(f"Duplicate stage names are not allowed: {duplicates}")
    return normalized


def _path_manifest(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {"path": None, "exists": False, "is_file": False, "size_bytes": None}
    p = Path(path)
    exists = p.exists()
    is_file = p.is_file() if exists else False
    size_bytes = p.stat().st_size if is_file else None
    return {"path": str(p), "exists": exists, "is_file": is_file, "size_bytes": size_bytes}


def build_protocol_metadata(
    *,
    protocol_version: str,
    run_options: dict,
    stage_names: List[str],
    expected_tools: List[str],
    stage_config: Optional[List[dict]] = None,
    references: List[Path],
    observed_videos: List[Path],
    observed_full_video: Optional[Path],
    calibration_path: Path,
) -> dict:
    calibration_dir = calibration_path if calibration_path.is_dir() else calibration_path.parent
    homography_path = calibration_dir / "homography.npy"
    calib_meta_path = calibration_dir / "calibration_meta.json"

    refs_manifest = [_path_manifest(p) for p in references]
    obs_manifest = [_path_manifest(p) for p in observed_videos]
    obs_full_manifest = _path_manifest(observed_full_video) if observed_full_video is not None else None

    checklist = {
        "calibration_path_exists": calibration_path.exists(),
        "homography_exists": homography_path.exists(),
        "calibration_meta_exists": calib_meta_path.exists(),
        "references_all_exist": all(item["exists"] for item in refs_manifest) if refs_manifest else False,
        "observed_inputs_all_exist": (
            all(item["exists"] for item in obs_manifest)
            if obs_manifest
            else bool(obs_full_manifest and obs_full_manifest["exists"])
        ),
    }

    return {
        "version": protocol_version,
        "min_reader_version": "1.1",
        "schema_changes": [
            "unordered_group_stage_schema",
            "group_progress_fields",
            "remaining_subtasks_total_subtasks_fields",
            "margin_gate_fields",
            "subtask_roi_gate_fields",
        ],
        "naming_policy": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
        "run_manifest": {
            "python_executable": sys.executable,
            "mode": run_options["mode"],
            "operation_mode": run_options.get("operation_mode", "dismantle"),
            "anchor_mode": run_options["anchor_mode"],
            "dominant_hand": run_options["dominant_hand"],
            "regions_config": run_options.get("regions_config"),
            "use_region_detector": bool(run_options.get("use_region_detector", False)),
            "motion_fsm_version": run_options.get("motion_fsm_version", "v2"),
            "stage_descriptors_config": run_options.get("stage_descriptors_config"),
            "stage_names": stage_names,
            "expected_tools": expected_tools,
            "requested_stage_thresholds": run_options.get("requested_stage_thresholds", []),
            "effective_stage_thresholds": run_options.get("effective_stage_thresholds", []),
            "stage_thresholds": run_options.get("stage_thresholds", []),
            "margin_mins": run_options.get("margin_mins", []),
            "stage_types": run_options.get("stage_types", []),
            "required_counts": run_options.get("required_counts", []),
            "lockout_enabled": run_options.get("lockout_enabled", []),
            "stage_schema": stage_config or [],
            "segment_split_policy": {
                "no_hand_frames_to_split": run_options["no_hand_frames_to_split"],
                "rearm_no_hand_frames": run_options["rearm_no_hand_frames"],
                "min_segment_points": run_options["min_segment_points"],
            },
        },
        "data_sources": {
            "calibration": _path_manifest(calibration_path),
            "references": refs_manifest,
            "observed_videos": obs_manifest,
            "observed_full_video": obs_full_manifest,
        },
        "camera_calibration_checklist": checklist,
    }
