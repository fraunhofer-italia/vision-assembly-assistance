from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from configs.experiment_configuration import (
    STAGE_TYPE_DEFAULT,
    STAGE_TYPES_ALLOWED,
    UNORDERED_GROUP_LOCKOUT_DEFAULT,
    UNORDERED_GROUP_REQUIRED_COUNT_DEFAULT,
)
from assembly_assistant.pipeline.protocol import validate_identifier, validate_stage_names


def csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def stage_ref_list(spec: str) -> List[Path]:
    # Multiple references for one stage are separated by '|'
    return [Path(p.strip()) for p in spec.split("|") if p.strip()]


def _csv_bool_list(value: str) -> List[bool]:
    out = []
    for token in csv_list(value):
        v = token.lower()
        if v in ("1", "true", "yes", "y", "on"):
            out.append(True)
        elif v in ("0", "false", "no", "n", "off"):
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token '{token}'. Use true/false.")
    return out


def _parse_subtask_group_spec(spec: str, n_refs: int) -> Dict[str, List[int]]:
    # Example spec: "0+1|2+3|4"
    groups = [g.strip() for g in str(spec).split("|") if g.strip()]
    out: Dict[str, List[int]] = {}
    covered = set()
    for gi, group in enumerate(groups):
        idxs = [int(x.strip()) for x in group.split("+") if x.strip()]
        if not idxs:
            raise ValueError(f"Invalid subtask group '{group}' (empty indices).")
        for idx in idxs:
            if idx < 0 or idx >= n_refs:
                raise ValueError(f"Subtask reference index {idx} out of range [0,{n_refs - 1}].")
            if idx in covered:
                raise ValueError(f"Subtask reference index {idx} appears in multiple groups.")
            covered.add(idx)
        out[f"S{gi + 1}"] = idxs
    missing = [i for i in range(n_refs) if i not in covered]
    if missing:
        raise ValueError(f"Subtask groups missing reference indices: {missing}")
    return out


def _resolve_default_calibration_root() -> Path:
    candidates = [
        Path("src/assembly_assistant/calibration/calibration_data"),
        Path("calibration/calibration_data"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Prefer new src-layout path if neither exists yet.
    return candidates[0]


@dataclass
class ResolvedRuntimeConfig:
    stage_names: List[str]
    expected_tools: List[str]
    stage_ref_videos: Dict[str, List[Path]]
    stage_ref_depths: Dict[str, List[Path]] | None
    all_ref_videos: List[Path]
    obs_videos: List[Path]
    obs_full_video: Optional[Path]
    obs_full_depth: Optional[Path]
    calib_path: Path
    stage_config: List[dict]
    run_options: dict


def resolve_runtime_config(args, score_threshold: float, resolve_latest_calibration_dir) -> ResolvedRuntimeConfig:
    stage_names = validate_stage_names(csv_list(args.stage_names))
    experiment_name = validate_identifier(args.experiment_name, "experiment_name")
    user_id = validate_identifier(args.user_id, "user_id")
    session_id = validate_identifier(args.session_id, "session_id")
    raw_operator_id = getattr(args, "operator_id", None)
    operator_id = validate_identifier(raw_operator_id if raw_operator_id else user_id, "operator_id")
    trial_id = validate_identifier(getattr(args, "trial_id", "trial_001"), "trial_id")
    ref_specs = csv_list(args.reference_videos)
    ref_depth_specs = csv_list(args.reference_depths) if getattr(args, "reference_depths", None) else []
    obs_videos = [Path(p) for p in csv_list(args.observed_videos)] if args.observed_videos else []
    obs_full_video = Path(args.observed_full_video) if args.observed_full_video else None
    depth_recording = getattr(args, "depth_recording", None)
    observed_depth_arg = getattr(args, "observed_depth_npz", None) or depth_recording
    obs_full_depth = Path(observed_depth_arg) if observed_depth_arg else None
    expected_tools = csv_list(args.expected_tools)
    stage_thresholds = csv_list(args.stage_thresholds) if getattr(args, "stage_thresholds", None) else []
    subtask_groups_specs = csv_list(args.subtask_groups) if getattr(args, "subtask_groups", None) else []
    margin_mins = csv_list(args.margin_mins) if getattr(args, "margin_mins", None) else []
    stage_types = (
        csv_list(getattr(args, "stage_types"))
        if getattr(args, "stage_types", None)
        else [STAGE_TYPE_DEFAULT] * len(stage_names)
    )
    required_counts = (
        [int(v) for v in csv_list(getattr(args, "required_counts"))]
        if getattr(args, "required_counts", None)
        else [UNORDERED_GROUP_REQUIRED_COUNT_DEFAULT] * len(stage_names)
    )
    lockout_enabled = (
        _csv_bool_list(getattr(args, "lockout_enabled"))
        if getattr(args, "lockout_enabled", None)
        else [UNORDERED_GROUP_LOCKOUT_DEFAULT] * len(stage_names)
    )

    if len(stage_names) != len(ref_specs):
        raise ValueError("stage-names and reference-videos must have same length.")
    if ref_depth_specs and len(ref_depth_specs) != len(stage_names):
        raise ValueError("reference-depths must match stage count when provided.")
    if obs_full_video is None and len(stage_names) != len(obs_videos):
        raise ValueError("When using --observed-videos, it must match number of stages.")
    if len(expected_tools) == 1 and len(stage_names) > 1:
        expected_tools = expected_tools * len(stage_names)
    if len(expected_tools) != len(stage_names):
        raise ValueError("expected-tools must be length 1 or same as number of stages.")
    if len(stage_thresholds) == 1 and len(stage_names) > 1:
        stage_thresholds = stage_thresholds * len(stage_names)
    if stage_thresholds and len(stage_thresholds) != len(stage_names):
        raise ValueError("stage-thresholds must be length 1 or same as number of stages.")
    if len(subtask_groups_specs) == 1 and len(stage_names) > 1:
        subtask_groups_specs = subtask_groups_specs * len(stage_names)
    if subtask_groups_specs and len(subtask_groups_specs) != len(stage_names):
        raise ValueError("subtask-groups must be length 1 or same as number of stages.")
    if len(margin_mins) == 1 and len(stage_names) > 1:
        margin_mins = margin_mins * len(stage_names)
    if margin_mins and len(margin_mins) != len(stage_names):
        raise ValueError("margin-mins must be length 1 or same as number of stages.")
    if len(stage_types) == 1 and len(stage_names) > 1:
        stage_types = stage_types * len(stage_names)
    if len(required_counts) == 1 and len(stage_names) > 1:
        required_counts = required_counts * len(stage_names)
    if len(lockout_enabled) == 1 and len(stage_names) > 1:
        lockout_enabled = lockout_enabled * len(stage_names)
    if len(stage_types) != len(stage_names):
        raise ValueError("stage-types must be length 1 or same as number of stages.")
    if len(required_counts) != len(stage_names):
        raise ValueError("required-counts must be length 1 or same as number of stages.")
    if len(lockout_enabled) != len(stage_names):
        raise ValueError("lockout-enabled must be length 1 or same as number of stages.")

    if stage_thresholds:
        parsed_thresholds = [float(v) for v in stage_thresholds]
    else:
        parsed_thresholds = [float(score_threshold)] * len(stage_names)
    if margin_mins:
        parsed_margin_mins = [float(max(0.0, float(v))) for v in margin_mins]
    else:
        parsed_margin_mins = [0.0] * len(stage_names)
    normalized_stage_types = []
    for t in stage_types:
        tt = str(t).strip().lower()
        if tt not in STAGE_TYPES_ALLOWED:
            raise ValueError(f"Unsupported stage type '{t}'. Allowed: {list(STAGE_TYPES_ALLOWED)}")
        normalized_stage_types.append(tt)
    for idx, rc in enumerate(required_counts):
        if rc < 1:
            raise ValueError(f"required-counts[{idx}] must be >= 1.")

    stage_ref_videos: Dict[str, List[Path]] = {}
    stage_ref_depths: Dict[str, List[Path]] | None = None
    all_ref_videos: List[Path] = []
    for stage_name, spec in zip(stage_names, ref_specs):
        refs_for_stage = stage_ref_list(spec)
        if not refs_for_stage:
            raise ValueError(f"No reference video provided for stage '{stage_name}'.")
        stage_ref_videos[stage_name] = refs_for_stage
        all_ref_videos.extend(refs_for_stage)

    if ref_depth_specs:
        stage_ref_depths = {}
        for stage_name, spec in zip(stage_names, ref_depth_specs):
            depths_for_stage = stage_ref_list(spec)
            if not depths_for_stage:
                raise ValueError(f"No reference depth provided for stage '{stage_name}'.")
            if len(depths_for_stage) != len(stage_ref_videos[stage_name]):
                raise ValueError(
                    f"Reference depths for stage '{stage_name}' must match number of reference videos "
                    f"({len(stage_ref_videos[stage_name])})."
                )
            stage_ref_depths[stage_name] = depths_for_stage

    for p in all_ref_videos + obs_videos:
        if not p.exists():
            raise FileNotFoundError(f"Missing video file: {p}")
    if obs_full_video is not None and not obs_full_video.exists():
        raise FileNotFoundError(f"Missing video file: {obs_full_video}")
    if obs_full_depth is not None and obs_full_video is None:
        raise ValueError("--observed-depth-npz/--depth-recording requires --observed-full-video.")
    if stage_ref_depths:
        for depths in stage_ref_depths.values():
            for p in depths:
                if not p.exists():
                    raise FileNotFoundError(f"Missing depth file: {p}")
    if obs_full_depth is not None and not obs_full_depth.exists():
        raise FileNotFoundError(f"Missing observed depth file: {obs_full_depth}")

    if args.calibration:
        calib_path = Path(args.calibration)
    else:
        calib_path = resolve_latest_calibration_dir(_resolve_default_calibration_root())

    stage_config = []
    for i, (n, t, th, stype, req_count, lockout, margin_min) in enumerate(zip(
        stage_names,
        expected_tools,
        parsed_thresholds,
        normalized_stage_types,
        required_counts,
        lockout_enabled,
        parsed_margin_mins,
    )):
        n_refs = len(stage_ref_videos[n])
        if subtask_groups_specs:
            subtask_groups = _parse_subtask_group_spec(subtask_groups_specs[i], n_refs=n_refs)
        else:
            subtask_groups = {f"S{j + 1}": [j] for j in range(n_refs)}
        if stype == "unordered_group" and req_count > len(subtask_groups):
            raise ValueError(
                f"Stage '{n}' required_count={req_count} exceeds number of subtasks={len(subtask_groups)}."
            )
        stage_config.append(
            {
                "name": n,
                "expected_tools": [t],
                "score_threshold": th,
                "margin_min": float(margin_min),
                "stage_type": stype,
                "required_count": int(req_count),
                "lockout_enabled": bool(lockout),
                "subtask_groups": subtask_groups,
                # Schema hook: subtask_refs are stage-scoped reference sets.
                # For ordered stages this is informational. For unordered_group
                # FSM logic can consume it as candidate subtasks.
                "subtask_refs": [str(p) for p in stage_ref_videos[n]],
            }
        )

    return ResolvedRuntimeConfig(
        stage_names=stage_names,
        expected_tools=expected_tools,
        stage_ref_videos=stage_ref_videos,
        stage_ref_depths=stage_ref_depths,
        all_ref_videos=all_ref_videos,
        obs_videos=obs_videos,
        obs_full_video=obs_full_video,
        obs_full_depth=obs_full_depth,
        calib_path=calib_path,
        stage_config=stage_config,
        run_options={
            "mode": args.mode,
            "operation_mode": getattr(args, "operation_mode", "dismantle"),
            "anchor_mode": args.anchor_mode,
            "dominant_hand": args.dominant_hand,
            "active_hand_policy": getattr(args, "active_hand_policy", "dominant"),
            "regions_config": getattr(args, "regions_config", None),
            "use_region_detector": bool(getattr(args, "use_region_detector", False)),
            "motion_fsm_version": getattr(args, "motion_fsm_version", "v2"),
            "stage_descriptors_config": getattr(args, "stage_descriptors_config", None),
            "show": args.show,
            "delay_ms": args.delay_ms,
            "no_hand_frames_to_split": args.no_hand_frames_to_split,
            "rearm_no_hand_frames": args.rearm_no_hand_frames,
            "min_segment_points": args.min_segment_points,
            "experiment_name": experiment_name,
            "user_id": user_id,
            "session_id": session_id,
            "operator_id": operator_id,
            "trial_id": trial_id,
            "calibration": args.calibration,
            "stage_thresholds": parsed_thresholds,
            "margin_mins": parsed_margin_mins,
            "stage_types": normalized_stage_types,
            "required_counts": required_counts,
            "lockout_enabled": lockout_enabled,
            "reference_depths": ref_depth_specs,
            "observed_depth_npz": str(obs_full_depth) if obs_full_depth is not None else None,
        },
    )
