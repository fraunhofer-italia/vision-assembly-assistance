"""
Full video-driven pipeline entrypoint (keeps existing main.py untouched).

Examples:
python main_full_pipeline.py \
  --mode offline \
  --stage-names StageA_Single \
  --reference-videos data/references/vids/ref_stage1_good_take01.mp4 \
  --observed-videos data/observed/vids/obs_stage1_poor_spatial_take01.mp4 \
  --expected-tools NONE \
  --anchor-mode wrist

python main_full_pipeline.py \
  --mode offline \
  --stage-names StageA,StageB \
  --reference-videos data/references/vids/ref_stage1_good_take01.mp4,data/references/vids/ref_stage2_good_take01.mp4 \
  --observed-full-video data/observed/vids/obs_full_2stage_good_01.mp4 \
  --expected-tools NONE,NONE \
  --anchor-mode wrist_index_midpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CALIB_DATA_ROOT = SRC / "assembly_assistant" / "calibration" / "calibration_data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.calibration.calibration_loader import CalibrationLoader
from configs.experiment_configuration import (
    ACTIVE_HAND_POLICY_DEFAULT,
    ANCHOR_MODE_DEFAULT,
    DELAY_MS,
    COLOR_TRIGGER_ENABLED,
    COLOR_BASELINE_FRAMES,
    COLOR_DELTA_THRESHOLD,
    COLOR_DELTA_RANGE,
    FUSED_TRIGGER_THRESHOLD,
    FUSED_WEIGHT_COLOR,
    FUSED_WEIGHT_DEPTH,
    FUSED_WEIGHT_RGB,
    DEPTH_BASELINE_FRAMES,
    DEPTH_DELTA_MM,
    DEPTH_DWELL_FRAMES,
    DEPTH_FALLBACK_TO_RGB,
    DEPTH_ROI_OVERLAY_ENABLED,
    DEPTH_TRIGGER_ENABLED,
    DOMINANT_HAND,
    DROP_REGION,
    MOTION_FSM_DEBUG_LOGGING_ENABLED,
    MOTION_FSM_MAX_HAND_JUMP,
    MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS,
    MOTION_FSM_MIN_CONSECUTIVE_VISIBLE,
    MOTION_FSM_MIN_RECORDING_FRAMES,
    MOTION_FSM_VERSION,
    MOTION_FSM_WARMUP_FRAMES,
    MOTION_TRIGGER_ENABLED,
    MOTION_BASELINE_FRAMES,
    MOTION_FOCUS_IGNORE_FRAMES,
    MOTION_DIFF_THRESHOLD,
    MOTION_SCORE_THRESHOLD,
    MOTION_DWELL_FRAMES,
    MOTION_HISTORY_FRAMES,
    MOTION_BLUR_KERNEL,
    MOTION_LOG_INTERVAL_FRAMES,
    MOTION_GLOBAL_SUPPRESS_THRESHOLD,
    MOTION_GLOBAL_RELATIVE_FACTOR,
    MIN_SEGMENT_POINTS,
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    NO_HAND_FRAMES_TO_SPLIT,
    REARM_NO_HAND_FRAMES,
    STAGE_SCORE_THRESHOLD,
    STAGE_THRESHOLDS_BY_STAGE,
    STAGE_WEIGHTS,
    SYNC_MAX_TIME_DIFF,
    TOOL_HAND_DISTANCE_THRESHOLD,
    PROTOCOL_VERSION,
    WORK_REGION,
)
from assembly_assistant.fsm.stage_fsm import StageFSM
from assembly_assistant.pipeline.video_extractors import (
    extract_all_stage_segments_from_video_depthai,
    extract_all_stage_segments_from_video_offline,
    extract_stage_from_video,
)
from assembly_assistant.pipeline.config_resolver import resolve_runtime_config
from assembly_assistant.pipeline.experiment_io import (
    append_motion_fsm_debug_payload,
    build_system_config,
    log_stage_evaluation_artifacts,
    start_experiment_tracking,
)
from assembly_assistant.pipeline.region_detector import RegionDetector
from assembly_assistant.pipeline.region_registry import RegionRegistry
from assembly_assistant.pipeline.reference_build import build_stage_references_from_videos
from assembly_assistant.pipeline.stage_descriptor_loader import load_stage_descriptors
from assembly_assistant.pipeline.protocol import build_protocol_metadata
from assembly_assistant.pipeline.virtual_projection import VirtualProjectionGuide
from assembly_assistant.pose.bridge import PosePilotBridge, PosePilotExporter
from assembly_assistant.pose.pose_classifier import PoseClassifierManager, parse_pose_classifier_model_map
from assembly_assistant.trajectory.frame_processor import FramePreprocessor
from assembly_assistant.utils.logger import get_logger
from assembly_assistant.utils.metadata_validator import validate_metadata_file


logger = get_logger(__name__)


def _log_segment_summary(summary_rows: list[dict], *, header: str) -> None:
    if not summary_rows:
        logger.info("%s: no segments recorded.", header)
        return
    logger.info("%s:", header)
    for row in summary_rows:
        logger.info(
            "  stage=%s start_frame=%s end=%s duration_s=%s closure=%s",
            row.get("stage_name", "UNKNOWN"),
            row.get("start_frame"),
            ("EOF" if row.get("end_frame") is None else row.get("end_frame")),
            ("-" if row.get("duration_s") is None else f"{float(row['duration_s']):.2f}"),
            row.get("closure_type", "unknown"),
        )


def _resolve_latest_calibration_dir(calib_root: Path) -> Path:
    if not calib_root.exists():
        raise FileNotFoundError(
            f"Calibration directory not found: {calib_root}. "
            "Run `python scripts/run_calibrate.py` first."
        )
    candidates = [p for p in calib_root.iterdir() if p.is_dir() and p.name.startswith("calibration_")]
    if not candidates:
        raise FileNotFoundError(
            f"No calibration folders found in {calib_root}. Run `python scripts/run_calibrate.py` first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _per_stage_run_should_continue(feedback: dict) -> bool:
    """Per-stage mode guard: unordered stages may pass an attempt without completion."""
    if not bool(feedback.get("stage_passed", False)):
        return False
    if str(feedback.get("stage_type", "ordered")).lower() != "unordered_group":
        return True
    return bool(
        feedback.get("stage_complete", False)
        or (feedback.get("group_progress") or {}).get("group_complete", False)
    )


def _build_descriptor_unordered_groups(stage_descriptors):
    by_stage = {}
    groups = {}
    for idx, desc in enumerate(stage_descriptors or []):
        name = str((desc or {}).get("stage_name", "")).strip()
        if not name:
            continue
        by_stage[name] = {
            "index": idx,
            "group_id": str((desc or {}).get("unordered_group_id", "")).strip() or None,
        }
        group_id = str((desc or {}).get("unordered_group_id", "")).strip()
        if not group_id:
            continue
        members = [str(x).strip() for x in ((desc or {}).get("unordered_group_members") or []) if str(x).strip()]
        if not members:
            members = [name]
        groups[group_id] = {
            "members": [m for m in members if m in {str((d or {}).get('stage_name', '')).strip() for d in (stage_descriptors or [])}],
        }
    return by_stage, groups


def _sync_stage_fsm_to_segment_stage(
    *,
    stage_fsm,
    segment_stage_name: str | None,
    descriptor_stage_info: dict,
):
    if not segment_stage_name:
        return None
    info = descriptor_stage_info.get(str(segment_stage_name))
    if not info:
        return None
    stage_fsm.current_stage_index = int(info["index"])
    return info


def _postprocess_unordered_segment_progress(
    *,
    stage_fsm,
    feedback: dict,
    evaluated_stage_name: str,
    descriptor_stage_info: dict,
    descriptor_unordered_groups: dict,
    completed_unordered_members: set,
):
    info = descriptor_stage_info.get(str(evaluated_stage_name))
    if not info:
        return
    group_id = info.get("group_id")
    if not group_id:
        return
    group = descriptor_unordered_groups.get(group_id) or {}
    members = [m for m in (group.get("members") or []) if m in descriptor_stage_info]
    if not members:
        return

    completed_unordered_members.add(str(evaluated_stage_name))
    remaining = [m for m in members if m not in completed_unordered_members]
    total_stages = len(stage_fsm.stages)

    if remaining:
        next_name = min(remaining, key=lambda n: int(descriptor_stage_info[n]["index"]))
        next_index = int(descriptor_stage_info[next_name]["index"])
        stage_fsm.current_stage_index = next_index
        feedback["next_stage"] = next_name
        feedback["task_complete"] = False
        feedback["task_status"] = "IN_PROGRESS"
        feedback["stages_left"] = total_stages - next_index
        return

    next_index = max(int(descriptor_stage_info[m]["index"]) for m in members) + 1
    stage_fsm.current_stage_index = next_index
    if next_index < total_stages:
        feedback["next_stage"] = stage_fsm.stages[next_index]["name"]
        feedback["task_complete"] = False
        feedback["task_status"] = "IN_PROGRESS"
        feedback["stages_left"] = total_stages - next_index
    else:
        feedback["next_stage"] = None
        feedback["task_complete"] = True
        feedback["task_status"] = "COMPLETE"
        feedback["stages_left"] = 0


def _build_stage_descriptors(stage_names, stage_config, region_registry, operation_mode: str):
    op = str(operation_mode).lower()
    out = []
    cfg_by_name = {str(s["name"]): s for s in stage_config}
    for sname in stage_names:
        s_cfg = cfg_by_name.get(str(sname), {})
        desc = {"stage_name": str(sname)}
        if region_registry is not None:
            mode_rules = (region_registry.stage_bindings.get(str(sname), {}) or {}).get(op, {}) or {}
            valid_source = list(mode_rules.get("valid_source_regions", []) or [])
            valid_target = list(mode_rules.get("valid_target_regions", []) or [])
            valid_tool = list(mode_rules.get("valid_tool_regions", []) or [])
        else:
            valid_source, valid_target, valid_tool = [], [], []

        if op == "dismantle":
            desc.update(
                {
                    "manual_start_allowed": bool(s_cfg.get("manual_start_allowed", True)),
                    "start_tool_regions": valid_tool,
                    "completion_regions": valid_target,
                    "completion_roles": s_cfg.get("completion_roles", ["part_drop", "tool_drop"]),
                    "cooldown_frames": int(s_cfg.get("cooldown_frames", 8)),
                }
            )
        else:
            next_source = []
            if region_registry is not None:
                try:
                    i = list(stage_names).index(sname)
                    if i + 1 < len(stage_names):
                        next_name = stage_names[i + 1]
                        next_rules = (region_registry.stage_bindings.get(str(next_name), {}) or {}).get(op, {}) or {}
                        next_source = list(next_rules.get("valid_source_regions", []) or [])
                except Exception:
                    next_source = []
            desc.update(
                {
                    "manual_start_allowed": bool(s_cfg.get("manual_start_allowed", False)),
                    "start_roles": s_cfg.get("start_roles", ["part_pick"]),
                    "start_source_regions": valid_source,
                    "completion_mode": str(s_cfg.get("assembly_completion_mode", "next_stage_pick")),
                    "next_stage_trigger_roles": s_cfg.get("next_stage_trigger_roles", ["part_pick"]),
                    "next_stage_source_regions": s_cfg.get("next_stage_source_regions", next_source),
                    "completion_regions": s_cfg.get("completion_regions", valid_target),
                    "completion_roles": s_cfg.get("completion_roles", ["assembly_target"]),
                    "cooldown_frames": int(s_cfg.get("cooldown_frames", 1)),
                }
            )
        desc.setdefault("expected_parts", list(s_cfg.get("expected_parts", []) or []))
        desc.setdefault("expected_tools", list(s_cfg.get("expected_tools", ["NONE"]) or ["NONE"]))
        desc.setdefault("expected_inventory", dict(s_cfg.get("expected_inventory", {}) or {}))
        desc.setdefault("required_part_counts", dict(s_cfg.get("required_part_counts", {}) or {}))
        desc.setdefault("required_tool_counts", dict(s_cfg.get("required_tool_counts", {}) or {}))
        desc.setdefault("active_part_candidates", list(s_cfg.get("active_part_candidates", desc["expected_parts"]) or []))
        desc.setdefault("active_tool_candidates", list(s_cfg.get("active_tool_candidates", desc["expected_tools"]) or ["NONE"]))
        desc.setdefault("forbidden_parts", list(s_cfg.get("forbidden_parts", []) or []))
        desc.setdefault("forbidden_tools", list(s_cfg.get("forbidden_tools", []) or []))
        desc.setdefault("expected_poses", list(s_cfg.get("expected_poses", []) or []))
        desc.setdefault(
            "enforce_active_tool_match",
            bool(s_cfg.get("enforce_active_tool_match", any(str(x).strip() != "NONE" for x in desc["active_tool_candidates"]))),
        )
        desc.setdefault("enforce_active_part_match", bool(s_cfg.get("enforce_active_part_match", False)))
        desc.setdefault(
            "enforce_expected_pose_match",
            bool(s_cfg.get("enforce_expected_pose_match", bool(desc["expected_poses"]))),
        )
        out.append(desc)
    return out


def _merge_stage_descriptor_expectations(stage_config, stage_descriptors):
    if not stage_config or not stage_descriptors:
        return stage_config
    by_name = {str(d.get("stage_name", "")).strip(): dict(d) for d in (stage_descriptors or [])}
    merged = []
    for stage in stage_config:
        item = dict(stage)
        desc = by_name.get(str(stage.get("name", "")).strip(), {})
        if desc:
            for key in [
                "expected_parts",
                "expected_tools",
                "expected_inventory",
                "required_part_counts",
                "required_tool_counts",
                "active_part_candidates",
                "active_tool_candidates",
                "forbidden_parts",
                "forbidden_tools",
                "expected_poses",
                "enforce_active_tool_match",
                "enforce_active_part_match",
                "enforce_expected_pose_match",
            ]:
                if key in desc:
                    item[key] = desc.get(key)
        merged.append(item)
    return merged


def _descriptors_have_expected_poses(stage_descriptors) -> bool:
    return any(bool((desc or {}).get("expected_poses")) for desc in (stage_descriptors or []))


def _parse_kernel(value: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("kernel must be 'w,h' (e.g. 5,5)")
    try:
        return int(parts[0]), int(parts[1])
    except Exception as exc:
        raise argparse.ArgumentTypeError("kernel must be integers, e.g. 5,5") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "depthai"], default="offline")
    parser.add_argument(
        "--operation-mode",
        choices=["assembly", "dismantle"],
        default="dismantle",
        help="Segmentation operation mode for MotionFSM/region semantics.",
    )
    parser.add_argument(
        "--regions-config",
        default=None,
        help="Optional path to region registry JSON (mixed workspace_norm/image_px regions).",
    )
    parser.add_argument(
        "--motion-fsm-version",
        choices=["v1", "v2"],
        default=MOTION_FSM_VERSION,
        help="Select segmentation FSM engine version.",
    )
    parser.add_argument(
        "--stage-descriptors-config",
        default=None,
        help="Optional JSON file with mode-specific stage descriptors for FSM v2.",
    )
    parser.add_argument(
        "--use-region-detector",
        action="store_true",
        help="Enable classical-CV colored region detector (falls back to registry polygons if detection is weak).",
    )
    parser.add_argument("--stage-names", required=True, help="Comma-separated stage names")
    parser.add_argument(
        "--reference-videos",
        required=True,
        help=(
            "Comma-separated references per stage. "
            "Use '|' to provide multiple references for one stage, "
            "e.g. stageA_ref1.mp4|stageA_ref2.mp4,stageB_ref1.mp4"
        ),
    )
    parser.add_argument(
        "--reference-depths",
        default=None,
        help=(
            "Optional comma-separated depth .npz per stage, matching reference-videos structure. "
            "Use '|' to provide multiple depths for one stage."
        ),
    )
    observed_group = parser.add_mutually_exclusive_group(required=True)
    observed_group.add_argument("--observed-videos", help="Comma-separated observed video paths")
    observed_group.add_argument("--observed-full-video", help="Single observed video containing all stages in sequence")
    parser.add_argument("--observed-depth-npz", default=None, help="Optional depth .npz for --observed-full-video")
    parser.add_argument(
        "--depth-recording",
        default=None,
        help="Alias for --observed-depth-npz (path to recorded depth .npz).",
    )
    parser.add_argument("--expected-tools", default="NONE", help="Comma-separated expected tool labels per stage")
    parser.add_argument(
        "--stage-thresholds",
        default=None,
        help="Comma-separated score thresholds per stage (length 1 or match stage count).",
    )
    parser.add_argument(
        "--margin-mins",
        default=None,
        help="Comma-separated unordered-group margin thresholds per stage (Gate A). Length 1 or match stage count.",
    )
    parser.add_argument(
        "--stage-types",
        default=None,
        help="Comma-separated stage types (ordered,unordered_group). Length 1 or match stage count.",
    )
    parser.add_argument(
        "--required-counts",
        default=None,
        help="Comma-separated required unique completion counts per stage (for unordered_group).",
    )
    parser.add_argument(
        "--subtask-groups",
        default=None,
        help=(
            "Optional comma-separated per-stage subtask grouping by reference indices. "
            "Stage spec format: groups separated by '|', indices in group separated by '+'. "
            "Example: '0+1|2+3,0|1'."
        ),
    )
    parser.add_argument(
        "--lockout-enabled",
        default=None,
        help="Comma-separated booleans per stage to enable used-reference lockout (true/false).",
    )
    parser.add_argument("--anchor-mode", choices=["wrist", "wrist_index_midpoint"], default=ANCHOR_MODE_DEFAULT)
    parser.add_argument("--dominant-hand", choices=["RIGHT", "LEFT"], default=DOMINANT_HAND)
    parser.add_argument(
        "--active-hand-policy",
        choices=["dominant", "closest_to_tool", "best_match"],
        default=ACTIVE_HAND_POLICY_DEFAULT,
    )
    parser.add_argument("--experiment-name", default="video_pipeline")
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--operator-id", default=None, help="Operator identifier for reproducibility metadata.")
    parser.add_argument("--session-id", default="session_video_001")
    parser.add_argument("--trial-id", default="trial_001", help="Trial identifier within a session.")
    parser.add_argument("--calibration", default=None, help="Calibration folder/json path. Default: latest calibration dir")
    parser.add_argument("--show", action="store_true", help="Show frame window while processing")
    parser.add_argument(
        "--pose-pilot-config",
        default=None,
        help="Optional JSON config enabling stage-aware export of pose-pilot candidate frames.",
    )
    parser.add_argument(
        "--pose-pilot-export-dir",
        default=None,
        help="Optional output directory for exported pose-pilot manifest/images.",
    )
    parser.add_argument(
        "--pose-pilot-export-every-n-frames",
        type=int,
        default=5,
        help="Export every Nth eligible frame for the pose pilot.",
    )
    parser.add_argument(
        "--disable-pose-classifier",
        action="store_true",
        help="Disable descriptor-driven object pose classification even when stages define expected_poses.",
    )
    parser.add_argument(
        "--pose-classifier-models",
        default=None,
        help=(
            "Optional comma-separated part=model mapping, e.g. "
            "front_cover=runs/.../best.pt,piston_rod=runs/.../best.pt. "
            "If omitted, matching runs/classify/**/weights/best.pt files are auto-discovered."
        ),
    )
    parser.add_argument(
        "--pose-classifier-device",
        default=None,
        help="Optional YOLO classifier device override, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--pose-classifier-bbox-pad",
        type=float,
        default=0.20,
        help="Detector bbox padding ratio used before pose-classifier crops.",
    )
    parser.add_argument("--delay-ms", type=int, default=DELAY_MS)
    parser.add_argument(
        "--no-hand-frames-to-split",
        type=int,
        default=NO_HAND_FRAMES_TO_SPLIT,
        help="For --observed-full-video: split stage when hand is not visible for N consecutive frames while recording.",
    )
    parser.add_argument(
        "--rearm-no-hand-frames",
        type=int,
        default=REARM_NO_HAND_FRAMES,
        help=(
            "After segment close, require N frames outside drop-region before allowing a new "
            "IDLE->RECORDING start."
        ),
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=MIN_SEGMENT_POINTS,
        help="Discard very short detected segments (< N points) as transient triggers.",
    )
    parser.add_argument("--depth-trigger-enabled", action="store_true", help="Enable depth-based ROI triggers.")
    parser.add_argument("--depth-baseline-frames", type=int, default=DEPTH_BASELINE_FRAMES)
    parser.add_argument("--depth-delta-mm", type=float, default=DEPTH_DELTA_MM)
    parser.add_argument("--depth-dwell-frames", type=int, default=DEPTH_DWELL_FRAMES)
    parser.add_argument("--depth-fallback-rgb", action="store_true", default=DEPTH_FALLBACK_TO_RGB)
    parser.add_argument("--color-trigger-enabled", action="store_true", help="Enable color-change ROI triggers.")
    parser.add_argument("--no-color-trigger", action="store_true", help="Disable color-change triggers.")
    parser.add_argument("--color-baseline-frames", type=int, default=COLOR_BASELINE_FRAMES)
    parser.add_argument("--color-delta-threshold", type=float, default=COLOR_DELTA_THRESHOLD)
    parser.add_argument("--color-delta-range", type=float, default=COLOR_DELTA_RANGE)
    parser.add_argument("--motion-trigger-enabled", action="store_true", help="Enable ROI motion triggers.")
    parser.add_argument("--motion-baseline-frames", type=int, default=MOTION_BASELINE_FRAMES)
    parser.add_argument("--motion-ignore-frames", type=int, default=MOTION_FOCUS_IGNORE_FRAMES)
    parser.add_argument("--motion-diff-threshold", type=int, default=MOTION_DIFF_THRESHOLD)
    parser.add_argument("--motion-score-threshold", type=float, default=MOTION_SCORE_THRESHOLD)
    parser.add_argument("--motion-dwell-frames", type=int, default=MOTION_DWELL_FRAMES)
    parser.add_argument("--motion-history-frames", type=int, default=MOTION_HISTORY_FRAMES)
    parser.add_argument("--motion-blur-kernel", type=_parse_kernel, default=MOTION_BLUR_KERNEL)
    parser.add_argument("--motion-log-interval-frames", type=int, default=MOTION_LOG_INTERVAL_FRAMES)
    parser.add_argument("--motion-global-suppress-threshold", type=float, default=MOTION_GLOBAL_SUPPRESS_THRESHOLD)
    parser.add_argument("--motion-global-relative-factor", type=float, default=MOTION_GLOBAL_RELATIVE_FACTOR)
    parser.add_argument("--fused-trigger-threshold", type=float, default=FUSED_TRIGGER_THRESHOLD)
    parser.add_argument("--fused-weight-color", type=float, default=FUSED_WEIGHT_COLOR)
    parser.add_argument("--fused-weight-depth", type=float, default=FUSED_WEIGHT_DEPTH)
    parser.add_argument("--fused-weight-rgb", type=float, default=FUSED_WEIGHT_RGB)
    parser.add_argument(
        "--hide-depth-roi-overlay",
        action="store_true",
        help="Hide per-ROI depth overlay boxes while keeping FSM overlay.",
    )
    parser.add_argument(
        "--virtual-projection-enabled",
        action="store_true",
        help="Draw stage-aware virtual placement guides on the video overlay.",
    )
    parser.add_argument(
        "--virtual-projection-config",
        default="configs/virtual_projections_v1.json",
        help="JSON config describing per-stage virtual projection guide items.",
    )
    parser.add_argument(
        "--virtual-projection-output-video",
        default=None,
        help="Optional mp4 path to save the pipeline overlay video with virtual projections.",
    )
    parser.add_argument(
        "--virtual-projection-output-fps",
        type=float,
        default=30.0,
        help="FPS used when writing --virtual-projection-output-video.",
    )
    parser.add_argument(
        "--motion-fsm-debug",
        action="store_true",
        help="Enable per-stage FSM debug logging to motion_fsm_debug.json.",
    )
    parser.add_argument(
        "--no-motion-fsm-debug",
        action="store_true",
        help="Disable per-stage FSM debug logging.",
    )
    args = parser.parse_args()

    resolved = resolve_runtime_config(
        args=args,
        score_threshold=STAGE_SCORE_THRESHOLD,
        resolve_latest_calibration_dir=lambda _unused: _resolve_latest_calibration_dir(CALIB_DATA_ROOT),
    )
    stage_names = resolved.stage_names
    stage_ref_videos = resolved.stage_ref_videos
    stage_ref_depths = resolved.stage_ref_depths
    all_ref_videos = resolved.all_ref_videos
    obs_videos = resolved.obs_videos
    obs_full_video = resolved.obs_full_video
    obs_full_depth = resolved.obs_full_depth
    stage_config = resolved.stage_config
    run_options = resolved.run_options
    run_options["operation_mode"] = args.operation_mode
    run_options["regions_config"] = args.regions_config
    run_options["use_region_detector"] = bool(args.use_region_detector)
    run_options["motion_fsm_version"] = args.motion_fsm_version
    run_options["stage_descriptors_config"] = args.stage_descriptors_config
    run_options["pose_classifier_disabled"] = bool(args.disable_pose_classifier)
    run_options["pose_classifier_models"] = args.pose_classifier_models
    run_options["pose_classifier_device"] = args.pose_classifier_device
    run_options["pose_classifier_bbox_pad"] = float(args.pose_classifier_bbox_pad)
    run_options["depth_trigger_enabled"] = bool(args.depth_trigger_enabled or DEPTH_TRIGGER_ENABLED)
    run_options["depth_baseline_frames"] = int(args.depth_baseline_frames)
    run_options["depth_delta_mm"] = float(args.depth_delta_mm)
    run_options["depth_dwell_frames"] = int(args.depth_dwell_frames)
    run_options["depth_fallback_rgb"] = bool(args.depth_fallback_rgb)
    run_options["color_trigger_enabled"] = bool(args.color_trigger_enabled or (COLOR_TRIGGER_ENABLED and not args.no_color_trigger))
    run_options["color_baseline_frames"] = int(args.color_baseline_frames)
    run_options["color_delta_threshold"] = float(args.color_delta_threshold)
    run_options["color_delta_range"] = float(args.color_delta_range)
    run_options["motion_trigger_enabled"] = bool(args.motion_trigger_enabled or MOTION_TRIGGER_ENABLED)
    run_options["motion_baseline_frames"] = int(args.motion_baseline_frames)
    run_options["motion_ignore_frames"] = int(args.motion_ignore_frames)
    run_options["motion_diff_threshold"] = int(args.motion_diff_threshold)
    run_options["motion_score_threshold"] = float(args.motion_score_threshold)
    run_options["motion_dwell_frames"] = int(args.motion_dwell_frames)
    run_options["motion_history_frames"] = int(args.motion_history_frames)
    run_options["motion_blur_kernel"] = tuple(args.motion_blur_kernel)
    run_options["motion_log_interval_frames"] = int(args.motion_log_interval_frames)
    run_options["motion_global_suppress_threshold"] = float(args.motion_global_suppress_threshold)
    run_options["motion_global_relative_factor"] = float(args.motion_global_relative_factor)
    run_options["fused_trigger_threshold"] = float(args.fused_trigger_threshold)
    run_options["fused_weight_color"] = float(args.fused_weight_color)
    run_options["fused_weight_depth"] = float(args.fused_weight_depth)
    run_options["fused_weight_rgb"] = float(args.fused_weight_rgb)
    run_options["show_depth_roi_overlay"] = bool(DEPTH_ROI_OVERLAY_ENABLED and not args.hide_depth_roi_overlay)
    run_options["virtual_projection_enabled"] = bool(
        args.virtual_projection_enabled or bool(args.virtual_projection_output_video)
    )
    run_options["virtual_projection_config"] = args.virtual_projection_config
    run_options["virtual_projection_output_video"] = args.virtual_projection_output_video
    run_options["virtual_projection_output_fps"] = float(args.virtual_projection_output_fps)
    run_options["motion_fsm_debug"] = bool(args.motion_fsm_debug or (MOTION_FSM_DEBUG_LOGGING_ENABLED and not args.no_motion_fsm_debug))
    requested_stage_thresholds = [float(v) for v in run_options.get("stage_thresholds", [])]
    if STAGE_THRESHOLDS_BY_STAGE:
        for s in stage_config:
            if s["name"] in STAGE_THRESHOLDS_BY_STAGE:
                s["score_threshold"] = float(STAGE_THRESHOLDS_BY_STAGE[s["name"]])
    effective_stage_thresholds = [float(s["score_threshold"]) for s in stage_config]
    run_options["requested_stage_thresholds"] = requested_stage_thresholds
    run_options["effective_stage_thresholds"] = effective_stage_thresholds
    # Keep stage_thresholds aligned with actual thresholds used by StageFSM.
    run_options["stage_thresholds"] = effective_stage_thresholds
    calib_path = resolved.calib_path

    _ = CalibrationLoader.load_homography(str(calib_path))  # early validation
    pre = FramePreprocessor(str(calib_path))
    region_registry = RegionRegistry.from_file(args.regions_config) if args.regions_config else None
    region_detector = RegionDetector(region_registry, preprocessor=pre) if (region_registry and args.use_region_detector) else None
    if args.stage_descriptors_config:
        stage_descriptors = load_stage_descriptors(
            config_path=args.stage_descriptors_config,
            operation_mode=run_options["operation_mode"],
            stage_names=stage_names,
        )
    else:
        stage_descriptors = _build_stage_descriptors(
            stage_names=stage_names,
            stage_config=stage_config,
            region_registry=region_registry,
            operation_mode=run_options["operation_mode"],
        )
    stage_config = _merge_stage_descriptor_expectations(stage_config, stage_descriptors)
    pose_pilot_bridge = PosePilotBridge.from_json(args.pose_pilot_config) if args.pose_pilot_config else None
    pose_pilot_exporter = (
        PosePilotExporter(
            args.pose_pilot_export_dir,
            write_every_n_frames=int(args.pose_pilot_export_every_n_frames),
        )
        if args.pose_pilot_export_dir
        else None
    )
    pose_classifier = None
    if _descriptors_have_expected_poses(stage_descriptors) and not args.disable_pose_classifier:
        pose_classifier = PoseClassifierManager.auto(
            model_map=parse_pose_classifier_model_map(args.pose_classifier_models),
            device=args.pose_classifier_device,
            bbox_pad=float(args.pose_classifier_bbox_pad),
        )
        logger.info(
            "Pose classifier enabled for descriptor expected_poses (bbox_pad=%.2f)",
            float(args.pose_classifier_bbox_pad),
        )
    if pose_pilot_bridge is not None:
        logger.info("Pose pilot bridge enabled via %s", args.pose_pilot_config)
    if pose_pilot_exporter is not None:
        logger.info("Pose pilot export enabled at %s", args.pose_pilot_export_dir)
    virtual_projector = None
    if run_options["virtual_projection_enabled"]:
        virtual_projector = VirtualProjectionGuide.from_file(args.virtual_projection_config)
        logger.info("Virtual projection guides enabled via %s", args.virtual_projection_config)
        if region_registry is None:
            logger.warning("Virtual projections need --regions-config; no projection regions will be drawn.")
    descriptor_stage_info, descriptor_unordered_groups = _build_descriptor_unordered_groups(stage_descriptors)
    completed_unordered_members = set()
    stage_references, stage_reference_names = build_stage_references_from_videos(
        stage_names=stage_names,
        stage_ref_videos=stage_ref_videos,
        stage_ref_depths=stage_ref_depths,
        mode=run_options["mode"],
        pre=pre,
        anchor_mode=run_options["anchor_mode"],
        dominant_hand=run_options["dominant_hand"],
        active_hand_policy=run_options["active_hand_policy"],
        show=run_options["show"],
        delay_ms=run_options["delay_ms"],
        logger=logger,
        operation_mode=run_options["operation_mode"],
        region_registry=region_registry,
        region_detector=region_detector,
        motion_fsm_version=run_options["motion_fsm_version"],
        stage_descriptors=stage_descriptors,
        depth_trigger_enabled=run_options["depth_trigger_enabled"],
        depth_baseline_frames=run_options["depth_baseline_frames"],
        depth_delta_mm=run_options["depth_delta_mm"],
        depth_dwell_frames=run_options["depth_dwell_frames"],
        depth_fallback_rgb=run_options["depth_fallback_rgb"],
        show_depth_roi_overlay=run_options["show_depth_roi_overlay"],
        color_trigger_enabled=run_options["color_trigger_enabled"],
        color_baseline_frames=run_options["color_baseline_frames"],
        color_delta_threshold=run_options["color_delta_threshold"],
        color_delta_range=run_options["color_delta_range"],
        motion_trigger_enabled=run_options["motion_trigger_enabled"],
        motion_baseline_frames=run_options["motion_baseline_frames"],
        motion_ignore_frames=run_options["motion_ignore_frames"],
        motion_diff_threshold=run_options["motion_diff_threshold"],
        motion_score_threshold=run_options["motion_score_threshold"],
        motion_dwell_frames=run_options["motion_dwell_frames"],
        motion_history_frames=run_options["motion_history_frames"],
        motion_blur_kernel=run_options["motion_blur_kernel"],
        motion_log_interval_frames=run_options["motion_log_interval_frames"],
        motion_global_suppress_threshold=run_options["motion_global_suppress_threshold"],
        motion_global_relative_factor=run_options["motion_global_relative_factor"],
        fused_trigger_threshold=run_options["fused_trigger_threshold"],
        fused_weight_color=run_options["fused_weight_color"],
        fused_weight_depth=run_options["fused_weight_depth"],
        fused_weight_rgb=run_options["fused_weight_rgb"],
    )

    system_config = build_system_config(
        run_options=run_options,
        stage_weights=STAGE_WEIGHTS,
        stage_score_threshold=STAGE_SCORE_THRESHOLD,
        work_region=WORK_REGION,
        drop_region=DROP_REGION,
        motion_fsm_warmup_frames=MOTION_FSM_WARMUP_FRAMES,
        motion_fsm_min_consecutive_visible=MOTION_FSM_MIN_CONSECUTIVE_VISIBLE,
        motion_fsm_min_consecutive_drop_hits=MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS,
        motion_fsm_min_recording_frames=MOTION_FSM_MIN_RECORDING_FRAMES,
        motion_fsm_max_hand_jump=MOTION_FSM_MAX_HAND_JUMP,
        sync_max_time_diff=SYNC_MAX_TIME_DIFF,
        tool_hand_distance_threshold=TOOL_HAND_DISTANCE_THRESHOLD,
        mp_max_num_hands=MP_MAX_NUM_HANDS,
        mp_model_complexity=MP_MODEL_COMPLEXITY,
        mp_min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        mp_min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        preprocess_observed_oob_policy={"mode": "drop_interpolate", "bounds": [0, 1], "max_gap_threshold": 20},
        preprocess_reference_oob_policy={"mode": "none"},
        operation_mode=run_options["operation_mode"],
        regions_config=run_options["regions_config"],
        use_region_detector=run_options["use_region_detector"],
        motion_fsm_version=run_options["motion_fsm_version"],
        stage_descriptors_config=run_options["stage_descriptors_config"],
    )
    protocol_meta = build_protocol_metadata(
        protocol_version=PROTOCOL_VERSION,
        run_options=run_options,
        stage_names=stage_names,
        expected_tools=resolved.expected_tools,
        stage_config=stage_config,
        references=all_ref_videos,
        observed_videos=obs_videos,
        observed_full_video=obs_full_video,
        calibration_path=calib_path,
    )
    metadata, exp_logger = start_experiment_tracking(
        experiment_name=run_options["experiment_name"],
        user_id=run_options["user_id"],
        operator_id=run_options["operator_id"],
        session_id=run_options["session_id"],
        trial_id=run_options["trial_id"],
        references=[str(p) for p in all_ref_videos],
        system_config=system_config,
        protocol_meta=protocol_meta,
    )

    stage_fsm = StageFSM(
        stage_config=stage_config,
        weights=STAGE_WEIGHTS,
        stage_references=stage_references,
        load_references=False,
    )

    observed_segments = None
    if obs_full_video is not None:
        if run_options["mode"] == "offline":
            observed_segments = extract_all_stage_segments_from_video_offline(
                video_path=obs_full_video,
                pre=pre,
                anchor_mode=run_options["anchor_mode"],
                dominant_hand=run_options["dominant_hand"],
                active_hand_policy=run_options["active_hand_policy"],
                show=run_options["show"],
                delay_ms=run_options["delay_ms"],
                no_hand_frames_to_split=run_options["no_hand_frames_to_split"],
                rearm_no_hand_frames=run_options["rearm_no_hand_frames"],
                min_segment_points=run_options["min_segment_points"],
                operation_mode=run_options["operation_mode"],
                region_registry=region_registry,
                region_detector=region_detector,
                motion_fsm_version=run_options["motion_fsm_version"],
                stage_descriptors=stage_descriptors,
                depth_npz=obs_full_depth,
                depth_trigger_enabled=run_options["depth_trigger_enabled"],
                depth_baseline_frames=run_options["depth_baseline_frames"],
                depth_delta_mm=run_options["depth_delta_mm"],
                depth_dwell_frames=run_options["depth_dwell_frames"],
                depth_fallback_rgb=run_options["depth_fallback_rgb"],
                show_depth_roi_overlay=run_options["show_depth_roi_overlay"],
                color_trigger_enabled=run_options["color_trigger_enabled"],
                color_baseline_frames=run_options["color_baseline_frames"],
                color_delta_threshold=run_options["color_delta_threshold"],
                color_delta_range=run_options["color_delta_range"],
                motion_trigger_enabled=run_options["motion_trigger_enabled"],
                motion_baseline_frames=run_options["motion_baseline_frames"],
                motion_ignore_frames=run_options["motion_ignore_frames"],
                motion_diff_threshold=run_options["motion_diff_threshold"],
                motion_score_threshold=run_options["motion_score_threshold"],
                motion_dwell_frames=run_options["motion_dwell_frames"],
                motion_history_frames=run_options["motion_history_frames"],
                motion_blur_kernel=run_options["motion_blur_kernel"],
                motion_log_interval_frames=run_options["motion_log_interval_frames"],
                motion_global_suppress_threshold=run_options["motion_global_suppress_threshold"],
                motion_global_relative_factor=run_options["motion_global_relative_factor"],
                fused_trigger_threshold=run_options["fused_trigger_threshold"],
                fused_weight_color=run_options["fused_weight_color"],
                fused_weight_depth=run_options["fused_weight_depth"],
                fused_weight_rgb=run_options["fused_weight_rgb"],
                pose_pilot_bridge=pose_pilot_bridge,
                pose_pilot_exporter=pose_pilot_exporter,
                pose_classifier=pose_classifier,
                virtual_projector=virtual_projector,
                virtual_projection_output_video=run_options["virtual_projection_output_video"],
                virtual_projection_output_fps=run_options["virtual_projection_output_fps"],
            )
        else:
            observed_segments = extract_all_stage_segments_from_video_depthai(
                video_path=obs_full_video,
                pre=pre,
                anchor_mode=run_options["anchor_mode"],
                dominant_hand=run_options["dominant_hand"],
                active_hand_policy=run_options["active_hand_policy"],
                show=run_options["show"],
                delay_ms=run_options["delay_ms"],
                no_hand_frames_to_split=run_options["no_hand_frames_to_split"],
                rearm_no_hand_frames=run_options["rearm_no_hand_frames"],
                min_segment_points=run_options["min_segment_points"],
                operation_mode=run_options["operation_mode"],
                region_registry=region_registry,
                region_detector=region_detector,
                motion_fsm_version=run_options["motion_fsm_version"],
                stage_descriptors=stage_descriptors,
                depth_npz=obs_full_depth,
                depth_trigger_enabled=run_options["depth_trigger_enabled"],
                depth_baseline_frames=run_options["depth_baseline_frames"],
                depth_delta_mm=run_options["depth_delta_mm"],
                depth_dwell_frames=run_options["depth_dwell_frames"],
                depth_fallback_rgb=run_options["depth_fallback_rgb"],
                show_depth_roi_overlay=run_options["show_depth_roi_overlay"],
                color_trigger_enabled=run_options["color_trigger_enabled"],
                color_baseline_frames=run_options["color_baseline_frames"],
                color_delta_threshold=run_options["color_delta_threshold"],
                color_delta_range=run_options["color_delta_range"],
                motion_trigger_enabled=run_options["motion_trigger_enabled"],
                motion_baseline_frames=run_options["motion_baseline_frames"],
                motion_ignore_frames=run_options["motion_ignore_frames"],
                motion_diff_threshold=run_options["motion_diff_threshold"],
                motion_score_threshold=run_options["motion_score_threshold"],
                motion_dwell_frames=run_options["motion_dwell_frames"],
                motion_history_frames=run_options["motion_history_frames"],
                motion_blur_kernel=run_options["motion_blur_kernel"],
                motion_log_interval_frames=run_options["motion_log_interval_frames"],
                motion_global_suppress_threshold=run_options["motion_global_suppress_threshold"],
                motion_global_relative_factor=run_options["motion_global_relative_factor"],
                fused_trigger_threshold=run_options["fused_trigger_threshold"],
                fused_weight_color=run_options["fused_weight_color"],
                fused_weight_depth=run_options["fused_weight_depth"],
                fused_weight_rgb=run_options["fused_weight_rgb"],
                pose_pilot_bridge=pose_pilot_bridge,
                pose_pilot_exporter=pose_pilot_exporter,
                pose_classifier=pose_classifier,
                virtual_projector=virtual_projector,
                virtual_projection_output_video=run_options["virtual_projection_output_video"],
                virtual_projection_output_fps=run_options["virtual_projection_output_fps"],
            )
        logger.info("Observed full-video segmentation found %d segment(s)", len(observed_segments))
        _log_segment_summary(observed_segments, header="Observed Segment Summary")
        if len(observed_segments) < len(stage_names):
            logger.warning(
                "Observed full video has fewer segments (%d) than stages (%d). Task may remain incomplete.",
                len(observed_segments),
                len(stage_names),
            )

    if observed_segments is not None:
        eval_items = list(enumerate(observed_segments))
    else:
        eval_items = list(enumerate(obs_videos))

    for i, item in eval_items:
        if stage_fsm.is_task_complete():
            break

        current_stage_before = stage_fsm.stages[stage_fsm.current_stage_index]["name"]
        if observed_segments is not None:
            seg = item
            _sync_stage_fsm_to_segment_stage(
                stage_fsm=stage_fsm,
                segment_stage_name=seg.get("stage_name"),
                descriptor_stage_info=descriptor_stage_info,
            )
            current_stage_before = stage_fsm.stages[stage_fsm.current_stage_index]["name"]
            observed_raw = seg["trajectory_raw"]
            fsm_transitions = seg["fsm_transitions"]
            tool_conf_timeline = seg["tool_conf_timeline"]
            event_timeline = seg.get("event_timeline", [])
            contract_rejections = seg.get("contract_rejections", [])
            trigger_reason = seg.get("trigger_reason")
            completion_reason = seg.get("completion_reason")
            rejected_trigger_reasons = seg.get("rejected_trigger_reasons", [])
            observed_raw_oob = seg["raw_oob_count"]
            bimanual_analytics = seg.get("bimanual_analytics")
            observed_video_label = f"{obs_full_video}#segment_{i}"
        else:
            obs_video = item
            observed_raw, fsm_transitions, tool_conf_timeline, observed_raw_oob, bimanual_analytics = extract_stage_from_video(
                video_path=obs_video,
                mode=run_options["mode"],
                pre=pre,
                anchor_mode=run_options["anchor_mode"],
                dominant_hand=run_options["dominant_hand"],
                active_hand_policy=run_options["active_hand_policy"],
                show=run_options["show"],
                delay_ms=run_options["delay_ms"],
                operation_mode=run_options["operation_mode"],
                region_registry=region_registry,
                region_detector=region_detector,
                motion_fsm_version=run_options["motion_fsm_version"],
                stage_descriptors=[stage_descriptors[min(i, len(stage_descriptors) - 1)]] if stage_descriptors else None,
                depth_npz=None,
                depth_trigger_enabled=run_options["depth_trigger_enabled"],
                depth_baseline_frames=run_options["depth_baseline_frames"],
                depth_delta_mm=run_options["depth_delta_mm"],
                depth_dwell_frames=run_options["depth_dwell_frames"],
                depth_fallback_rgb=run_options["depth_fallback_rgb"],
                show_depth_roi_overlay=run_options["show_depth_roi_overlay"],
                color_trigger_enabled=run_options["color_trigger_enabled"],
                color_baseline_frames=run_options["color_baseline_frames"],
                color_delta_threshold=run_options["color_delta_threshold"],
                color_delta_range=run_options["color_delta_range"],
                motion_trigger_enabled=run_options["motion_trigger_enabled"],
                motion_baseline_frames=run_options["motion_baseline_frames"],
                motion_ignore_frames=run_options["motion_ignore_frames"],
                motion_diff_threshold=run_options["motion_diff_threshold"],
                motion_score_threshold=run_options["motion_score_threshold"],
                motion_dwell_frames=run_options["motion_dwell_frames"],
                motion_history_frames=run_options["motion_history_frames"],
                motion_blur_kernel=run_options["motion_blur_kernel"],
                motion_log_interval_frames=run_options["motion_log_interval_frames"],
                motion_global_suppress_threshold=run_options["motion_global_suppress_threshold"],
                motion_global_relative_factor=run_options["motion_global_relative_factor"],
                fused_trigger_threshold=run_options["fused_trigger_threshold"],
                fused_weight_color=run_options["fused_weight_color"],
                fused_weight_depth=run_options["fused_weight_depth"],
                fused_weight_rgb=run_options["fused_weight_rgb"],
                pose_pilot_bridge=pose_pilot_bridge,
                pose_pilot_exporter=pose_pilot_exporter,
                pose_classifier=pose_classifier,
                virtual_projector=virtual_projector,
                virtual_projection_output_video=run_options["virtual_projection_output_video"],
                virtual_projection_output_fps=run_options["virtual_projection_output_fps"],
            )
            event_timeline = (bimanual_analytics or {}).get("event_timeline", [])
            contract_rejections = (bimanual_analytics or {}).get("contract_rejections", [])
            trigger_reason = (bimanual_analytics or {}).get("trigger_reason")
            completion_reason = (bimanual_analytics or {}).get("completion_reason")
            rejected_trigger_reasons = (bimanual_analytics or {}).get("rejected_trigger_reasons", [])
            observed_video_label = str(obs_video)

        feedback = stage_fsm.evaluate(observed_raw, raw_oob_count=observed_raw_oob)
        evaluated_stage_name = feedback.get("stage_name", current_stage_before)
        if observed_segments is not None:
            _postprocess_unordered_segment_progress(
                stage_fsm=stage_fsm,
                feedback=feedback,
                evaluated_stage_name=evaluated_stage_name,
                descriptor_stage_info=descriptor_stage_info,
                descriptor_unordered_groups=descriptor_unordered_groups,
                completed_unordered_members=completed_unordered_members,
            )
        reference_id = feedback.get("reference_id", -1)
        if reference_id is not None and reference_id >= 0:
            feedback["reference_name"] = stage_reference_names[evaluated_stage_name][reference_id]

        log_stage_evaluation_artifacts(
            metadata=metadata,
            exp_logger=exp_logger,
            evaluated_stage_name=evaluated_stage_name,
            feedback=feedback,
            observed_raw=observed_raw,
            observed_raw_oob=observed_raw_oob,
            observed_video_label=observed_video_label,
            input_segment_index=i,
            fsm_transitions=fsm_transitions,
            tool_conf_timeline=tool_conf_timeline,
            event_timeline=event_timeline,
            contract_rejections=contract_rejections,
            trigger_reason=trigger_reason,
            completion_reason=completion_reason,
            rejected_trigger_reasons=rejected_trigger_reasons,
            bimanual_analytics=bimanual_analytics,
            stage_references=stage_references,
            user_id=run_options["user_id"],
            operation_mode=run_options["operation_mode"],
            motion_fsm_debug_enabled=run_options["motion_fsm_debug"],
        )

        logger.info(
            "Stage=%s raw_points=%d raw_oob=%d score=%s pass=%s",
            evaluated_stage_name,
            len(observed_raw),
            observed_raw_oob,
            feedback.get("final_score"),
            feedback.get("stage_passed"),
        )

        # In per-stage observed video mode, stop when current stage has not fully
        # advanced. For unordered_group this requires group completion, not just pass.
        if observed_segments is None and not _per_stage_run_should_continue(feedback):
            if bool(feedback.get("stage_passed", False)):
                logger.warning(
                    "Per-stage mode: stage attempt passed but group not complete; "
                    "stopping to avoid consuming next stage video early."
                )
            else:
                logger.warning("Stage failed in per-stage mode; stopping further stage evaluation.")
            break

    if observed_segments is not None and run_options["motion_fsm_debug"] and getattr(metadata, "experiment_path", None) is not None:
        append_motion_fsm_debug_payload(
            metadata.experiment_path,
            {
                "kind": "run_summary",
                "operation_mode": run_options["operation_mode"],
                "observed_full_video": (str(obs_full_video) if obs_full_video is not None else None),
                "segments": [
                    {
                        "stage_name": row.get("stage_name", "UNKNOWN"),
                        "start_frame": row.get("start_frame"),
                        "end_frame": row.get("end_frame"),
                        "end_label": ("EOF" if row.get("end_frame") is None else row.get("end_frame")),
                        "duration_s": row.get("duration_s"),
                        "closure_type": row.get("closure_type", "unknown"),
                    }
                    for row in observed_segments
                ],
            },
        )

    metadata.finalize()
    metadata_errors = validate_metadata_file(metadata.experiment_path / "metadata.json")
    if metadata_errors:
        raise RuntimeError(f"Generated metadata failed validation: {metadata_errors}")
    logger.info("Done. Outputs under: %s", metadata.experiment_path)


if __name__ == "__main__":
    main()
