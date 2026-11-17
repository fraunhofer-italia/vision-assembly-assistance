from __future__ import annotations

from typing import Dict, List

from assembly_assistant.analysis.fatigue_metrics import compute_phase1_fatigue_metrics
from assembly_assistant.trajectory.preprocessing import TrajectoryPreprocessor
from assembly_assistant.utils.experiment_logger import ExperimentLogger
from assembly_assistant.utils.fatigue_logger import append_fatigue_metrics
from assembly_assistant.utils.metadata_manager import MetadataManager
from assembly_assistant.utils.stage_metrics_logger import append_stage_metrics
from pathlib import Path
import json


def append_motion_fsm_debug_payload(experiment_path: Path, payload: dict) -> None:
    path = experiment_path / "motion_fsm_debug.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    else:
        existing = []
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(payload)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def build_system_config(
    *,
    run_options: dict,
    stage_weights: dict,
    stage_score_threshold: float,
    work_region,
    drop_region,
    motion_fsm_warmup_frames: int,
    motion_fsm_min_consecutive_visible: int,
    motion_fsm_min_consecutive_drop_hits: int,
    motion_fsm_min_recording_frames: int,
    motion_fsm_max_hand_jump: float,
    sync_max_time_diff: float,
    tool_hand_distance_threshold: float,
    mp_max_num_hands: int,
    mp_model_complexity: int,
    mp_min_detection_confidence: float,
    mp_min_tracking_confidence: float,
    preprocess_observed_oob_policy: dict,
    preprocess_reference_oob_policy: dict,
    operation_mode: str = "dismantle",
    regions_config: str | None = None,
    use_region_detector: bool = False,
    motion_fsm_version: str = "v2",
    stage_descriptors_config: str | None = None,
) -> dict:
    return {
        "mode": run_options["mode"],
        "anchor_mode": run_options["anchor_mode"],
        "dominant_hand": run_options["dominant_hand"],
        "active_hand_policy": run_options.get("active_hand_policy", "dominant"),
        "operation_mode": operation_mode,
        "regions_config": regions_config,
        "use_region_detector": bool(use_region_detector),
        "motion_fsm_version": str(motion_fsm_version),
        "stage_descriptors_config": stage_descriptors_config,
        "stage_weights": stage_weights,
        "stage_score_threshold": stage_score_threshold,
        "motion_fsm": {"work_region": work_region, "drop_region": drop_region},
        "motion_fsm_trigger_point_mode": run_options["anchor_mode"],
        "motion_fsm_warmup_frames": motion_fsm_warmup_frames,
        "motion_fsm_min_consecutive_visible": motion_fsm_min_consecutive_visible,
        "motion_fsm_min_consecutive_drop_hits": motion_fsm_min_consecutive_drop_hits,
        "motion_fsm_min_recording_frames": motion_fsm_min_recording_frames,
        "motion_fsm_max_hand_jump": motion_fsm_max_hand_jump,
        "sync_max_time_diff": sync_max_time_diff,
        "tool_hand_distance_threshold": tool_hand_distance_threshold,
        "mediapipe": {
            "max_num_hands": mp_max_num_hands,
            "model_complexity": mp_model_complexity,
            "min_detection_confidence": mp_min_detection_confidence,
            "min_tracking_confidence": mp_min_tracking_confidence,
        },
        "segment_split_policy": {
            "no_hand_frames_to_split": run_options["no_hand_frames_to_split"],
            "rearm_no_hand_frames": run_options["rearm_no_hand_frames"],
            "min_segment_points": run_options["min_segment_points"],
        },
        "preprocess_observed_oob_policy": preprocess_observed_oob_policy,
        "preprocess_reference_oob_policy": preprocess_reference_oob_policy,
    }


def start_experiment_tracking(
    *,
    experiment_name: str,
    user_id: str,
    session_id: str,
    operator_id: str,
    trial_id: str,
    references: List[str],
    system_config: dict,
    protocol_meta: dict,
    experiments_root: str | None = None,
):
    metadata = MetadataManager.get_instance()
    metadata.create_experiment(
        experiment_name=experiment_name,
        user_id=user_id,
        session_id=session_id,
        references=references,
        config=system_config,
        session_meta={
            "operator_id": operator_id,
            "trial_id": trial_id,
        },
        protocol_meta=protocol_meta,
        root=experiments_root,
    )

    exp_logger = ExperimentLogger.get_instance()
    exp_logger.start_experiment(metadata.experiment_path.name, root=experiments_root)
    return metadata, exp_logger


def log_stage_evaluation_artifacts(
    *,
    metadata,
    exp_logger,
    evaluated_stage_name: str,
    feedback: Dict,
    observed_raw: List[dict],
    observed_raw_oob: int,
    observed_video_label: str,
    input_segment_index: int,
    fsm_transitions: List[dict],
    tool_conf_timeline: List[dict],
    event_timeline: List[dict] | None,
    contract_rejections: List[dict] | None,
    trigger_reason: Dict | None,
    completion_reason: Dict | None,
    rejected_trigger_reasons: List[dict] | None,
    bimanual_analytics: Dict | None,
    stage_references: Dict[str, List[List[dict]]],
    user_id: str,
    operation_mode: str | None = None,
    motion_fsm_debug_enabled: bool = True,
):
    stage_type = str(feedback.get("stage_type", "ordered")).lower()
    completed_subtasks = feedback.get("completed_subtasks")
    if not isinstance(completed_subtasks, list):
        completed_subtasks = []
    completed_subtasks = [str(x) for x in completed_subtasks]

    total_subtasks = None
    remaining_subtasks = None
    if stage_type == "unordered_group":
        gp = feedback.get("group_progress") or {}
        total_subtasks = gp.get("total_subtasks")
        if total_subtasks is None:
            total_subtasks = int(len(stage_references.get(evaluated_stage_name, [])))
        total_subtasks = int(total_subtasks)
        # Default subtask ids are S1..SN unless custom schema provides explicit IDs.
        all_ids = [f"S{i + 1}" for i in range(total_subtasks)]
        completed_set = set(completed_subtasks)
        remaining_subtasks = [sid for sid in all_ids if sid not in completed_set]

    bimanual_meta = dict(bimanual_analytics or {})
    for k in (
        "event_timeline",
        "contract_rejections",
        "trigger_reason",
        "completion_reason",
        "rejected_trigger_reasons",
        "depth_stats",
        "depth_fallback_count",
        "color_trigger_stats",
        "fused_trigger_hits",
    ):
        bimanual_meta.pop(k, None)

    metadata_feedback = dict(feedback)
    metadata_feedback.pop("fatigue_metrics", None)
    metadata_feedback.pop("temporal_window_scores", None)

    metadata.add_fsm_decision(
        {
            "stage": evaluated_stage_name,
            "operation_mode": operation_mode,
            "feedback": metadata_feedback,
            "raw_oob_count": observed_raw_oob,
            "observed_video": observed_video_label,
            "input_segment_index": input_segment_index,
            "matched_subtask_id": feedback.get("matched_subtask_id"),
            "matched_reference_id": feedback.get("matched_reference_id"),
            "completed_subtasks": completed_subtasks,
            "remaining_subtasks": remaining_subtasks,
            "total_subtasks": total_subtasks,
            "excluded_reference_indices": feedback.get("excluded_reference_indices"),
            "group_progress": feedback.get("group_progress"),
        }
    )

    observed_processed = TrajectoryPreprocessor.preprocess(
        observed_raw,
        oob_policy={"mode": "drop_interpolate", "bounds": [0, 1], "max_gap_threshold": 20},
    )
    fatigue_metrics = compute_phase1_fatigue_metrics(observed_processed)

    reference_id = feedback.get("reference_id", -1)
    reference_used = None
    if reference_id is not None and reference_id >= 0:
        reference_used = stage_references[evaluated_stage_name][reference_id]

    exp_logger.log_stage_decision(
        trajectory_id=evaluated_stage_name,
        stage_name=evaluated_stage_name,
        reference_id=reference_id,
        dtw_distance=1.0 - feedback.get("dtw_similarity", 0.0),
        stage_passed=feedback.get("stage_passed", False),
        fsm_transitions=fsm_transitions,
        tool_confidence_timeline=tool_conf_timeline,
        dtw_alignment_path=feedback.get("alignment_path"),
        observed_traj=observed_processed,
        reference_traj=reference_used,
        user_id=user_id,
    )

    if getattr(metadata, "experiment_path", None) is not None:
        append_stage_metrics(
            metadata.experiment_path,
            {
                "stage_name": evaluated_stage_name,
                "input_segment_index": input_segment_index,
                "observed_video": observed_video_label,
                "raw_oob_count": observed_raw_oob,
                "reference_id": reference_id,
                "reference_name": feedback.get("reference_name"),
                "final_score": feedback.get("final_score"),
                "dtw_similarity": feedback.get("dtw_similarity"),
                "tool_match_score": feedback.get("tool_match_score"),
                "smoothness_score": feedback.get("smoothness_score"),
                "efficiency_score": feedback.get("efficiency_score"),
                "temporal_stability_score": feedback.get("temporal_stability_score"),
                "stage_passed": feedback.get("stage_passed"),
                "fatigue_metrics": feedback.get("fatigue_metrics", fatigue_metrics),
                "temporal_window_scores": feedback.get("temporal_window_scores", []),
                "temporal_consistency": feedback.get("temporal_consistency", {}),
                "bimanual_analytics": bimanual_meta,
            },
        )

        append_fatigue_metrics(
            metadata.experiment_path,
            {
                "stage_name": evaluated_stage_name,
                "input_segment_index": input_segment_index,
                "observed_video": observed_video_label,
                "final_score": feedback.get("final_score"),
                "dtw_similarity": feedback.get("dtw_similarity"),
                "tool_match_score": feedback.get("tool_match_score"),
                "smoothness_score": feedback.get("smoothness_score"),
                "efficiency_score": feedback.get("efficiency_score"),
                "temporal_stability_score": feedback.get("temporal_stability_score"),
                "stage_passed": feedback.get("stage_passed"),
                "metrics": fatigue_metrics,
                "bimanual_analytics": bimanual_meta,
            },
        )

        if motion_fsm_debug_enabled:
            debug_payload = {
                "stage": evaluated_stage_name,
                "operation_mode": operation_mode,
                "observed_video": observed_video_label,
                "input_segment_index": input_segment_index,
                "contract_rejections": list(contract_rejections or []),
                "rejected_trigger_reasons": list(rejected_trigger_reasons or []),
                "depth_stats": (bimanual_analytics or {}).get("depth_stats"),
                "depth_fallback_count": (bimanual_analytics or {}).get("depth_fallback_count", 0),
                "color_trigger_stats": (bimanual_analytics or {}).get("color_trigger_stats"),
                "fused_trigger_hits": (bimanual_analytics or {}).get("fused_trigger_hits", 0),
            }
            if debug_payload.get("depth_stats") is not None:
                debug_payload["depth_trigger_hits"] = debug_payload["depth_stats"].get("trigger_hits", 0)
            append_motion_fsm_debug_payload(metadata.experiment_path, debug_payload)
