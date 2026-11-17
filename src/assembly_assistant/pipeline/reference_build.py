from __future__ import annotations

from typing import Dict, List

from assembly_assistant.pipeline.video_extractors import extract_stage_from_video
from assembly_assistant.trajectory.preprocessing import TrajectoryPreprocessor


def build_stage_references_from_videos(
    *,
    stage_names: List[str],
    stage_ref_videos: Dict[str, List],
    stage_ref_depths: Dict[str, List] | None = None,
    mode: str,
    pre,
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    show: bool,
    delay_ms: int,
    logger,
    operation_mode: str = "dismantle",
    region_registry=None,
    region_detector=None,
    motion_fsm_version: str = "v1",
    stage_descriptors: List[dict] | None = None,
    depth_trigger_enabled: bool = False,
    depth_baseline_frames: int = 30,
    depth_delta_mm: float = 25.0,
    depth_dwell_frames: int = 6,
    depth_fallback_rgb: bool = True,
    show_depth_roi_overlay: bool = True,
    color_trigger_enabled: bool = False,
    color_baseline_frames: int = 30,
    color_delta_threshold: float = 18.0,
    color_delta_range: float = 40.0,
    motion_trigger_enabled: bool = False,
    motion_baseline_frames: int = 30,
    motion_ignore_frames: int = 0,
    motion_diff_threshold: int = 18,
    motion_score_threshold: float = 0.02,
    motion_dwell_frames: int = 6,
    motion_history_frames: int = 10,
    motion_blur_kernel: tuple[int, int] = (5, 5),
    motion_log_interval_frames: int = 30,
    motion_global_suppress_threshold: float = 0.03,
    motion_global_relative_factor: float = 2.0,
    motion_strong_score_threshold: float = 0.20,
    motion_strong_relative_factor: float = 1.5,
    fused_trigger_threshold: float = 0.65,
    fused_weight_color: float = 0.6,
    fused_weight_depth: float = 0.2,
    fused_weight_rgb: float = 0.2,
):
    """
    Build and preprocess stage references from reference video files.

    Returns:
      stage_references: dict[stage_name] -> list[processed_reference_trajectory]
      stage_reference_names: dict[stage_name] -> list[reference_name]
    """
    stage_references: Dict[str, List[List[dict]]] = {}
    stage_reference_names: Dict[str, List[str]] = {}

    for stage_name in stage_names:
        refs = stage_ref_videos[stage_name]
        processed_refs = []
        ref_names = []
        for ref_idx, ref_video in enumerate(refs):
            depth_npz = None
            if stage_ref_depths and stage_name in stage_ref_depths:
                depth_list = stage_ref_depths.get(stage_name, [])
                if ref_idx < len(depth_list):
                    depth_npz = depth_list[ref_idx]
            ref_raw, _, _, ref_oob, _ = extract_stage_from_video(
                video_path=ref_video,
                mode=mode,
                pre=pre,
                anchor_mode=anchor_mode,
                dominant_hand=dominant_hand,
                active_hand_policy=active_hand_policy,
                show=show,
                delay_ms=delay_ms,
                operation_mode=operation_mode,
                region_registry=region_registry,
                region_detector=region_detector,
                motion_fsm_version=motion_fsm_version,
                stage_descriptors=stage_descriptors,
                stage_name=stage_name,
                depth_npz=depth_npz,
                depth_trigger_enabled=depth_trigger_enabled,
                depth_baseline_frames=depth_baseline_frames,
                depth_delta_mm=depth_delta_mm,
                depth_dwell_frames=depth_dwell_frames,
                depth_fallback_rgb=depth_fallback_rgb,
                show_depth_roi_overlay=show_depth_roi_overlay,
                color_trigger_enabled=color_trigger_enabled,
                color_baseline_frames=color_baseline_frames,
                color_delta_threshold=color_delta_threshold,
                color_delta_range=color_delta_range,
                motion_trigger_enabled=motion_trigger_enabled,
                motion_baseline_frames=motion_baseline_frames,
                motion_ignore_frames=motion_ignore_frames,
                motion_diff_threshold=motion_diff_threshold,
                motion_score_threshold=motion_score_threshold,
                motion_dwell_frames=motion_dwell_frames,
                motion_history_frames=motion_history_frames,
                motion_blur_kernel=motion_blur_kernel,
                motion_log_interval_frames=motion_log_interval_frames,
                motion_global_suppress_threshold=motion_global_suppress_threshold,
                motion_global_relative_factor=motion_global_relative_factor,
                motion_strong_score_threshold=motion_strong_score_threshold,
                motion_strong_relative_factor=motion_strong_relative_factor,
                fused_trigger_threshold=fused_trigger_threshold,
                fused_weight_color=fused_weight_color,
                fused_weight_depth=fused_weight_depth,
                fused_weight_rgb=fused_weight_rgb,
            )
            ref_processed = TrajectoryPreprocessor.preprocess(ref_raw, oob_policy={"mode": "none"})
            processed_refs.append(ref_processed)
            ref_names.append(ref_video.stem)
            logger.info(
                "Reference stage=%s ref=%s points=%d raw_oob=%d",
                stage_name,
                ref_video.name,
                len(ref_processed),
                ref_oob,
            )

        stage_references[stage_name] = processed_refs
        stage_reference_names[stage_name] = ref_names

    return stage_references, stage_reference_names
