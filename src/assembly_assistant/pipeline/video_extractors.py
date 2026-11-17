from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import cv2
import numpy as np

from configs.experiment_configuration import (
    ACTIVE_HAND_SWITCH_CONFIRM_FRAMES,
    ACTIVE_HAND_SWITCH_SCORE_MARGIN,
    ACTIVE_HAND_TOOL_DISTANCE_NORM,
    ACTIVE_HAND_VELOCITY_NORM_SCALE,
    ACTIVE_HAND_WEIGHT_DOMINANT,
    ACTIVE_HAND_WEIGHT_TOOL,
    ACTIVE_HAND_WEIGHT_VELOCITY,
    DROP_REGION,
    HANDEDNESS_FALLBACK_IMAGE_MIRRORED,
    HANDEDNESS_SWAP_LABELS,
    MOTION_FSM_ACTIVITY_V_MIN,
    MOTION_FSM_DONE_COOLDOWN_FRAMES,
    MOTION_FSM_DROP_WINDOW_FRAMES,
    MOTION_FSM_VERSION,
    MOTION_FSM_MAX_HAND_JUMP,
    MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS,
    MOTION_FSM_MIN_CONSECUTIVE_VISIBLE,
    MOTION_FSM_START_NO_ACTIVITY_FRAMES,
    MOTION_FSM_START_RECENT_HIT_COUNT,
    MOTION_FSM_START_RECENT_HIT_WINDOW,
    MOTION_FSM_REQUIRE_ANCHOR_BEFORE_COMPLETION,
    MOTION_FSM_ANCHOR_STARTUP_GRACE_FRAMES,
    MOTION_FSM_STRICT_MOTION_HITS_ONLY,
    MOTION_FSM_MIN_RECORDING_FRAMES,
    MOTION_FSM_OCCLUSION_TOLERANCE_FRAMES,
    MOTION_FSM_PAUSE_TOLERANCE_FRAMES,
    MOTION_FSM_WARMUP_FRAMES,
    MOTION_GLOBAL_RELATIVE_FACTOR,
    MOTION_GLOBAL_SUPPRESS_THRESHOLD,
    MOTION_STRONG_RELATIVE_FACTOR,
    MOTION_STRONG_SCORE_THRESHOLD,
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    SYNC_MAX_TIME_DIFF,
    TOOL_HAND_DISTANCE_THRESHOLD,
    WORK_REGION,
)
from assembly_assistant.fsm.motion_fsm import MotionFSM
from assembly_assistant.fsm.motion_fsm_v2 import AssemblyPolicy, DismantlePolicy, MotionFSMV2
from assembly_assistant.pipeline.depth_trigger import DepthTriggerExtractor
from assembly_assistant.pipeline.color_change_trigger import ColorChangeTrigger
from assembly_assistant.pipeline.roi_motion_trigger import ROIMotionTrigger
from assembly_assistant.pipeline.event_extractor import RegionEventExtractor
from assembly_assistant.pipeline.region_detector import RegionDetector
from assembly_assistant.pipeline.region_registry import RegionRegistry
from assembly_assistant.pipeline.virtual_projection import VirtualProjectionGuide
from assembly_assistant.detectors.rfdetr_object_detector import RFDETRDependencyError, RFDETRObjectDetector
from assembly_assistant.pose.bridge import PosePilotBridge, PosePilotExporter
from assembly_assistant.pose.pose_classifier import PoseClassifierManager
from assembly_assistant.synchronization.sync_manager import SyncManager
from assembly_assistant.trajectory.active_hand_selector import ActiveHandSelector
from assembly_assistant.trajectory.frame_processor import FramePreprocessor
from assembly_assistant.trajectory.trajectory_builder import TrajectoryBuilder

from assembly_assistant.pipeline.overlays import (
    draw_anchor_overlay,
    draw_color_roi_overlay,
    draw_detection_overlay,
    draw_depth_roi_overlay,
    draw_depth_trigger_overlay,
    draw_fsm_overlay,
    draw_motion_roi_overlay,
    draw_region_registry_overlay,
    draw_workspace_and_drop_overlay,
)
from assembly_assistant.utils.logger import get_logger


logger = get_logger(__name__)


RFDETR_CHECKPOINT_DEFAULT = Path(
    "data/detector_dataset_v1/runs/piston_detector_rfdetr_small_120e_run00/checkpoint_best_total.pth"
)
RFDETR_DATASET_DIR_DEFAULT = Path("data/detector_dataset_v1/Assembly_parts_detection.v2i.coco")
OBJECT_DEBUG_LOG_INTERVAL_FRAMES = 120
MOTION_HAND_FALLBACK_SCORE_THRESHOLD = 0.70
MOTION_INTERACTION_NEAR_MARGIN_NORM = 0.055
MOTION_INTERACTION_APPROACH_DELTA_NORM = 0.025


def _load_depth_npz(path: Path):
    data = np.load(str(path))
    depth_mm = data.get("depth_mm")
    rgb_ts = data.get("rgb_timestamps")
    depth_ts = data.get("depth_timestamps")
    return depth_mm, rgb_ts, depth_ts


def _warp_norm_to_px(pt_norm, H_inv):
    x, y = float(pt_norm[0]), float(pt_norm[1])
    p = np.array([x, y, 1.0], dtype=float)
    p = H_inv @ p
    p = p / max(1e-9, p[2])
    return [float(p[0]), float(p[1])]


def _resolve_video_shape(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for shape: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame for shape: {video_path}")
    h, w = frame.shape[:2]
    return (h, w)


def _build_depth_roi_masks(pre: FramePreprocessor, region_registry: RegionRegistry, shape_hw: tuple[int, int], operation_mode: str):
    h, w = shape_hw
    masks = {}
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        H_inv = None
    for rid, region in region_registry.regions.items():
        roles = [r for r in region.roles if region_registry.resolve_role_enabled(rid, operation_mode, r)]
        if not any(r in {"part_pick", "part_drop", "tool_pick", "tool_drop"} for r in roles):
            continue
        pts = None
        if region.coordinate_frame == "image_px":
            pts = region.polygon
        elif region.coordinate_frame == "workspace_norm":
            if H_inv is None:
                continue
            pts = [_warp_norm_to_px(p, H_inv) for p in region.polygon]
        if not pts:
            continue
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [arr], 255)
        masks[rid] = mask
    return masks


def _build_color_roi_masks(pre: FramePreprocessor, region_registry: RegionRegistry, shape_hw: tuple[int, int], operation_mode: str):
    # Reuse depth ROI masking rules (pick/drop/tool regions only).
    return _build_depth_roi_masks(pre, region_registry, shape_hw, operation_mode)


def _none_tool(timestamp: float) -> dict:
    return {
        "class": "NONE",
        "bbox": (0.0, 0.0, 0.0, 0.0),
        "confidence": 0.0,
        "timestamp": timestamp,
    }


def _make_object_detector():
    if not RFDETR_CHECKPOINT_DEFAULT.exists():
        logger.info("RF-DETR checkpoint not found at %s; object-aware detection disabled.", RFDETR_CHECKPOINT_DEFAULT)
        return None
    try:
        detector = RFDETRObjectDetector(
            checkpoint_path=RFDETR_CHECKPOINT_DEFAULT,
            dataset_dir=RFDETR_DATASET_DIR_DEFAULT,
            model_size="small",
            device="cpu",
            threshold=0.25,
        )
        logger.info("Loaded RF-DETR detector from %s", RFDETR_CHECKPOINT_DEFAULT)
        return detector
    except (RFDETRDependencyError, FileNotFoundError, ValueError) as exc:
        logger.warning("Failed to initialize RF-DETR detector: %s", exc)
        return None


def _make_workspace_filter(pre: FramePreprocessor):
    x_min, y_min, x_max, y_max = WORK_REGION

    def _predicate(det: dict) -> bool:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        try:
            warped = pre.warp_point(center)
        except Exception:
            return False
        wx, wy = float(warped[0]), float(warped[1])
        return x_min <= wx <= x_max and y_min <= wy <= y_max

    return _predicate


def _resolve_current_stage_descriptor(motion, stage_descriptors: List[dict] | None, fallback_stage_name: str | None = None) -> dict:
    descriptors = list(stage_descriptors or [])
    if fallback_stage_name:
        target = str(fallback_stage_name).strip()
        for desc in descriptors:
            if str(desc.get("stage_name", "")).strip() == target:
                return desc
    if motion is not None and descriptors:
        idx = int(getattr(motion, "current_stage_index", -1))
        if 0 <= idx < len(descriptors):
            return descriptors[idx]
    return {}


def _log_object_awareness_debug(frame_data: dict, frame_id: int, stage_name: str | None = None) -> None:
    if frame_id % OBJECT_DEBUG_LOG_INTERVAL_FRAMES != 0:
        return
    logger.info(
        "Object awareness frame=%s stage=%s active_part=%s active_tool=%s workspace_inventory=%s expected_parts=%s expected_tools=%s",
        frame_id,
        stage_name,
        (frame_data.get("active_part") or {}).get("label"),
        (frame_data.get("active_tool") or {}).get("label"),
        frame_data.get("workspace_inventory", {}),
        frame_data.get("expected_parts", []),
        frame_data.get("expected_tools", []),
    )


def _maybe_export_pose_pilot_candidate(
    *,
    frame,
    frame_data: dict,
    stage_name: str | None,
    frame_id: int,
    timestamp: float,
    pose_pilot_bridge: PosePilotBridge | None,
    pose_pilot_exporter: PosePilotExporter | None,
) -> None:
    if pose_pilot_bridge is None:
        return
    decision = pose_pilot_bridge.build_decision(
        frame_data=frame_data,
        stage_name=stage_name,
        frame_id=frame_id,
        timestamp=timestamp,
        frame_shape=frame.shape,
    )
    if decision is None:
        frame_data["pose_pilot_candidate"] = None
        return
    frame_data["pose_pilot_candidate"] = decision.to_record()
    exported_path = None
    if pose_pilot_exporter is not None:
        exported_path = pose_pilot_exporter.export(frame=frame.copy(), decision=decision)
    if exported_path is not None:
        frame_data["pose_pilot_candidate"]["rgb_frame_path"] = str(exported_path)
    logger.debug(
        "Pose pilot candidate frame=%s stage=%s part=%s conf=%.3f exported=%s",
        frame_id,
        stage_name,
        decision.part_label,
        decision.confidence,
        str(exported_path) if exported_path is not None else None,
    )


def _attach_pose_classifier_predictions(
    *,
    frame,
    frame_data: dict,
    current_stage_desc: dict,
    frame_id: int,
    pose_classifier: PoseClassifierManager | None,
) -> None:
    expected_poses = list((current_stage_desc or {}).get("expected_poses") or [])
    frame_data["expected_poses"] = expected_poses
    frame_data["pose_predictions"] = []
    if pose_classifier is None or not expected_poses:
        return
    predictions = pose_classifier.predict(
        frame_bgr=frame,
        detections=frame_data.get("detections") or [],
        expected_poses=expected_poses,
    )
    for pred in predictions:
        pred["frame_id"] = int(frame_id)
    frame_data["pose_predictions"] = predictions
    frame_data.setdefault("object_debug", {})["pose_predictions"] = predictions


def _fallback_label_from_geometry(kp_norm, image_mirrored: bool) -> str:
    # Use wrist x in normalized workspace to infer a stable side label when
    # detector handedness metadata is unavailable.
    wrist_x = float(kp_norm[0][0])
    if image_mirrored:
        return "RIGHT" if wrist_x < 0.5 else "LEFT"
    return "LEFT" if wrist_x < 0.5 else "RIGHT"


def _resolve_offline_hand_label(
    *,
    raw_label,
    hand_index: int,
    kp_norm,
    prev_labels_by_index: dict,
    left_assigned: bool,
    right_assigned: bool,
) -> str:
    label = str(raw_label).upper() if raw_label is not None else None
    if HANDEDNESS_SWAP_LABELS:
        if label == "LEFT":
            label = "RIGHT"
        elif label == "RIGHT":
            label = "LEFT"
    if label in {"LEFT", "RIGHT"}:
        prev_labels_by_index[hand_index] = label
        return label

    # 1) Temporal continuity by detector index.
    prev = prev_labels_by_index.get(hand_index)
    if prev in {"LEFT", "RIGHT"}:
        # Avoid duplicating same side for two hands in the same frame.
        if prev == "LEFT" and left_assigned and not right_assigned:
            prev = "RIGHT"
        elif prev == "RIGHT" and right_assigned and not left_assigned:
            prev = "LEFT"
        prev_labels_by_index[hand_index] = prev
        return prev

    # 2) Geometric fallback from wrist x + camera mirror config.
    geom = _fallback_label_from_geometry(kp_norm, HANDEDNESS_FALLBACK_IMAGE_MIRRORED)
    if geom == "LEFT" and left_assigned and not right_assigned:
        geom = "RIGHT"
    elif geom == "RIGHT" and right_assigned and not left_assigned:
        geom = "LEFT"
    prev_labels_by_index[hand_index] = geom
    return geom


def _get_mediapipe_hands():
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Offline mode requires MediaPipe. Install it in the active environment, e.g. "
            "`pip install mediapipe==0.10.14`."
        ) from exc

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return mp.solutions.hands

    try:
        from mediapipe.python.solutions import hands as mp_hands  # type: ignore

        return mp_hands
    except Exception as exc:
        raise RuntimeError(
            "Installed MediaPipe package does not expose Hands API (`mediapipe.solutions.hands`). "
            "Reinstall a full MediaPipe build, e.g. `pip install --force-reinstall mediapipe==0.10.14`."
        ) from exc


def _make_runtime(
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    work_region,
    drop_region,
    done_cooldown_frames: int = MOTION_FSM_DONE_COOLDOWN_FRAMES,
    rearm_clear_frames: int = MOTION_FSM_DONE_COOLDOWN_FRAMES,
    operation_mode: str = "dismantle",
    motion_fsm_version: str = MOTION_FSM_VERSION,
    stage_descriptors: List[dict] | None = None,
    use_depth_triggers: bool = False,
    depth_fallback_rgb: bool = True,
    use_motion_triggers: bool = False,
):
    sync = SyncManager(max_time_diff=SYNC_MAX_TIME_DIFF, preferred_hand=dominant_hand)
    selector = ActiveHandSelector(
        policy=active_hand_policy,
        dominant_hand=dominant_hand,
        weight_dominant=ACTIVE_HAND_WEIGHT_DOMINANT,
        weight_velocity=ACTIVE_HAND_WEIGHT_VELOCITY,
        weight_tool=ACTIVE_HAND_WEIGHT_TOOL,
        velocity_norm_scale=ACTIVE_HAND_VELOCITY_NORM_SCALE,
        tool_distance_norm=ACTIVE_HAND_TOOL_DISTANCE_NORM,
        switch_score_margin=ACTIVE_HAND_SWITCH_SCORE_MARGIN,
        switch_confirm_frames=ACTIVE_HAND_SWITCH_CONFIRM_FRAMES,
    )
    if str(motion_fsm_version).lower() == "v2":
        policy = AssemblyPolicy() if str(operation_mode).lower() == "assembly" else DismantlePolicy()
        motion = MotionFSMV2(
            policy=policy,
            stage_descriptors=stage_descriptors,
            work_region=work_region,
            drop_region=drop_region,
            trigger_point_mode=anchor_mode,
            warmup_frames=MOTION_FSM_WARMUP_FRAMES,
            min_consecutive_visible=MOTION_FSM_MIN_CONSECUTIVE_VISIBLE,
            start_no_activity_frames=MOTION_FSM_START_NO_ACTIVITY_FRAMES,
            start_recent_hit_window=MOTION_FSM_START_RECENT_HIT_WINDOW,
            start_recent_hit_count=MOTION_FSM_START_RECENT_HIT_COUNT,
            require_anchor_before_completion=MOTION_FSM_REQUIRE_ANCHOR_BEFORE_COMPLETION,
            anchor_startup_grace_frames=MOTION_FSM_ANCHOR_STARTUP_GRACE_FRAMES,
            strict_motion_hits_only=MOTION_FSM_STRICT_MOTION_HITS_ONLY,
            min_consecutive_completion_hits=MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS,
            completion_confirm_frames=max(1, min(3, MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS)),
            min_recording_frames=MOTION_FSM_MIN_RECORDING_FRAMES,
            max_hand_jump=MOTION_FSM_MAX_HAND_JUMP,
            activity_v_min=MOTION_FSM_ACTIVITY_V_MIN,
            occlusion_tolerance_frames=MOTION_FSM_OCCLUSION_TOLERANCE_FRAMES,
            cooldown_frames=done_cooldown_frames,
            clear_rearm_frames=max(0, int(rearm_clear_frames)),
            use_depth_triggers=use_depth_triggers,
            depth_fallback_rgb=depth_fallback_rgb,
            use_motion_triggers=use_motion_triggers,
        )
    else:
        motion = MotionFSM(
            work_region=work_region,
            drop_region=drop_region,
            operation_mode=operation_mode,
            trigger_point_mode=anchor_mode,
            warmup_frames=MOTION_FSM_WARMUP_FRAMES,
            min_consecutive_visible=MOTION_FSM_MIN_CONSECUTIVE_VISIBLE,
            start_no_activity_frames=MOTION_FSM_START_NO_ACTIVITY_FRAMES,
            min_consecutive_drop_hits=MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS,
            min_recording_frames=MOTION_FSM_MIN_RECORDING_FRAMES,
            max_hand_jump=MOTION_FSM_MAX_HAND_JUMP,
            activity_v_min=MOTION_FSM_ACTIVITY_V_MIN,
            pause_tolerance_frames=MOTION_FSM_PAUSE_TOLERANCE_FRAMES,
            occlusion_tolerance_frames=MOTION_FSM_OCCLUSION_TOLERANCE_FRAMES,
            done_cooldown_frames=done_cooldown_frames,
            drop_window_frames=MOTION_FSM_DROP_WINDOW_FRAMES,
            rearm_clear_frames=rearm_clear_frames,
        )
    builder = TrajectoryBuilder(
        dominant_hand=dominant_hand,
        anchor_mode=anchor_mode,
        tool_hand_distance_thresh=TOOL_HAND_DISTANCE_THRESHOLD,
    )
    return sync, selector, motion, builder


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    # Ray casting (non-self-intersecting polygon expected).
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_segment_distance(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = x - ax, y - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return float(((x - ax) ** 2 + (y - ay) ** 2) ** 0.5)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    px, py = ax + t * vx, ay + t * vy
    return float(((x - px) ** 2 + (y - py) ** 2) ** 0.5)


def _point_polygon_distance(x: float, y: float, polygon: list) -> float:
    if not polygon:
        return float("inf")
    if _point_in_polygon(x, y, polygon):
        return 0.0
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        ax, ay = float(polygon[i][0]), float(polygon[i][1])
        bx, by = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        best = min(best, _point_segment_distance(x, y, ax, ay, bx, by))
    return float(best)


def _resolve_hand_points_norm(hand_block: dict):
    pts = []
    wrist = hand_block.get("wrist")
    if wrist is not None:
        pts.append((float(wrist[0]), float(wrist[1])))
    k = hand_block.get("keypoints")
    if k is not None and len(k) > 5:
        idx = (float(k[5][0]), float(k[5][1]))
        pts.append(idx)
        mid = (float((k[0][0] + k[5][0]) * 0.5), float((k[0][1] + k[5][1]) * 0.5))
        pts.append(mid)
    # de-dup
    out = []
    seen = set()
    for p in pts:
        key = (round(p[0], 5), round(p[1], 5))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolve_hand_points_px(keypoints_px):
    if keypoints_px is None or len(keypoints_px) == 0:
        return []
    pts = []
    pts.append((float(keypoints_px[0][0]), float(keypoints_px[0][1])))
    if len(keypoints_px) > 5:
        pts.append((float(keypoints_px[5][0]), float(keypoints_px[5][1])))
        pts.append(
            (
                float((keypoints_px[0][0] + keypoints_px[5][0]) * 0.5),
                float((keypoints_px[0][1] + keypoints_px[5][1]) * 0.5),
            )
        )
    out = []
    seen = set()
    for p in pts:
        key = (round(p[0], 3), round(p[1], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _annotate_region_hits(
    *,
    frame_data: dict,
    keypoints_px_by_label: dict,
    selected_label: str | None,
    region_registry: RegionRegistry | None,
    detected_regions: dict | None,
    operation_mode: str,
    trigger_point_mode: str,
) -> None:
    if region_registry is None:
        return

    hits_by_hand = {}
    any_regions = set()
    any_roles = set()
    for label in ("LEFT", "RIGHT"):
        hb = (frame_data.get("hands") or {}).get(label.lower()) or {}
        if not hb.get("visible", False):
            hits_by_hand[label] = {"regions": [], "roles": []}
            continue

        p_norms = _resolve_hand_points_norm(hb)
        p_pxs = _resolve_hand_points_px(keypoints_px_by_label.get(label))
        regions = []
        roles = []
        for rid, region in region_registry.regions.items():
            use_frame = region.coordinate_frame
            use_polygon = region.polygon
            if detected_regions is not None and rid in detected_regions:
                det = detected_regions[rid]
                if det.coordinate_frame == "image_px" and det.polygon_px is not None:
                    use_frame = "image_px"
                    use_polygon = det.polygon_px
                elif det.coordinate_frame == "workspace_norm" and det.polygon_norm is not None:
                    use_frame = "workspace_norm"
                    use_polygon = det.polygon_norm
            hit = False
            if use_frame == "workspace_norm" and p_norms:
                for p in p_norms:
                    if _point_in_polygon(p[0], p[1], use_polygon):
                        hit = True
                        break
            elif use_frame == "image_px" and p_pxs:
                for p in p_pxs:
                    if _point_in_polygon(p[0], p[1], use_polygon):
                        hit = True
                        break
            if not hit:
                continue
            regions.append(rid)
            for role in region.roles:
                if region_registry.resolve_role_enabled(rid, operation_mode, role):
                    roles.append(role)
                    any_roles.add(role)
            any_regions.add(rid)
        hits_by_hand[label] = {"regions": sorted(set(regions)), "roles": sorted(set(roles))}

    active = {"regions": [], "roles": []}
    if selected_label in {"LEFT", "RIGHT"}:
        active = hits_by_hand.get(selected_label, {"regions": [], "roles": []})

    frame_data["region_hits"] = {
        "mode": str(operation_mode).lower(),
        "active_label": selected_label,
        "hands": hits_by_hand,
        "active": active,
        "any": {"regions": sorted(any_regions), "roles": sorted(any_roles)},
    }


def _new_builder(dominant_hand: str, anchor_mode: str) -> TrajectoryBuilder:
    return TrajectoryBuilder(
        dominant_hand=dominant_hand,
        anchor_mode=anchor_mode,
        tool_hand_distance_thresh=TOOL_HAND_DISTANCE_THRESHOLD,
    )


def _resolve_event_hand_scope(motion, stage_descriptors: List[dict] | None) -> str:
    scope = "any"
    try:
        descriptors = list(stage_descriptors or [])
        if not descriptors:
            return scope
        idx = int(getattr(motion, "current_stage_index", 0))
        idx = max(0, min(idx, len(descriptors) - 1))
        desc = descriptors[idx] or {}
        scope = str(desc.get("hand_scope", "any")).strip().lower()
        if scope not in {"any", "active"}:
            scope = "any"
    except Exception:
        scope = "any"
    return scope


def _resolve_motion_allowed_regions(
    motion,
    stage_descriptors: List[dict] | None,
    operation_mode: str,
    phase: str,
) -> set[str]:
    if not stage_descriptors or motion is None:
        return set()
    try:
        descriptors = list(stage_descriptors or [])
        if not descriptors:
            return set()
        idx = int(getattr(motion, "current_stage_index", 0))
        idx = max(0, min(idx, len(descriptors) - 1))
        desc = descriptors[idx] or {}
    except Exception:
        return set()

    descs_for_phase = [desc]
    group_id = str((desc or {}).get("unordered_group_id", "")).strip()
    if group_id and hasattr(motion, "_unordered_remaining_descs"):
        try:
            remaining_descs = list(getattr(motion, "_unordered_remaining_descs")(group_id) or [])
            if remaining_descs:
                descs_for_phase = remaining_descs
        except Exception:
            pass

    allowed = set()
    op = str(operation_mode).lower()
    phase_key = str(phase).lower()
    if op == "assembly":
        for candidate_desc in descs_for_phase:
            if phase_key == "start":
                allowed.update(candidate_desc.get("start_source_regions", []) or [])
                allowed.update(candidate_desc.get("start_tool_regions", []) or [])
            else:
                mode = str(candidate_desc.get("completion_mode", "next_stage_pick")).lower()
                if mode == "next_stage_pick":
                    allowed.update(candidate_desc.get("next_stage_source_regions", []) or [])
                else:
                    allowed.update(candidate_desc.get("completion_regions", []) or [])
    else:
        for candidate_desc in descs_for_phase:
            if phase_key == "start":
                allowed.update(candidate_desc.get("start_tool_regions", []) or [])
                allowed.update(candidate_desc.get("start_source_regions", []) or [])
            else:
                allowed.update(candidate_desc.get("completion_regions", []) or [])
    return {str(x) for x in allowed if str(x)}


def _depth_events_to_hits(events: list[dict]) -> dict:
    regions = sorted({e.get("region_id") for e in events if e.get("region_id")})
    roles = sorted({r for e in events for r in e.get("roles", [])})
    return {
        "regions": regions,
        "roles": roles,
    }


def _init_depth_stats() -> dict:
    return {"regions": {}, "frames": 0, "trigger_hits": 0}


def _update_depth_stats(stats: dict, results: dict) -> None:
    if not results:
        return
    stats["frames"] = int(stats.get("frames", 0)) + 1
    for rid, res in results.items():
        entry = stats["regions"].setdefault(
            rid,
            {"count": 0, "sum_delta": 0.0, "min_delta": None, "max_delta": None, "active_frames": 0},
        )
        delta = float(res.delta_mm)
        entry["count"] += 1
        entry["sum_delta"] += delta
        entry["min_delta"] = delta if entry["min_delta"] is None else min(entry["min_delta"], delta)
        entry["max_delta"] = delta if entry["max_delta"] is None else max(entry["max_delta"], delta)
        if bool(res.active):
            entry["active_frames"] += 1
            stats["trigger_hits"] = int(stats.get("trigger_hits", 0)) + 1


def _finalize_depth_stats(stats: dict) -> dict:
    out = {"frames": int(stats.get("frames", 0)), "trigger_hits": int(stats.get("trigger_hits", 0)), "regions": {}}
    for rid, entry in (stats.get("regions") or {}).items():
        count = int(entry.get("count", 0))
        mean = float(entry.get("sum_delta", 0.0) / count) if count else 0.0
        out["regions"][rid] = {
            "count": count,
            "mean_delta": mean,
            "min_delta": entry.get("min_delta"),
            "max_delta": entry.get("max_delta"),
            "active_frames": int(entry.get("active_frames", 0)),
        }
    return out


def _init_color_stats() -> dict:
    return {"regions": {}, "frames": 0}


def _update_color_stats(stats: dict, results: dict) -> None:
    if not results:
        return
    stats["frames"] = int(stats.get("frames", 0)) + 1
    for rid, res in results.items():
        entry = stats["regions"].setdefault(
            rid,
            {"count": 0, "sum_delta": 0.0, "sum_score": 0.0, "active_frames": 0},
        )
        entry["count"] += 1
        entry["sum_delta"] += float(res.delta)
        entry["sum_score"] += float(res.score)
        if bool(res.active):
            entry["active_frames"] += 1


def _finalize_color_stats(stats: dict) -> dict:
    out = {"frames": int(stats.get("frames", 0)), "regions": {}}
    for rid, entry in (stats.get("regions") or {}).items():
        count = int(entry.get("count", 0))
        mean_delta = float(entry.get("sum_delta", 0.0) / count) if count else 0.0
        mean_score = float(entry.get("sum_score", 0.0) / count) if count else 0.0
        out["regions"][rid] = {
            "count": count,
            "mean_delta": mean_delta,
            "mean_score": mean_score,
            "active_frames": int(entry.get("active_frames", 0)),
        }
    return out


def _init_motion_stats() -> dict:
    return {"regions": {}, "frames": 0, "trigger_hits": 0}


def _update_motion_stats(stats: dict, results: dict) -> None:
    if not results:
        return
    stats["frames"] = int(stats.get("frames", 0)) + 1
    for rid, res in results.items():
        entry = stats["regions"].setdefault(
            rid,
            {"count": 0, "sum_score": 0.0, "max_score": 0.0, "active_frames": 0},
        )
        entry["count"] += 1
        entry["sum_score"] += float(res.motion_score)
        entry["max_score"] = max(float(entry.get("max_score", 0.0)), float(res.motion_score))
        if bool(res.active):
            entry["active_frames"] += 1
            stats["trigger_hits"] = int(stats.get("trigger_hits", 0)) + 1


def _finalize_motion_stats(stats: dict) -> dict:
    out = {"frames": int(stats.get("frames", 0)), "trigger_hits": int(stats.get("trigger_hits", 0)), "regions": {}}
    for rid, entry in (stats.get("regions") or {}).items():
        count = int(entry.get("count", 0))
        mean_score = float(entry.get("sum_score", 0.0) / count) if count else 0.0
        out["regions"][rid] = {
            "count": count,
            "mean_score": mean_score,
            "max_score": float(entry.get("max_score", 0.0)),
            "active_frames": int(entry.get("active_frames", 0)),
        }
    return out


def _build_motion_region_hits(
    *,
    region_registry: RegionRegistry | None,
    operation_mode: str,
    motion_results: dict | None,
) -> dict:
    if not motion_results or region_registry is None:
        return {}
    regions = []
    roles = set()
    for rid, res in motion_results.items():
        if not bool(res.active):
            continue
        region = region_registry.regions.get(rid)
        if region is None:
            continue
        enabled_roles = [r for r in region.roles if region_registry.resolve_role_enabled(rid, operation_mode, r)]
        if not any(r in {"part_pick", "part_drop", "tool_pick", "tool_drop"} for r in enabled_roles):
            continue
        regions.append(rid)
        roles.update(enabled_roles)
    hits = {"regions": sorted(set(regions)), "roles": sorted(roles)}
    return {"active": hits, "any": hits}


class ROIInteractionGate:
    def __init__(
        self,
        *,
        region_registry: RegionRegistry | None,
        operation_mode: str,
        fallback_score_threshold: float = MOTION_HAND_FALLBACK_SCORE_THRESHOLD,
        near_margin_norm: float = MOTION_INTERACTION_NEAR_MARGIN_NORM,
        approach_delta_norm: float = MOTION_INTERACTION_APPROACH_DELTA_NORM,
        history_frames: int = 6,
    ):
        self.region_registry = region_registry
        self.operation_mode = str(operation_mode)
        self.fallback_score_threshold = float(max(0.0, fallback_score_threshold))
        self.near_margin_norm = float(max(0.0, near_margin_norm))
        self.approach_delta_norm = float(max(0.0, approach_delta_norm))
        self.history_frames = int(max(2, history_frames))
        self._distance_history: dict[str, deque] = {}

    def filter(self, motion_results: dict | None, frame_data: dict) -> dict | None:
        if not motion_results:
            return motion_results
        raw_hits = frame_data.get("region_hits") or {}
        hand_regions = set((raw_hits.get("any") or {}).get("regions", []))
        hand_points = self._hand_points_norm(frame_data)
        for rid, res in motion_results.items():
            if not bool(getattr(res, "active", False)):
                continue
            if str(rid) in hand_regions:
                self._update_distance(str(rid), 0.0)
                continue
            score = float(getattr(res, "mean_history", getattr(res, "motion_score", 0.0)) or 0.0)
            if score < self.fallback_score_threshold:
                res.active = False
                continue
            distance = self._min_distance_to_region_norm(str(rid), hand_points)
            self._update_distance(str(rid), distance)
            if distance <= self.near_margin_norm:
                continue
            if self._is_approaching(str(rid)):
                continue
            res.active = False
        return motion_results

    def _hand_points_norm(self, frame_data: dict) -> list[tuple[float, float]]:
        points = []
        for label in ("left", "right"):
            hb = (frame_data.get("hands") or {}).get(label) or {}
            if not hb.get("visible", False):
                continue
            points.extend(_resolve_hand_points_norm(hb))
        return points

    def _min_distance_to_region_norm(self, rid: str, hand_points: list[tuple[float, float]]) -> float:
        if not hand_points or self.region_registry is None:
            return float("inf")
        region = self.region_registry.regions.get(str(rid))
        if region is None or region.coordinate_frame != "workspace_norm":
            return float("inf")
        return min(_point_polygon_distance(float(x), float(y), region.polygon) for x, y in hand_points)

    def _update_distance(self, rid: str, distance: float) -> None:
        hist = self._distance_history.setdefault(str(rid), deque(maxlen=self.history_frames))
        hist.append(float(distance))

    def _is_approaching(self, rid: str) -> bool:
        hist = self._distance_history.get(str(rid))
        if hist is None or len(hist) < 3:
            return False
        start = float(hist[0])
        end = float(hist[-1])
        if not np.isfinite(start) or not np.isfinite(end):
            return False
        return bool((start - end) >= self.approach_delta_norm)


def _hand_confirmed_motion_results(
    motion_results: dict | None,
    frame_data: dict,
    *,
    fallback_score_threshold: float = MOTION_HAND_FALLBACK_SCORE_THRESHOLD,
) -> dict | None:
    if not motion_results:
        return motion_results
    raw_hits = frame_data.get("region_hits") or {}
    hand_regions = set((raw_hits.get("any") or {}).get("regions", []))
    for rid, res in motion_results.items():
        if not bool(getattr(res, "active", False)):
            continue
        if str(rid) in hand_regions:
            continue
        res.active = False
    return motion_results


def _apply_depth_triggers(
    *,
    depth_trigger: DepthTriggerExtractor,
    depth_mm: np.ndarray,
    frame_id: int,
    timestamp: float,
    region_registry: RegionRegistry | None,
    operation_mode: str,
) -> tuple[list[dict], dict]:
    results = depth_trigger.update(depth_mm)
    if not results:
        return [], {}
    events: list[dict] = []
    for rid, res in results.items():
        if not res.active:
            continue
        roles = []
        if region_registry is not None and rid in region_registry.regions:
            roles = [
                r
                for r in region_registry.regions[rid].roles
                if region_registry.resolve_role_enabled(rid, operation_mode, r)
            ]
        evt_base = {
            "t": float(timestamp),
            "frame_id": int(frame_id),
            "region_id": str(rid),
            "roles": roles,
            "source": "depth",
            "baseline_mm": float(res.baseline_mm),
            "current_mm": float(res.current_mm),
            "delta_mm": float(res.delta_mm),
            "dwell_frames": int(res.dwell_frames),
        }
        if any(r in {"part_pick", "tool_pick"} for r in roles):
            events.append({**evt_base, "type": "depth_pick_candidate"})
        if any(r in {"part_drop", "tool_drop"} for r in roles):
            events.append({**evt_base, "type": "depth_drop_candidate"})
    return events, results


def _build_fused_hits(
    *,
    region_registry: RegionRegistry | None,
    operation_mode: str,
    rgb_region_hits: dict,
    depth_results: dict | None,
    color_results: dict | None,
    threshold: float,
    w_color: float,
    w_depth: float,
    w_rgb: float,
) -> tuple[dict, int]:
    if region_registry is None:
        return {}, 0
    any_rgb_regions = set((rgb_region_hits.get("any") or {}).get("regions", []))
    fused_regions = []
    fused_roles = set()
    fused_hits = 0
    for rid, region in region_registry.regions.items():
        roles = [r for r in region.roles if region_registry.resolve_role_enabled(rid, operation_mode, r)]
        if not any(r in {"part_pick", "part_drop", "tool_pick", "tool_drop"} for r in roles):
            continue
        color_score = 0.0
        if color_results and rid in color_results:
            color_score = float(color_results[rid].score)
        depth_active = 0.0
        if depth_results and rid in depth_results:
            depth_active = 1.0 if bool(depth_results[rid].active) else 0.0
        rgb_active = 1.0 if rid in any_rgb_regions else 0.0
        fused_score = (w_color * color_score) + (w_depth * depth_active) + (w_rgb * rgb_active)
        if fused_score >= float(threshold):
            fused_regions.append(rid)
            fused_roles.update(roles)
            fused_hits += 1
    return {"regions": sorted(fused_regions), "roles": sorted(fused_roles)}, fused_hits


def _render_frame(
    frame,
    *,
    pre: FramePreprocessor,
    motion_state: str,
    frame_id: int,
    timestamp: float,
    show: bool,
    delay_ms: int,
    anchor_mode: str,
    keypoints_px,
    work_region,
    drop_region,
    region_registry: RegionRegistry | None,
    depth_events: list | None,
    depth_trigger_results: dict | None,
    color_trigger_results: dict | None,
    motion_trigger_results: dict | None,
    frame_data: dict | None = None,
    show_depth_roi_overlay: bool = True,
    show_regions: bool = True,
    virtual_projector: VirtualProjectionGuide | None = None,
    current_stage_desc: dict | None = None,
    operation_mode: str = "dismantle",
    overlay_writer_state: dict | None = None,
) -> bool:
    should_write = bool(overlay_writer_state and overlay_writer_state.get("path"))
    if not show and not should_write:
        return False

    vis = frame.copy()
    if virtual_projector is not None:
        virtual_projector.draw(
            vis,
            pre=pre,
            region_registry=region_registry,
            stage_descriptor=current_stage_desc,
            operation_mode=operation_mode,
        )

    if show_regions:
        if region_registry is not None:
            draw_region_registry_overlay(vis, pre, region_registry)
        else:
            draw_workspace_and_drop_overlay(vis, pre, work_region=work_region, drop_region=drop_region)
        if keypoints_px is not None:
            draw_anchor_overlay(vis, keypoints_px, anchor_mode)

    draw_fsm_overlay(vis, motion_state, frame_id, timestamp)
    draw_detection_overlay(vis, frame_data)
    draw_depth_trigger_overlay(vis, depth_events)
    if show_depth_roi_overlay:
        draw_depth_roi_overlay(vis, pre, region_registry, depth_trigger_results)
        draw_color_roi_overlay(vis, pre, region_registry, color_trigger_results)
        draw_motion_roi_overlay(vis, pre, region_registry, motion_trigger_results)
    if should_write:
        _write_overlay_video_frame(overlay_writer_state, vis)
    if show:
        cv2.imshow("Video Pipeline", vis)
        return (cv2.waitKey(delay_ms) & 0xFF) == 27
    return False


def _write_overlay_video_frame(writer_state: dict, frame) -> None:
    writer = writer_state.get("writer")
    if writer is None:
        path = Path(writer_state["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        fps = float(writer_state.get("fps", 30.0) or 30.0)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open overlay video writer: {path}")
        writer_state["writer"] = writer
    writer.write(frame)


def _close_overlay_video_writer(writer_state: dict | None) -> None:
    if not writer_state:
        return
    writer = writer_state.get("writer")
    if writer is not None:
        writer.release()
        writer_state["writer"] = None


def _append_segment_if_valid(
    segments: List[dict],
    builder: TrajectoryBuilder,
    fsm_transitions: List[dict],
    tool_conf_timeline: List[dict],
    raw_oob_count: int,
    min_segment_points: int,
    stage_name: str | None = None,
    start_frame: int | None = None,
    start_time: float | None = None,
    end_frame: int | None = None,
    end_time: float | None = None,
    closure_type: str | None = None,
    event_timeline: List[dict] | None = None,
    contract_rejections: List[dict] | None = None,
    rejected_trigger_reasons: List[dict] | None = None,
) -> bool:
    traj = builder.get_trajectory()
    if len(traj) < min_segment_points:
        logger.info(
            "Dropped segment: points=%d below min_segment_points=%d",
            len(traj),
            int(min_segment_points),
        )
        return False
    segments.append(
        {
            "trajectory_raw": traj,
            "fsm_transitions": list(fsm_transitions),
            "tool_conf_timeline": list(tool_conf_timeline),
            "raw_oob_count": raw_oob_count,
            "stage_name": stage_name,
            "start_frame": start_frame,
            "start_time": start_time,
            "end_frame": end_frame,
            "end_time": end_time,
            "duration_s": (float(end_time) - float(start_time)) if (start_time is not None and end_time is not None) else None,
            "closure_type": closure_type,
            "event_timeline": list(event_timeline or []),
            "contract_rejections": list(contract_rejections or []),
            "rejected_trigger_reasons": list(rejected_trigger_reasons or []),
        }
    )
    return True


def _clear_active_segment_state():
    return {
        "fsm_transitions": [],
        "tool_conf_timeline": [],
        "event_timeline": [],
        "contract_rejections": [],
        "trigger_reason": None,
        "completion_reason": None,
        "rejected_trigger_reasons": [],
        "raw_oob_count": 0,
        "bimanual_stats": _init_bimanual_stats(),
        "depth_stats": _init_depth_stats(),
        "color_stats": _init_color_stats(),
        "motion_stats": _init_motion_stats(),
        "fused_trigger_hits": 0,
        "motion_trigger_hits": 0,
        "active": False,
        "active_stage_name": None,
        "active_start_frame": None,
        "active_start_time": None,
        "no_hand_streak": 0,
    }


def _init_bimanual_stats() -> dict:
    return {
        "recording_frames": 0,
        "active_left_frames": 0,
        "active_right_frames": 0,
        "both_visible_frames": 0,
        "hand_switch_count": 0,
        "total_recording_time_s": 0.0,
        "overlap_time_s": 0.0,
        "_last_active_label": None,
        "_last_t": None,
    }


def _update_bimanual_stats(stats: dict, frame_data: dict, timestamp: float) -> None:
    if not stats:
        return
    if frame_data.get("hand", {}).get("visible") is not True:
        stats["_last_t"] = timestamp
        return

    dt = 1.0 / 30.0
    last_t = stats.get("_last_t")
    if last_t is not None and timestamp > last_t:
        dt = float(timestamp - last_t)
    stats["_last_t"] = timestamp

    stats["recording_frames"] += 1
    stats["total_recording_time_s"] += dt

    hands = frame_data.get("hands") or {}
    active_label = hands.get("active_label")
    if active_label == "LEFT":
        stats["active_left_frames"] += 1
    elif active_label == "RIGHT":
        stats["active_right_frames"] += 1

    if int(hands.get("visible_count", 0)) >= 2:
        stats["both_visible_frames"] += 1
        stats["overlap_time_s"] += dt

    prev_label = stats.get("_last_active_label")
    if prev_label is not None and active_label is not None and active_label != prev_label:
        stats["hand_switch_count"] += 1
    if active_label is not None:
        stats["_last_active_label"] = active_label


def _finalize_bimanual_stats(stats: dict) -> dict:
    total_active = int(stats.get("active_left_frames", 0)) + int(stats.get("active_right_frames", 0))
    total_time = float(stats.get("total_recording_time_s", 0.0))
    overlap_time = float(stats.get("overlap_time_s", 0.0))
    out = {
        "active_left_ratio": (float(stats.get("active_left_frames", 0)) / total_active) if total_active > 0 else 0.0,
        "active_right_ratio": (float(stats.get("active_right_frames", 0)) / total_active) if total_active > 0 else 0.0,
        "hand_switch_count": int(stats.get("hand_switch_count", 0)),
        "overlap_time_s": overlap_time,
        "overlap_ratio": (overlap_time / total_time) if total_time > 0 else 0.0,
        "recording_frames": int(stats.get("recording_frames", 0)),
    }
    return out


def _offline_frame_packets(
    video_path: Path,
    pre: FramePreprocessor,
    dominant_hand: str,
    depth_mm: np.ndarray | None = None,
) -> Generator[Dict, None, None]:
    mp_hands = _get_mediapipe_hands()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_id = 0
    prev_labels_by_index = {}
    prev_wrist_by_label = {"LEFT": None, "RIGHT": None}
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_NUM_HANDS,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        ) as hands_model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                if depth_mm is not None and (frame_id - 1) >= depth_mm.shape[0]:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp <= 0:
                    timestamp = frame_id / 30.0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands_model.process(rgb)

                hand_data = None
                keypoints_px = None
                keypoints_px_by_label = {}
                if result.multi_hand_landmarks:
                    h, w = frame.shape[:2]
                    left_data = None
                    right_data = None
                    left_kp_px = None
                    right_kp_px = None
                    handedness_list = getattr(result, "multi_handedness", None) or []
                    for i, hand_lm in enumerate(result.multi_hand_landmarks):
                        lm = hand_lm.landmark
                        kp_px = np.array([[p.x * w, p.y * h] for p in lm], dtype=float)
                        kp_norm = pre.warp_keypoints(kp_px)
                        label = None
                        if i < len(handedness_list):
                            cls = handedness_list[i].classification
                            if cls:
                                label = str(cls[0].label).upper()
                        label = _resolve_offline_hand_label(
                            raw_label=label,
                            hand_index=i,
                            kp_norm=kp_norm,
                            prev_labels_by_index=prev_labels_by_index,
                            left_assigned=left_data is not None,
                            right_assigned=right_data is not None,
                        )
                        wrist = (float(kp_norm[0][0]), float(kp_norm[0][1]))
                        prev_wrist = prev_wrist_by_label.get(label)
                        velocity = 0.0
                        if prev_wrist is not None:
                            dx = wrist[0] - float(prev_wrist[0])
                            dy = wrist[1] - float(prev_wrist[1])
                            velocity = float((dx * dx + dy * dy) ** 0.5)
                        prev_wrist_by_label[label] = wrist
                        payload = {
                            "keypoints": kp_norm,
                            "keypoints_px": kp_px,
                            "velocity": velocity,
                            "wrist_px": (float(kp_px[0][0]), float(kp_px[0][1])),
                        }
                        if label == "LEFT":
                            left_data = payload
                            left_kp_px = kp_px
                        else:
                            right_data = payload
                            right_kp_px = kp_px
                    hand_data = {"left": left_data, "right": right_data}
                    preferred = dominant_hand.upper()
                    if preferred == "LEFT":
                        keypoints_px = left_kp_px if left_kp_px is not None else right_kp_px
                    else:
                        keypoints_px = right_kp_px if right_kp_px is not None else left_kp_px
                    keypoints_px_by_label = {"LEFT": left_kp_px, "RIGHT": right_kp_px}

                yield {
                    "frame": frame,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "hand_data": hand_data,
                    "keypoints_px": keypoints_px,
                    "keypoints_px_by_label": keypoints_px_by_label,
                    "depth_mm": depth_mm[frame_id - 1] if depth_mm is not None else None,
                }
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _depthai_frame_packets(
    video_path: Path,
    pre: FramePreprocessor,
    dominant_hand: str,
) -> Generator[Dict, None, None]:
    from dependencies.depthai_hand_tracker.depthai_hand_tracker.HandTracker import HandTracker

    tracker = HandTracker(input_src=str(video_path), use_gesture=False)
    frame_id = 0
    prev_wrist_by_label = {"LEFT": None, "RIGHT": None}
    try:
        while True:
            result = tracker.next_frame()
            if result is None:
                break

            if len(result) == 3:
                frame, hands, _ = result
            elif len(result) == 2:
                frame, hands = result
            else:
                raise ValueError(f"Unexpected tracker output length: {len(result)}")

            if frame is None:
                break

            frame_id += 1
            timestamp = frame_id / 30.0

            hand_data = None
            keypoints_px = None
            keypoints_px_by_label = {}
            if hands:
                left_data = None
                right_data = None
                left_kp_px = None
                right_kp_px = None
                for hand in hands:
                    kp_px = np.array(hand.landmarks, dtype=float)
                    kp_norm = pre.warp_keypoints(kp_px)
                    label = str(getattr(hand, "label", "")).upper()
                    if HANDEDNESS_SWAP_LABELS:
                        if label == "LEFT":
                            label = "RIGHT"
                        elif label == "RIGHT":
                            label = "LEFT"
                    wrist = (float(kp_norm[0][0]), float(kp_norm[0][1]))
                    prev_wrist = prev_wrist_by_label.get(label)
                    velocity = 0.0
                    if prev_wrist is not None:
                        dx = wrist[0] - float(prev_wrist[0])
                        dy = wrist[1] - float(prev_wrist[1])
                        velocity = float((dx * dx + dy * dy) ** 0.5)
                    prev_wrist_by_label[label] = wrist
                    payload = {
                        "keypoints": kp_norm,
                        "velocity": velocity,
                        "wrist_px": (float(kp_px[0][0]), float(kp_px[0][1])),
                    }
                    if label == "LEFT":
                        left_data = payload
                        left_kp_px = kp_px
                    elif label == "RIGHT":
                        right_data = payload
                        right_kp_px = kp_px
                    elif right_data is None:
                        right_data = payload
                        right_kp_px = kp_px
                    else:
                        left_data = payload
                        left_kp_px = kp_px
                hand_data = {"left": left_data, "right": right_data}
                preferred = dominant_hand.upper()
                if preferred == "LEFT":
                    keypoints_px = left_kp_px if left_kp_px is not None else right_kp_px
                else:
                    keypoints_px = right_kp_px if right_kp_px is not None else left_kp_px
                keypoints_px_by_label = {"LEFT": left_kp_px, "RIGHT": right_kp_px}

            yield {
                "frame": frame,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "hand_data": hand_data,
                "keypoints_px": keypoints_px,
                "keypoints_px_by_label": keypoints_px_by_label,
            }
    finally:
        tracker.exit()
        cv2.destroyAllWindows()


def _extract_stage_from_packets(
    packets,
    *,
    pre: FramePreprocessor,
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    show: bool,
    delay_ms: int,
    work_region,
    drop_region,
    operation_mode: str = "dismantle",
    region_registry: RegionRegistry | None = None,
    region_detector: RegionDetector | None = None,
    motion_fsm_version: str = MOTION_FSM_VERSION,
    stage_descriptors: List[dict] | None = None,
    stage_name: str | None = None,
    depth_trigger: DepthTriggerExtractor | None = None,
    depth_fallback_rgb: bool = True,
    show_depth_roi_overlay: bool = True,
    color_trigger: ColorChangeTrigger | None = None,
    motion_trigger: ROIMotionTrigger | None = None,
    motion_interaction_gate: ROIInteractionGate | None = None,
    motion_log_interval_frames: int = 30,
    fused_trigger_threshold: float = 0.65,
    fused_weight_color: float = 0.6,
    fused_weight_depth: float = 0.2,
    fused_weight_rgb: float = 0.2,
    pose_pilot_bridge: PosePilotBridge | None = None,
    pose_pilot_exporter: PosePilotExporter | None = None,
    pose_classifier: PoseClassifierManager | None = None,
    virtual_projector: VirtualProjectionGuide | None = None,
    virtual_projection_output_video: str | Path | None = None,
    virtual_projection_output_fps: float = 30.0,
) -> Tuple[List[dict], List[dict], List[dict], int, dict]:
    use_motion_triggers = bool(motion_trigger is not None)
    object_detector = _make_object_detector()
    workspace_filter = _make_workspace_filter(pre)
    sync, selector, motion, builder = _make_runtime(
        anchor_mode,
        dominant_hand,
        active_hand_policy,
        work_region,
        drop_region,
        operation_mode=operation_mode,
        motion_fsm_version=motion_fsm_version,
        stage_descriptors=stage_descriptors,
        use_depth_triggers=bool(depth_trigger is not None) and not use_motion_triggers,
        depth_fallback_rgb=depth_fallback_rgb,
        use_motion_triggers=use_motion_triggers,
    )
    if stage_descriptors and stage_name and hasattr(motion, "set_stage_index"):
        target = str(stage_name).strip()
        for idx, desc in enumerate(stage_descriptors):
            if str(desc.get("stage_name", "")).strip() == target:
                motion.set_stage_index(idx)
                break
    reference_stage_autostart = False
    if operation_mode == "dismantle" and stage_descriptors and stage_name:
        target = str(stage_name).strip()
        for desc in stage_descriptors:
            if str(desc.get("stage_name", "")).strip() == target:
                reference_stage_autostart = bool(
                    desc.get("manual_start_allowed", True)
                    and not (desc.get("start_tool_regions", []) or [])
                    and not (desc.get("start_source_regions", []) or [])
                )
                break

    fsm_transitions = []
    tool_conf_timeline = []
    event_timeline: List[dict] = []
    contract_rejections: List[dict] = []
    trigger_reason = None
    completion_reason = None
    rejected_trigger_reasons: List[dict] = []
    raw_oob_count = 0
    bimanual_stats = _init_bimanual_stats()
    depth_stats = _init_depth_stats()
    color_stats = _init_color_stats()
    motion_stats = _init_motion_stats()
    fused_trigger_hits = 0
    motion_trigger_hits = 0
    event_extractor = RegionEventExtractor(region_registry, operation_mode=operation_mode) if region_registry else None
    overlay_writer_state = (
        {"path": str(virtual_projection_output_video), "fps": float(virtual_projection_output_fps)}
        if virtual_projection_output_video
        else None
    )

    try:
        for pkt in packets:
            frame = pkt["frame"]
            frame_id = pkt["frame_id"]
            timestamp = pkt["timestamp"]
            hand_data = pkt["hand_data"]
            keypoints_px = pkt["keypoints_px"]
            keypoints_px_by_label = pkt.get("keypoints_px_by_label", {})
            depth_mm = pkt.get("depth_mm")
            current_stage_desc = _resolve_current_stage_descriptor(motion, stage_descriptors, stage_name)
            detections = object_detector.detect(frame) if object_detector is not None else None

            frame_data = sync.sync(
                hand_data=hand_data,
                tool_data=_none_tool(timestamp),
                timestamp=timestamp,
                frame_id=frame_id,
                detections=detections,
                expected_parts=current_stage_desc.get("expected_parts"),
                expected_tools=current_stage_desc.get("expected_tools"),
                expected_inventory=current_stage_desc.get("expected_inventory"),
                active_part_candidates=current_stage_desc.get("active_part_candidates"),
                active_tool_candidates=current_stage_desc.get("active_tool_candidates"),
                stage_name=current_stage_desc.get("stage_name", stage_name),
                workspace_filter=workspace_filter,
            )
            if reference_stage_autostart:
                frame_data["reference_stage_autostart"] = True
            frame_data["object_debug"] = {
                "active_part": frame_data.get("active_part"),
                "active_tool": frame_data.get("active_tool"),
                "inventory": frame_data.get("inventory", {}),
                "workspace_inventory": frame_data.get("workspace_inventory", {}),
            }
            _attach_pose_classifier_predictions(
                frame=frame,
                frame_data=frame_data,
                current_stage_desc=current_stage_desc,
                frame_id=int(frame_id),
                pose_classifier=pose_classifier,
            )
            _maybe_export_pose_pilot_candidate(
                frame=frame,
                frame_data=frame_data,
                stage_name=str(current_stage_desc.get("stage_name", stage_name)),
                frame_id=int(frame_id),
                timestamp=float(timestamp),
                pose_pilot_bridge=pose_pilot_bridge,
                pose_pilot_exporter=pose_pilot_exporter,
            )
            _log_object_awareness_debug(frame_data, int(frame_id), str(current_stage_desc.get("stage_name", stage_name)))
            cached_point = builder.build_trajectory_point(frame_data)
            frame_data["anchor_valid"] = cached_point is not None
            selected_label = selector.select(frame_data)
            if selected_label is not None:
                keypoints_px = keypoints_px_by_label.get(selected_label)
            detected_regions = region_detector.detect_regions(frame) if region_detector is not None else None
            _annotate_region_hits(
                frame_data=frame_data,
                keypoints_px_by_label=keypoints_px_by_label,
                selected_label=selected_label,
                region_registry=region_registry,
                detected_regions=detected_regions,
                operation_mode=operation_mode,
                trigger_point_mode=anchor_mode,
            )
            if event_extractor is not None:
                event_hand_scope = _resolve_event_hand_scope(motion, stage_descriptors)
                event_timeline.extend(
                    event_extractor.update_with_scope(
                        frame_data,
                        frame_id=int(frame_id),
                        timestamp=float(timestamp),
                        hand_scope=event_hand_scope,
                    )
                )
                frame_data["fsm_region_hits"] = event_extractor.build_stable_region_hits(
                    frame_data, hand_scope=event_hand_scope
                )
            if depth_trigger is not None and depth_mm is not None:
                depth_events, depth_results = _apply_depth_triggers(
                    depth_trigger=depth_trigger,
                    depth_mm=depth_mm,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    region_registry=region_registry,
                    operation_mode=operation_mode,
                )
                frame_data["depth_events"] = depth_events or []
                frame_data["depth_trigger_results"] = depth_results or {}
                _update_depth_stats(depth_stats, depth_results or {})
                if depth_fallback_rgb:
                    if depth_events:
                        event_timeline.extend(depth_events)
                        frame_data["depth_region_hits"] = _depth_events_to_hits(depth_events)
                    else:
                        frame_data["depth_region_hits"] = {}
            if color_trigger is not None:
                color_results = color_trigger.update(frame)
                frame_data["color_trigger_results"] = color_results or {}
                _update_color_stats(color_stats, color_results or {})
            if motion_trigger is not None:
                motion_results = motion_trigger.update(frame)
                phase = "start" if str(motion.state) == "WAIT_TRIGGER" else "completion"
                allowed = _resolve_motion_allowed_regions(motion, stage_descriptors, operation_mode, phase)
                if allowed:
                    motion_results = {rid: res for rid, res in (motion_results or {}).items() if rid in allowed}
                if motion_interaction_gate is not None:
                    motion_results = motion_interaction_gate.filter(motion_results, frame_data)
                else:
                    motion_results = _hand_confirmed_motion_results(motion_results, frame_data)
                frame_data["motion_trigger_results"] = motion_results or {}
                _update_motion_stats(motion_stats, motion_results or {})
                motion_hits = _build_motion_region_hits(
                    region_registry=region_registry,
                    operation_mode=operation_mode,
                    motion_results=motion_results or {},
                )
                frame_data["motion_region_hits"] = motion_hits
                if motion_hits:
                    motion_trigger_hits += int(len((motion_hits.get("any") or {}).get("regions", [])))
                if motion_results and motion_log_interval_frames > 0:
                    try:
                        if int(frame_id) % int(motion_log_interval_frames) == 0:
                            for rid in sorted(motion_results.keys()):
                                res = motion_results[rid]
                                logger.info(
                                    "[Motion] ROI %s m=%.3f active=%s",
                                    rid,
                                    float(res.motion_score),
                                    bool(res.active),
                                )
                    except Exception:
                        pass
            if region_registry is not None:
                fused_hits, fused_count = _build_fused_hits(
                    region_registry=region_registry,
                    operation_mode=operation_mode,
                    rgb_region_hits=frame_data.get("fsm_region_hits") or frame_data.get("region_hits") or {},
                    depth_results=frame_data.get("depth_trigger_results"),
                    color_results=frame_data.get("color_trigger_results"),
                    threshold=fused_trigger_threshold,
                    w_color=fused_weight_color,
                    w_depth=fused_weight_depth,
                    w_rgb=fused_weight_rgb,
                )
                fused_trigger_hits += int(fused_count)
                frame_data["fused_region_hits"] = {"active": fused_hits, "any": fused_hits}

            prev = motion.state
            motion.update(frame_data)
            if hasattr(motion, "consume_contract_rejections"):
                new_rejects = getattr(motion, "consume_contract_rejections")()
                contract_rejections.extend(new_rejects)
                rejected_trigger_reasons.extend([r for r in new_rejects if str(r.get("phase", "")).startswith("start")])
            if hasattr(motion, "consume_transition_reasons"):
                for rr in getattr(motion, "consume_transition_reasons")():
                    if rr.get("kind") == "trigger":
                        trigger_reason = rr
                    elif rr.get("kind") == "completion":
                        completion_reason = rr
            if motion.state != prev:
                fsm_transitions.append({"from": prev, "to": motion.state, "t": max(0.0, timestamp)})

            if motion.state == "RECORDING":
                _update_bimanual_stats(bimanual_stats, frame_data, timestamp)
                p = cached_point
                if p is not None:
                    builder.append(p)
                    tool_conf_timeline.append({"t": max(0.0, timestamp), "tool": "NONE", "confidence": 0.0})
                    if not (0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0):
                        raw_oob_count += 1

            if _render_frame(
                frame,
                pre=pre,
                motion_state=motion.state,
                frame_id=frame_id,
                timestamp=timestamp,
                show=show,
                delay_ms=delay_ms,
                anchor_mode=anchor_mode,
                keypoints_px=keypoints_px,
                work_region=work_region,
                drop_region=drop_region,
                region_registry=region_registry,
                depth_events=frame_data.get("depth_events"),
                depth_trigger_results=frame_data.get("depth_trigger_results"),
                color_trigger_results=frame_data.get("color_trigger_results"),
                motion_trigger_results=frame_data.get("motion_trigger_results"),
                frame_data=frame_data,
                show_depth_roi_overlay=show_depth_roi_overlay,
                show_regions=True,
                virtual_projector=virtual_projector,
                current_stage_desc=current_stage_desc,
                operation_mode=operation_mode,
                overlay_writer_state=overlay_writer_state,
            ):
                break

            segment_closed = bool(getattr(motion, "consume_segment_closed", lambda: False)())
            if motion.state == "DONE" or segment_closed:
                break
    finally:
        close = getattr(packets, "close", None)
        if callable(close):
            close()
        _close_overlay_video_writer(overlay_writer_state)

    out_stats = _finalize_bimanual_stats(bimanual_stats)
    out_stats["depth_stats"] = _finalize_depth_stats(depth_stats)
    out_stats["color_trigger_stats"] = _finalize_color_stats(color_stats)
    out_stats["fused_trigger_hits"] = int(fused_trigger_hits)
    out_stats["motion_trigger_stats"] = _finalize_motion_stats(motion_stats)
    out_stats["motion_trigger_hits"] = int(motion_trigger_hits)
    if hasattr(motion, "consume_depth_fallback_count"):
        out_stats["depth_fallback_count"] = getattr(motion, "consume_depth_fallback_count")()
    if event_timeline:
        out_stats["event_timeline"] = list(event_timeline)
    if contract_rejections:
        out_stats["contract_rejections"] = list(contract_rejections)
    if trigger_reason is not None:
        out_stats["trigger_reason"] = dict(trigger_reason)
    if completion_reason is not None:
        out_stats["completion_reason"] = dict(completion_reason)
    if rejected_trigger_reasons:
        out_stats["rejected_trigger_reasons"] = list(rejected_trigger_reasons)
    return (
        builder.get_trajectory(),
        fsm_transitions,
        tool_conf_timeline,
        raw_oob_count,
        out_stats,
    )


def _extract_segments_from_packets(
    packets,
    *,
    pre: FramePreprocessor,
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    show: bool,
    delay_ms: int,
    no_hand_frames_to_split: int,
    rearm_no_hand_frames: int,
    min_segment_points: int,
    work_region,
    drop_region,
    operation_mode: str = "dismantle",
    region_registry: RegionRegistry | None = None,
    region_detector: RegionDetector | None = None,
    motion_fsm_version: str = MOTION_FSM_VERSION,
    stage_descriptors: List[dict] | None = None,
    depth_trigger: DepthTriggerExtractor | None = None,
    depth_fallback_rgb: bool = True,
    show_depth_roi_overlay: bool = True,
    color_trigger: ColorChangeTrigger | None = None,
    motion_trigger: ROIMotionTrigger | None = None,
    motion_interaction_gate: ROIInteractionGate | None = None,
    motion_log_interval_frames: int = 30,
    fused_trigger_threshold: float = 0.65,
    fused_weight_color: float = 0.6,
    fused_weight_depth: float = 0.2,
    fused_weight_rgb: float = 0.2,
    pose_pilot_bridge: PosePilotBridge | None = None,
    pose_pilot_exporter: PosePilotExporter | None = None,
    pose_classifier: PoseClassifierManager | None = None,
    virtual_projector: VirtualProjectionGuide | None = None,
    virtual_projection_output_video: str | Path | None = None,
    virtual_projection_output_fps: float = 30.0,
) -> List[dict]:
    use_motion_triggers = bool(motion_trigger is not None)
    object_detector = _make_object_detector()
    workspace_filter = _make_workspace_filter(pre)
    sync, selector, motion, builder = _make_runtime(
        anchor_mode,
        dominant_hand,
        active_hand_policy,
        work_region,
        drop_region,
        operation_mode=operation_mode,
        motion_fsm_version=motion_fsm_version,
        stage_descriptors=stage_descriptors,
        rearm_clear_frames=max(0, int(rearm_no_hand_frames)),
        use_depth_triggers=bool(depth_trigger is not None) and not use_motion_triggers,
        depth_fallback_rgb=depth_fallback_rgb,
        use_motion_triggers=use_motion_triggers,
    )

    segments: List[dict] = []
    fsm_transitions: List[dict] = []
    tool_conf_timeline: List[dict] = []
    event_timeline: List[dict] = []
    contract_rejections: List[dict] = []
    trigger_reason = None
    completion_reason = None
    rejected_trigger_reasons: List[dict] = []
    raw_oob_count = 0
    bimanual_stats = _init_bimanual_stats()
    depth_stats = _init_depth_stats()
    color_stats = _init_color_stats()
    motion_stats = _init_motion_stats()
    fused_trigger_hits = 0
    motion_trigger_hits = 0
    event_extractor = RegionEventExtractor(region_registry, operation_mode=operation_mode) if region_registry else None
    overlay_writer_state = (
        {"path": str(virtual_projection_output_video), "fps": float(virtual_projection_output_fps)}
        if virtual_projection_output_video
        else None
    )

    active = False
    active_stage_name: str | None = None
    active_start_frame: int | None = None
    active_start_time: float | None = None
    no_hand_streak = 0

    def _reset_active_buffers(*, preserve_no_hand_streak: bool = False) -> None:
        nonlocal fsm_transitions
        nonlocal tool_conf_timeline
        nonlocal event_timeline
        nonlocal contract_rejections
        nonlocal trigger_reason
        nonlocal completion_reason
        nonlocal rejected_trigger_reasons
        nonlocal raw_oob_count
        nonlocal bimanual_stats
        nonlocal depth_stats
        nonlocal color_stats
        nonlocal motion_stats
        nonlocal fused_trigger_hits
        nonlocal motion_trigger_hits
        nonlocal active
        nonlocal active_stage_name
        nonlocal active_start_frame
        nonlocal active_start_time
        nonlocal no_hand_streak
        cleared = _clear_active_segment_state()
        fsm_transitions = cleared["fsm_transitions"]
        tool_conf_timeline = cleared["tool_conf_timeline"]
        event_timeline = cleared["event_timeline"]
        contract_rejections = cleared["contract_rejections"]
        trigger_reason = cleared["trigger_reason"]
        completion_reason = cleared["completion_reason"]
        rejected_trigger_reasons = cleared["rejected_trigger_reasons"]
        raw_oob_count = cleared["raw_oob_count"]
        bimanual_stats = cleared["bimanual_stats"]
        depth_stats = cleared["depth_stats"]
        color_stats = cleared["color_stats"]
        motion_stats = cleared["motion_stats"]
        fused_trigger_hits = cleared["fused_trigger_hits"]
        motion_trigger_hits = cleared["motion_trigger_hits"]
        active = bool(cleared["active"])
        active_stage_name = cleared["active_stage_name"]
        active_start_frame = cleared["active_start_frame"]
        active_start_time = cleared["active_start_time"]
        if not preserve_no_hand_streak:
            no_hand_streak = int(cleared["no_hand_streak"])

    try:
        for pkt in packets:
            frame = pkt["frame"]
            frame_id = pkt["frame_id"]
            timestamp = pkt["timestamp"]
            hand_data = pkt["hand_data"]
            keypoints_px = pkt["keypoints_px"]
            keypoints_px_by_label = pkt.get("keypoints_px_by_label", {})
            depth_mm = pkt.get("depth_mm")
            current_stage_desc = _resolve_current_stage_descriptor(motion, stage_descriptors, None)
            detections = object_detector.detect(frame) if object_detector is not None else None

            frame_data = sync.sync(
                hand_data=hand_data,
                tool_data=_none_tool(timestamp),
                timestamp=timestamp,
                frame_id=frame_id,
                detections=detections,
                expected_parts=current_stage_desc.get("expected_parts"),
                expected_tools=current_stage_desc.get("expected_tools"),
                expected_inventory=current_stage_desc.get("expected_inventory"),
                active_part_candidates=current_stage_desc.get("active_part_candidates"),
                active_tool_candidates=current_stage_desc.get("active_tool_candidates"),
                stage_name=current_stage_desc.get("stage_name"),
                workspace_filter=workspace_filter,
            )
            frame_data["object_debug"] = {
                "active_part": frame_data.get("active_part"),
                "active_tool": frame_data.get("active_tool"),
                "inventory": frame_data.get("inventory", {}),
                "workspace_inventory": frame_data.get("workspace_inventory", {}),
            }
            _attach_pose_classifier_predictions(
                frame=frame,
                frame_data=frame_data,
                current_stage_desc=current_stage_desc,
                frame_id=int(frame_id),
                pose_classifier=pose_classifier,
            )
            _maybe_export_pose_pilot_candidate(
                frame=frame,
                frame_data=frame_data,
                stage_name=str(current_stage_desc.get("stage_name", "")),
                frame_id=int(frame_id),
                timestamp=float(timestamp),
                pose_pilot_bridge=pose_pilot_bridge,
                pose_pilot_exporter=pose_pilot_exporter,
            )
            _log_object_awareness_debug(frame_data, int(frame_id), str(current_stage_desc.get("stage_name", "")))
            cached_point = builder.build_trajectory_point(frame_data)
            frame_data["anchor_valid"] = cached_point is not None
            selected_label = selector.select(frame_data)
            if selected_label is not None:
                keypoints_px = keypoints_px_by_label.get(selected_label)
            detected_regions = region_detector.detect_regions(frame) if region_detector is not None else None
            _annotate_region_hits(
                frame_data=frame_data,
                keypoints_px_by_label=keypoints_px_by_label,
                selected_label=selected_label,
                region_registry=region_registry,
                detected_regions=detected_regions,
                operation_mode=operation_mode,
                trigger_point_mode=anchor_mode,
            )
            if event_extractor is not None:
                event_hand_scope = _resolve_event_hand_scope(motion, stage_descriptors)
                event_timeline.extend(
                    event_extractor.update_with_scope(
                        frame_data,
                        frame_id=int(frame_id),
                        timestamp=float(timestamp),
                        hand_scope=event_hand_scope,
                    )
                )
                frame_data["fsm_region_hits"] = event_extractor.build_stable_region_hits(
                    frame_data, hand_scope=event_hand_scope
                )
            if depth_trigger is not None and depth_mm is not None:
                depth_events, depth_results = _apply_depth_triggers(
                    depth_trigger=depth_trigger,
                    depth_mm=depth_mm,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    region_registry=region_registry,
                    operation_mode=operation_mode,
                )
                frame_data["depth_events"] = depth_events or []
                frame_data["depth_trigger_results"] = depth_results or {}
                _update_depth_stats(depth_stats, depth_results or {})
                if depth_fallback_rgb:
                    if depth_events:
                        event_timeline.extend(depth_events)
                        frame_data["depth_region_hits"] = _depth_events_to_hits(depth_events)
                    else:
                        frame_data["depth_region_hits"] = {}
            if color_trigger is not None:
                color_results = color_trigger.update(frame)
                frame_data["color_trigger_results"] = color_results or {}
                _update_color_stats(color_stats, color_results or {})
            if motion_trigger is not None:
                motion_results = motion_trigger.update(frame)
                phase = "start" if str(motion.state) == "WAIT_TRIGGER" else "completion"
                allowed = _resolve_motion_allowed_regions(motion, stage_descriptors, operation_mode, phase)
                if allowed:
                    motion_results = {rid: res for rid, res in (motion_results or {}).items() if rid in allowed}
                if motion_interaction_gate is not None:
                    motion_results = motion_interaction_gate.filter(motion_results, frame_data)
                else:
                    motion_results = _hand_confirmed_motion_results(motion_results, frame_data)
                frame_data["motion_trigger_results"] = motion_results or {}
                _update_motion_stats(motion_stats, motion_results or {})
                motion_hits = _build_motion_region_hits(
                    region_registry=region_registry,
                    operation_mode=operation_mode,
                    motion_results=motion_results or {},
                )
                frame_data["motion_region_hits"] = motion_hits
                if motion_hits:
                    motion_trigger_hits += int(len((motion_hits.get("any") or {}).get("regions", [])))
                if motion_results and motion_log_interval_frames > 0:
                    try:
                        if int(frame_id) % int(motion_log_interval_frames) == 0:
                            logger.info("[Motion] global=%.4f", float(getattr(motion_trigger, "last_global_motion_score", 0.0)))
                            for rid in sorted(motion_results.keys()):
                                res = motion_results[rid]
                                logger.info(
                                    "[Motion] ROI %s m=%.3f active=%s",
                                    rid,
                                    float(res.motion_score),
                                    bool(res.active),
                                )
                    except Exception:
                        pass
            if region_registry is not None:
                fused_hits, fused_count = _build_fused_hits(
                    region_registry=region_registry,
                    operation_mode=operation_mode,
                    rgb_region_hits=frame_data.get("fsm_region_hits") or frame_data.get("region_hits") or {},
                    depth_results=frame_data.get("depth_trigger_results"),
                    color_results=frame_data.get("color_trigger_results"),
                    threshold=fused_trigger_threshold,
                    w_color=fused_weight_color,
                    w_depth=fused_weight_depth,
                    w_rgb=fused_weight_rgb,
                )
                fused_trigger_hits += int(fused_count)
                frame_data["fused_region_hits"] = {"active": fused_hits, "any": fused_hits}

            if frame_data["hand"]["visible"]:
                no_hand_streak = 0
            else:
                no_hand_streak += 1

            prev = motion.state
            motion.update(frame_data)
            if hasattr(motion, "consume_contract_rejections"):
                new_rejects = getattr(motion, "consume_contract_rejections")()
                contract_rejections.extend(new_rejects)
                rejected_trigger_reasons.extend([r for r in new_rejects if str(r.get("phase", "")).startswith("start")])
            if hasattr(motion, "consume_transition_reasons"):
                for rr in getattr(motion, "consume_transition_reasons")():
                    if rr.get("kind") == "trigger":
                        trigger_reason = rr
                    elif rr.get("kind") == "completion":
                        completion_reason = rr
            if motion.state != prev:
                transition = {"from": prev, "to": motion.state, "t": max(0.0, timestamp)}
                fsm_transitions.append(transition)
                if transition["to"] == "RECORDING":
                    active = True
                    builder = _new_builder(dominant_hand, anchor_mode)
                    tool_conf_timeline = []
                    raw_oob_count = 0
                    active_stage_name = str(
                        (
                            trigger_reason
                            or {}
                        ).get("stage_name")
                        or current_stage_before
                    ) if False else None
                    active_stage_name = str(
                        (trigger_reason or {}).get("stage_name")
                        or _resolve_event_stage_name(motion, stage_descriptors)
                        or ""
                    ).strip() or None
                    active_start_frame = int(frame_id)
                    active_start_time = float(timestamp)

            if motion.state == "RECORDING" and active:
                _update_bimanual_stats(bimanual_stats, frame_data, timestamp)
                p = cached_point
                if p is not None:
                    builder.append(p)
                    tool_conf_timeline.append({"t": max(0.0, timestamp), "tool": "NONE", "confidence": 0.0})
                    if not (0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0):
                        raw_oob_count += 1

            if _render_frame(
                frame,
                pre=pre,
                motion_state=motion.state,
                frame_id=frame_id,
                timestamp=timestamp,
                show=show,
                delay_ms=delay_ms,
                anchor_mode=anchor_mode,
                keypoints_px=keypoints_px,
                work_region=work_region,
                drop_region=drop_region,
                region_registry=region_registry,
                depth_events=frame_data.get("depth_events"),
                depth_trigger_results=frame_data.get("depth_trigger_results"),
                color_trigger_results=frame_data.get("color_trigger_results"),
                motion_trigger_results=frame_data.get("motion_trigger_results"),
                frame_data=frame_data,
                show_depth_roi_overlay=show_depth_roi_overlay,
                show_regions=True,
                virtual_projector=virtual_projector,
                current_stage_desc=current_stage_desc,
                operation_mode=operation_mode,
                overlay_writer_state=overlay_writer_state,
            ):
                break

            segment_closed = bool(getattr(motion, "consume_segment_closed", lambda: False)())
            if (motion.state == "DONE" or segment_closed) and active:
                closure_type = "segment_closed" if segment_closed else "done"
                completed_stage_name = str((completion_reason or {}).get("stage_name") or "").strip() or active_stage_name
                stats_snapshot = _finalize_bimanual_stats(bimanual_stats)
                stats_snapshot["depth_stats"] = _finalize_depth_stats(depth_stats)
                stats_snapshot["color_trigger_stats"] = _finalize_color_stats(color_stats)
                stats_snapshot["fused_trigger_hits"] = int(fused_trigger_hits)
                stats_snapshot["motion_trigger_stats"] = _finalize_motion_stats(motion_stats)
                stats_snapshot["motion_trigger_hits"] = int(motion_trigger_hits)
                if hasattr(motion, "consume_depth_fallback_count"):
                    stats_snapshot["depth_fallback_count"] = getattr(motion, "consume_depth_fallback_count")()
                appended = _append_segment_if_valid(
                    segments,
                    builder,
                    fsm_transitions,
                    tool_conf_timeline,
                    raw_oob_count,
                    min_segment_points,
                    stage_name=completed_stage_name,
                    start_frame=active_start_frame,
                    start_time=active_start_time,
                    end_frame=int(frame_id),
                    end_time=float(timestamp),
                    closure_type=closure_type,
                    event_timeline=event_timeline,
                    contract_rejections=contract_rejections,
                    rejected_trigger_reasons=rejected_trigger_reasons,
                )
                if appended:
                    segments[-1]["bimanual_analytics"] = stats_snapshot
                _reset_active_buffers()

            waiting_for_first_anchor = bool(
                getattr(motion, "is_waiting_for_first_anchor", lambda: False)()
            )
            if active and motion.state == "RECORDING" and waiting_for_first_anchor:
                continue

            if active and motion.state == "RECORDING" and no_hand_streak >= no_hand_frames_to_split:
                completed_stage_name = str((completion_reason or {}).get("stage_name") or "").strip() or active_stage_name
                stats_snapshot = _finalize_bimanual_stats(bimanual_stats)
                stats_snapshot["depth_stats"] = _finalize_depth_stats(depth_stats)
                stats_snapshot["color_trigger_stats"] = _finalize_color_stats(color_stats)
                stats_snapshot["fused_trigger_hits"] = int(fused_trigger_hits)
                stats_snapshot["motion_trigger_stats"] = _finalize_motion_stats(motion_stats)
                stats_snapshot["motion_trigger_hits"] = int(motion_trigger_hits)
                if hasattr(motion, "consume_depth_fallback_count"):
                    stats_snapshot["depth_fallback_count"] = getattr(motion, "consume_depth_fallback_count")()
                appended = _append_segment_if_valid(
                    segments,
                    builder,
                    fsm_transitions,
                    tool_conf_timeline,
                    raw_oob_count,
                    min_segment_points,
                    stage_name=completed_stage_name,
                    start_frame=active_start_frame,
                    start_time=active_start_time,
                    end_frame=int(frame_id),
                    end_time=float(timestamp),
                    closure_type="no_hand_split",
                    event_timeline=event_timeline,
                    contract_rejections=contract_rejections,
                    rejected_trigger_reasons=rejected_trigger_reasons,
                )
                if appended:
                    segments[-1]["bimanual_analytics"] = stats_snapshot
                _reset_active_buffers()
                if hasattr(motion, "abort_recording_keep_stage"):
                    getattr(motion, "abort_recording_keep_stage")()
                else:
                    motion.reset()
    finally:
        close = getattr(packets, "close", None)
        if callable(close):
            close()
        _close_overlay_video_writer(overlay_writer_state)

    if active and builder.get_trajectory():
        completed_stage_name = str((completion_reason or {}).get("stage_name") or "").strip() or active_stage_name
        stats_snapshot = _finalize_bimanual_stats(bimanual_stats)
        stats_snapshot["depth_stats"] = _finalize_depth_stats(depth_stats)
        stats_snapshot["color_trigger_stats"] = _finalize_color_stats(color_stats)
        stats_snapshot["fused_trigger_hits"] = int(fused_trigger_hits)
        stats_snapshot["motion_trigger_stats"] = _finalize_motion_stats(motion_stats)
        stats_snapshot["motion_trigger_hits"] = int(motion_trigger_hits)
        if hasattr(motion, "consume_depth_fallback_count"):
            stats_snapshot["depth_fallback_count"] = getattr(motion, "consume_depth_fallback_count")()
        appended = _append_segment_if_valid(
            segments,
            builder,
            fsm_transitions,
            tool_conf_timeline,
            raw_oob_count,
            min_segment_points,
            stage_name=completed_stage_name,
            start_frame=active_start_frame,
            start_time=active_start_time,
            end_frame=None,
            end_time=None,
            closure_type="video_end",
            event_timeline=event_timeline,
            contract_rejections=contract_rejections,
            rejected_trigger_reasons=rejected_trigger_reasons,
        )
        if appended:
            segments[-1]["bimanual_analytics"] = stats_snapshot

    return segments


def _resolve_event_stage_name(motion, stage_descriptors: List[dict] | None) -> str | None:
    if not stage_descriptors:
        return None
    try:
        idx = int(getattr(motion, "current_stage_index", 0))
    except Exception:
        return None
    if 0 <= idx < len(stage_descriptors):
        name = str(stage_descriptors[idx].get("stage_name", "")).strip()
        return name or None
    return None


def extract_stage_from_video(
    video_path: Path,
    mode: str,
    pre: FramePreprocessor,
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    show: bool,
    delay_ms: int,
    work_region=WORK_REGION,
    drop_region=DROP_REGION,
    operation_mode: str = "dismantle",
    region_registry: RegionRegistry | None = None,
    region_detector: RegionDetector | None = None,
    motion_fsm_version: str = MOTION_FSM_VERSION,
    stage_descriptors: List[dict] | None = None,
    stage_name: str | None = None,
    depth_npz: Path | None = None,
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
    motion_global_suppress_threshold: float = MOTION_GLOBAL_SUPPRESS_THRESHOLD,
    motion_global_relative_factor: float = MOTION_GLOBAL_RELATIVE_FACTOR,
    motion_strong_score_threshold: float = MOTION_STRONG_SCORE_THRESHOLD,
    motion_strong_relative_factor: float = MOTION_STRONG_RELATIVE_FACTOR,
    fused_trigger_threshold: float = 0.65,
    fused_weight_color: float = 0.6,
    fused_weight_depth: float = 0.2,
    fused_weight_rgb: float = 0.2,
    pose_pilot_bridge: PosePilotBridge | None = None,
    pose_pilot_exporter: PosePilotExporter | None = None,
    pose_classifier: PoseClassifierManager | None = None,
    virtual_projector: VirtualProjectionGuide | None = None,
    virtual_projection_output_video: str | Path | None = None,
    virtual_projection_output_fps: float = 30.0,
) -> Tuple[List[dict], List[dict], List[dict], int, dict]:
    """
    Returns:
      trajectory_raw, fsm_transitions, tool_conf_timeline, raw_oob_count, bimanual_analytics
    """
    depth_mm = None
    if depth_npz is not None:
        depth_mm, _, _ = _load_depth_npz(Path(depth_npz))
    if mode == "depthai":
        packets = _depthai_frame_packets(video_path, pre, dominant_hand=dominant_hand)
    else:
        packets = _offline_frame_packets(video_path, pre, dominant_hand=dominant_hand, depth_mm=depth_mm)

    depth_trigger = None
    if depth_trigger_enabled and depth_mm is not None and region_registry is not None:
        masks = _build_depth_roi_masks(pre, region_registry, depth_mm.shape[1:], operation_mode)
        depth_trigger = DepthTriggerExtractor(
            roi_masks=masks,
            baseline_frames=depth_baseline_frames,
            delta_mm=depth_delta_mm,
            dwell_frames=depth_dwell_frames,
        )
    color_trigger = None
    if color_trigger_enabled and region_registry is not None:
        shape_hw = depth_mm.shape[1:] if depth_mm is not None else _resolve_video_shape(video_path)
        masks = _build_color_roi_masks(pre, region_registry, shape_hw, operation_mode)
        color_trigger = ColorChangeTrigger(
            roi_masks=masks,
            baseline_frames=color_baseline_frames,
            delta_threshold=color_delta_threshold,
            delta_range=color_delta_range,
        )
    motion_trigger = None
    motion_interaction_gate = None
    if motion_trigger_enabled and region_registry is not None:
        shape_hw = depth_mm.shape[1:] if depth_mm is not None else _resolve_video_shape(video_path)
        masks = _build_depth_roi_masks(pre, region_registry, shape_hw, operation_mode)
        motion_trigger = ROIMotionTrigger(
            roi_masks=masks,
            baseline_frames=motion_baseline_frames,
            ignore_frames=motion_ignore_frames,
            diff_threshold=motion_diff_threshold,
            score_threshold=motion_score_threshold,
            dwell_frames=motion_dwell_frames,
            history_frames=motion_history_frames,
            blur_kernel=motion_blur_kernel,
            global_suppress_threshold=motion_global_suppress_threshold,
            global_relative_factor=motion_global_relative_factor,
            strong_score_threshold=motion_strong_score_threshold,
            strong_relative_factor=motion_strong_relative_factor,
        )
        motion_interaction_gate = ROIInteractionGate(
            region_registry=region_registry,
            operation_mode=operation_mode,
            fallback_score_threshold=MOTION_HAND_FALLBACK_SCORE_THRESHOLD,
            history_frames=motion_history_frames,
        )

    return _extract_stage_from_packets(
        packets,
        pre=pre,
        anchor_mode=anchor_mode,
        dominant_hand=dominant_hand,
        active_hand_policy=active_hand_policy,
        show=show,
        delay_ms=delay_ms,
        work_region=work_region,
        drop_region=drop_region,
        operation_mode=operation_mode,
        region_registry=region_registry,
        region_detector=region_detector,
        motion_fsm_version=motion_fsm_version,
        stage_descriptors=stage_descriptors,
        stage_name=stage_name,
        depth_trigger=depth_trigger,
        depth_fallback_rgb=depth_fallback_rgb,
        show_depth_roi_overlay=show_depth_roi_overlay,
        color_trigger=color_trigger,
        motion_trigger=motion_trigger,
        motion_interaction_gate=motion_interaction_gate,
        motion_log_interval_frames=motion_log_interval_frames,
        fused_trigger_threshold=fused_trigger_threshold,
        fused_weight_color=fused_weight_color,
        fused_weight_depth=fused_weight_depth,
        fused_weight_rgb=fused_weight_rgb,
        pose_pilot_bridge=pose_pilot_bridge,
        pose_pilot_exporter=pose_pilot_exporter,
        pose_classifier=pose_classifier,
        virtual_projector=virtual_projector,
        virtual_projection_output_video=virtual_projection_output_video,
        virtual_projection_output_fps=virtual_projection_output_fps,
    )


def extract_all_stage_segments_from_video_offline(
    video_path: Path,
    pre: FramePreprocessor,
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    show: bool,
    delay_ms: int,
    no_hand_frames_to_split: int = 20,
    rearm_no_hand_frames: int = 8,
    min_segment_points: int = 40,
    work_region=WORK_REGION,
    drop_region=DROP_REGION,
    operation_mode: str = "dismantle",
    region_registry: RegionRegistry | None = None,
    region_detector: RegionDetector | None = None,
    motion_fsm_version: str = MOTION_FSM_VERSION,
    stage_descriptors: List[dict] | None = None,
    depth_npz: Path | None = None,
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
    motion_global_suppress_threshold: float = MOTION_GLOBAL_SUPPRESS_THRESHOLD,
    motion_global_relative_factor: float = MOTION_GLOBAL_RELATIVE_FACTOR,
    motion_strong_score_threshold: float = MOTION_STRONG_SCORE_THRESHOLD,
    motion_strong_relative_factor: float = MOTION_STRONG_RELATIVE_FACTOR,
    fused_trigger_threshold: float = 0.65,
    fused_weight_color: float = 0.6,
    fused_weight_depth: float = 0.2,
    fused_weight_rgb: float = 0.2,
    pose_pilot_bridge: PosePilotBridge | None = None,
    pose_pilot_exporter: PosePilotExporter | None = None,
    pose_classifier: PoseClassifierManager | None = None,
    virtual_projector: VirtualProjectionGuide | None = None,
    virtual_projection_output_video: str | Path | None = None,
    virtual_projection_output_fps: float = 30.0,
) -> List[dict]:
    """
    Extracts sequential stage segments from one full observed video.
    A segment is defined as one IDLE->RECORDING->DONE cycle.
    """
    depth_mm = None
    if depth_npz is not None:
        depth_mm, _, _ = _load_depth_npz(Path(depth_npz))
    packets = _offline_frame_packets(video_path, pre, dominant_hand=dominant_hand, depth_mm=depth_mm)
    depth_trigger = None
    if depth_trigger_enabled and depth_mm is not None and region_registry is not None:
        masks = _build_depth_roi_masks(pre, region_registry, depth_mm.shape[1:], operation_mode)
        depth_trigger = DepthTriggerExtractor(
            roi_masks=masks,
            baseline_frames=depth_baseline_frames,
            delta_mm=depth_delta_mm,
            dwell_frames=depth_dwell_frames,
        )
    color_trigger = None
    if color_trigger_enabled and region_registry is not None:
        shape_hw = depth_mm.shape[1:] if depth_mm is not None else _resolve_video_shape(video_path)
        masks = _build_color_roi_masks(pre, region_registry, shape_hw, operation_mode)
        color_trigger = ColorChangeTrigger(
            roi_masks=masks,
            baseline_frames=color_baseline_frames,
            delta_threshold=color_delta_threshold,
            delta_range=color_delta_range,
        )
    motion_trigger = None
    motion_interaction_gate = None
    if motion_trigger_enabled and region_registry is not None:
        shape_hw = depth_mm.shape[1:] if depth_mm is not None else _resolve_video_shape(video_path)
        masks = _build_depth_roi_masks(pre, region_registry, shape_hw, operation_mode)
        motion_trigger = ROIMotionTrigger(
            roi_masks=masks,
            baseline_frames=motion_baseline_frames,
            ignore_frames=motion_ignore_frames,
            diff_threshold=motion_diff_threshold,
            score_threshold=motion_score_threshold,
            dwell_frames=motion_dwell_frames,
            history_frames=motion_history_frames,
            blur_kernel=motion_blur_kernel,
            global_suppress_threshold=motion_global_suppress_threshold,
            global_relative_factor=motion_global_relative_factor,
            strong_score_threshold=motion_strong_score_threshold,
            strong_relative_factor=motion_strong_relative_factor,
        )
        motion_interaction_gate = ROIInteractionGate(
            region_registry=region_registry,
            operation_mode=operation_mode,
            fallback_score_threshold=MOTION_HAND_FALLBACK_SCORE_THRESHOLD,
            history_frames=motion_history_frames,
        )
    return _extract_segments_from_packets(
        packets,
        pre=pre,
        anchor_mode=anchor_mode,
        dominant_hand=dominant_hand,
        active_hand_policy=active_hand_policy,
        show=show,
        delay_ms=delay_ms,
        no_hand_frames_to_split=no_hand_frames_to_split,
        rearm_no_hand_frames=rearm_no_hand_frames,
        min_segment_points=min_segment_points,
        work_region=work_region,
        drop_region=drop_region,
        operation_mode=operation_mode,
        region_registry=region_registry,
        region_detector=region_detector,
        motion_fsm_version=motion_fsm_version,
        stage_descriptors=stage_descriptors,
        depth_trigger=depth_trigger,
        depth_fallback_rgb=depth_fallback_rgb,
        show_depth_roi_overlay=show_depth_roi_overlay,
        color_trigger=color_trigger,
        motion_trigger=motion_trigger,
        motion_interaction_gate=motion_interaction_gate,
        motion_log_interval_frames=motion_log_interval_frames,
        fused_trigger_threshold=fused_trigger_threshold,
        fused_weight_color=fused_weight_color,
        fused_weight_depth=fused_weight_depth,
        fused_weight_rgb=fused_weight_rgb,
        pose_pilot_bridge=pose_pilot_bridge,
        pose_pilot_exporter=pose_pilot_exporter,
        pose_classifier=pose_classifier,
        virtual_projector=virtual_projector,
        virtual_projection_output_video=virtual_projection_output_video,
        virtual_projection_output_fps=virtual_projection_output_fps,
    )


def extract_all_stage_segments_from_video_depthai(
    video_path: Path,
    pre: FramePreprocessor,
    anchor_mode: str,
    dominant_hand: str,
    active_hand_policy: str,
    show: bool,
    delay_ms: int,
    no_hand_frames_to_split: int = 20,
    rearm_no_hand_frames: int = 8,
    min_segment_points: int = 40,
    work_region=WORK_REGION,
    drop_region=DROP_REGION,
    operation_mode: str = "dismantle",
    region_registry: RegionRegistry | None = None,
    region_detector: RegionDetector | None = None,
    motion_fsm_version: str = MOTION_FSM_VERSION,
    stage_descriptors: List[dict] | None = None,
    depth_npz: Path | None = None,
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
    motion_global_suppress_threshold: float = MOTION_GLOBAL_SUPPRESS_THRESHOLD,
    motion_global_relative_factor: float = MOTION_GLOBAL_RELATIVE_FACTOR,
    motion_strong_score_threshold: float = MOTION_STRONG_SCORE_THRESHOLD,
    motion_strong_relative_factor: float = MOTION_STRONG_RELATIVE_FACTOR,
    fused_trigger_threshold: float = 0.65,
    fused_weight_color: float = 0.6,
    fused_weight_depth: float = 0.2,
    fused_weight_rgb: float = 0.2,
    pose_pilot_bridge: PosePilotBridge | None = None,
    pose_pilot_exporter: PosePilotExporter | None = None,
    pose_classifier: PoseClassifierManager | None = None,
    virtual_projector: VirtualProjectionGuide | None = None,
    virtual_projection_output_video: str | Path | None = None,
    virtual_projection_output_fps: float = 30.0,
) -> List[dict]:
    """
    Extracts sequential stage segments from one full observed video using DepthAI HandTracker.
    A segment is defined as one IDLE->RECORDING->DONE cycle.
    """
    depth_mm = None
    if depth_npz is not None:
        depth_mm, _, _ = _load_depth_npz(Path(depth_npz))
    packets = _depthai_frame_packets(video_path, pre, dominant_hand=dominant_hand)
    depth_trigger = None
    if depth_trigger_enabled and depth_mm is not None and region_registry is not None:
        masks = _build_depth_roi_masks(pre, region_registry, depth_mm.shape[1:], operation_mode)
        depth_trigger = DepthTriggerExtractor(
            roi_masks=masks,
            baseline_frames=depth_baseline_frames,
            delta_mm=depth_delta_mm,
            dwell_frames=depth_dwell_frames,
        )
    color_trigger = None
    if color_trigger_enabled and region_registry is not None:
        shape_hw = depth_mm.shape[1:] if depth_mm is not None else _resolve_video_shape(video_path)
        masks = _build_color_roi_masks(pre, region_registry, shape_hw, operation_mode)
        color_trigger = ColorChangeTrigger(
            roi_masks=masks,
            baseline_frames=color_baseline_frames,
            delta_threshold=color_delta_threshold,
            delta_range=color_delta_range,
        )
    motion_trigger = None
    motion_interaction_gate = None
    if motion_trigger_enabled and region_registry is not None:
        shape_hw = depth_mm.shape[1:] if depth_mm is not None else _resolve_video_shape(video_path)
        masks = _build_depth_roi_masks(pre, region_registry, shape_hw, operation_mode)
        motion_trigger = ROIMotionTrigger(
            roi_masks=masks,
            baseline_frames=motion_baseline_frames,
            ignore_frames=motion_ignore_frames,
            diff_threshold=motion_diff_threshold,
            score_threshold=motion_score_threshold,
            dwell_frames=motion_dwell_frames,
            history_frames=motion_history_frames,
            blur_kernel=motion_blur_kernel,
            global_suppress_threshold=motion_global_suppress_threshold,
            global_relative_factor=motion_global_relative_factor,
            strong_score_threshold=motion_strong_score_threshold,
            strong_relative_factor=motion_strong_relative_factor,
        )
        motion_interaction_gate = ROIInteractionGate(
            region_registry=region_registry,
            operation_mode=operation_mode,
            fallback_score_threshold=MOTION_HAND_FALLBACK_SCORE_THRESHOLD,
            history_frames=motion_history_frames,
        )
    return _extract_segments_from_packets(
        packets,
        pre=pre,
        anchor_mode=anchor_mode,
        dominant_hand=dominant_hand,
        active_hand_policy=active_hand_policy,
        show=show,
        delay_ms=delay_ms,
        no_hand_frames_to_split=no_hand_frames_to_split,
        rearm_no_hand_frames=rearm_no_hand_frames,
        min_segment_points=min_segment_points,
        work_region=work_region,
        drop_region=drop_region,
        operation_mode=operation_mode,
        region_registry=region_registry,
        region_detector=region_detector,
        motion_fsm_version=motion_fsm_version,
        stage_descriptors=stage_descriptors,
        depth_trigger=depth_trigger,
        depth_fallback_rgb=depth_fallback_rgb,
        show_depth_roi_overlay=show_depth_roi_overlay,
        color_trigger=color_trigger,
        motion_trigger=motion_trigger,
        motion_interaction_gate=motion_interaction_gate,
        motion_log_interval_frames=motion_log_interval_frames,
        fused_trigger_threshold=fused_trigger_threshold,
        fused_weight_color=fused_weight_color,
        fused_weight_depth=fused_weight_depth,
        fused_weight_rgb=fused_weight_rgb,
        pose_pilot_bridge=pose_pilot_bridge,
        pose_pilot_exporter=pose_pilot_exporter,
        pose_classifier=pose_classifier,
        virtual_projector=virtual_projector,
        virtual_projection_output_video=virtual_projection_output_video,
        virtual_projection_output_fps=virtual_projection_output_fps,
    )
