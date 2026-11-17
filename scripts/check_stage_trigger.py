#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.calibration.calibration_loader import CalibrationLoader
from assembly_assistant.pipeline.overlays import draw_fsm_overlay
from assembly_assistant.pipeline.region_registry import RegionRegistry
from assembly_assistant.pipeline.roi_motion_trigger import ROIMotionTrigger
from assembly_assistant.pipeline.stage_descriptor_loader import load_stage_descriptors
from assembly_assistant.trajectory.frame_processor import FramePreprocessor
from assembly_assistant.utils.logger import get_logger

from assembly_assistant.pipeline.video_extractors import (
    _build_depth_roi_masks,
    _build_motion_region_hits,
    _make_runtime,
    _none_tool,
    _offline_frame_packets,
    _resolve_video_shape,
)
from configs.experiment_configuration import WORK_REGION, DROP_REGION

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


def _project_norm_to_image(H_inv: np.ndarray, x: float, y: float):
    p = H_inv @ np.array([float(x), float(y), 1.0], dtype=float)
    if abs(float(p[2])) < 1e-9:
        return None
    return int(round(float(p[0] / p[2]))), int(round(float(p[1] / p[2])))


def _draw_focus_regions(frame, pre: FramePreprocessor, region_registry: RegionRegistry, focus_regions: dict):
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        H_inv = None

    colors = {
        "start": (0, 255, 255),
        "completion": (255, 128, 0),
        "tool": (0, 200, 0),
        "next": (255, 0, 255),
    }
    for bucket, region_ids in focus_regions.items():
        color = colors.get(bucket, (200, 200, 200))
        for rid in region_ids:
            region = region_registry.regions.get(rid)
            if region is None:
                continue
            pts = None
            if region.coordinate_frame == "image_px":
                pts = region.polygon
            elif region.coordinate_frame == "workspace_norm":
                if H_inv is None:
                    continue
                pts = []
                for x, y in region.polygon:
                    pt = _project_norm_to_image(H_inv, x, y)
                    if pt is None:
                        pts = None
                        break
                    pts.append(pt)
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], True, color, 2, cv2.LINE_AA)
            lx, ly = int(pts[0][0]), int(pts[0][1])
            cv2.putText(
                frame,
                f"{bucket}:{rid}",
                (lx + 4, ly + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


def _build_focus_regions(stage_desc: dict, *, motion=None) -> dict:
    descs = [stage_desc]
    group_id = str((stage_desc or {}).get("unordered_group_id", "")).strip()
    if group_id and motion is not None and hasattr(motion, "_unordered_remaining_descs"):
        try:
            remaining_descs = list(getattr(motion, "_unordered_remaining_descs")(group_id) or [])
            if remaining_descs:
                descs = remaining_descs
        except Exception:
            pass

    focus = {"start": [], "completion": [], "tool": [], "next": []}
    for desc in descs:
        focus["start"].extend(list(desc.get("start_source_regions", []) or []))
        focus["completion"].extend(list(desc.get("completion_regions", []) or []))
        focus["tool"].extend(list(desc.get("start_tool_regions", []) or []))
        focus["next"].extend(list(desc.get("next_stage_source_regions", []) or []))
    return {k: sorted({str(x) for x in vals if str(x)}) for k, vals in focus.items()}


def _allowed_motion_regions(stage_desc: dict, operation_mode: str, phase: str, *, motion=None, stage_descs: list[dict] | None = None) -> set[str]:
    descs_for_phase = [stage_desc]
    group_id = str((stage_desc or {}).get("unordered_group_id", "")).strip()
    if group_id and motion is not None and hasattr(motion, "_unordered_remaining_descs"):
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
        for desc in descs_for_phase:
            if phase_key == "start":
                allowed.update(desc.get("start_source_regions", []) or [])
                allowed.update(desc.get("start_tool_regions", []) or [])
            else:
                mode = str(desc.get("completion_mode", "next_stage_pick")).lower()
                if mode == "next_stage_pick":
                    allowed.update(desc.get("next_stage_source_regions", []) or [])
                else:
                    allowed.update(desc.get("completion_regions", []) or [])
    else:
        for desc in descs_for_phase:
            if phase_key == "start":
                allowed.update(desc.get("start_tool_regions", []) or [])
                allowed.update(desc.get("start_source_regions", []) or [])
            else:
                allowed.update(desc.get("completion_regions", []) or [])
    return {str(x) for x in allowed if str(x)}


def _current_stage_descriptor(motion, stage_descs: list[dict], fallback_desc: dict) -> dict:
    try:
        idx = int(getattr(motion, "current_stage_index", 0))
    except Exception:
        idx = 0
    if 0 <= idx < len(stage_descs):
        return stage_descs[idx]
    return fallback_desc


def main():
    parser = argparse.ArgumentParser(description="Inspect FSM stage triggering/segmentation on a reference or full observed video.")
    parser.add_argument("--video", required=True, help="Path to video file.")
    parser.add_argument("--stage-name", help="Optional stage name to start from. Omit for full-sequence checking from the first stage.")
    parser.add_argument("--operation-mode", choices=["assembly", "dismantle"], default="assembly")
    parser.add_argument("--regions-config", required=True, help="Region registry JSON.")
    parser.add_argument("--stage-descriptors-config", required=True, help="Stage descriptors JSON.")
    parser.add_argument("--calibration", required=True, help="Calibration folder.")
    parser.add_argument("--dominant-hand", default="RIGHT")
    parser.add_argument("--active-hand-policy", default="best_match")
    parser.add_argument("--anchor-mode", default="wrist_index_midpoint")
    parser.add_argument("--motion-baseline-frames", type=int, default=30)
    parser.add_argument("--motion-ignore-frames", type=int, default=0)
    parser.add_argument("--motion-diff-threshold", type=int, default=18)
    parser.add_argument("--motion-score-threshold", type=float, default=0.02)
    parser.add_argument("--motion-dwell-frames", type=int, default=6)
    parser.add_argument("--motion-history-frames", type=int, default=10)
    parser.add_argument("--motion-blur-kernel", type=str, default="5,5")
    parser.add_argument("--motion-global-suppress-threshold", type=float, default=0.03)
    parser.add_argument("--motion-global-relative-factor", type=float, default=2.0)
    parser.add_argument("--motion-log-interval-frames", type=int, default=30)
    parser.add_argument("--stop-after-first-segment", action="store_true", help="Useful for reference clips: stop once the first segment closes.")
    parser.add_argument("--delay-ms", type=int, default=30)
    args = parser.parse_args()

    calib_path = Path(args.calibration)
    _ = CalibrationLoader.load_homography(str(calib_path))
    pre = FramePreprocessor(str(calib_path))
    region_registry = RegionRegistry.from_file(args.regions_config)
    # Load stage names from descriptor config to preserve order.
    data = Path(args.stage_descriptors_config).read_text(encoding="utf-8")
    stage_json = json.loads(data)
    mode_block = (stage_json.get("modes") or {}).get(str(args.operation_mode).lower(), {})
    raw_stages = mode_block.get("stages") or []
    stage_names = [str(s.get("stage_name", "")).strip() for s in raw_stages if str(s.get("stage_name", "")).strip()]
    stage_descs = load_stage_descriptors(
        config_path=args.stage_descriptors_config,
        operation_mode=args.operation_mode,
        stage_names=stage_names,
    )
    stage_index = 0
    if args.stage_name:
        initial_stage_desc = None
        for idx, desc in enumerate(stage_descs):
            if str(desc.get("stage_name", "")).strip() == str(args.stage_name).strip():
                initial_stage_desc = desc
                stage_index = idx
                break
        if initial_stage_desc is None:
            raise RuntimeError(f"Stage '{args.stage_name}' not found in descriptors.")
    else:
        initial_stage_desc = stage_descs[0] if stage_descs else {}

    kparts = [p.strip() for p in str(args.motion_blur_kernel).split(",") if p.strip()]
    if len(kparts) != 2:
        raise RuntimeError("--motion-blur-kernel must be 'w,h' (e.g. 5,5)")
    blur_kernel = (int(kparts[0]), int(kparts[1]))

    shape_hw = _resolve_video_shape(Path(args.video))
    masks = _build_depth_roi_masks(pre, region_registry, shape_hw, args.operation_mode)
    motion_trigger = ROIMotionTrigger(
        roi_masks=masks,
        baseline_frames=int(args.motion_baseline_frames),
        ignore_frames=int(args.motion_ignore_frames),
        diff_threshold=int(args.motion_diff_threshold),
        score_threshold=float(args.motion_score_threshold),
        dwell_frames=int(args.motion_dwell_frames),
        history_frames=int(args.motion_history_frames),
        blur_kernel=blur_kernel,
        global_suppress_threshold=float(args.motion_global_suppress_threshold),
        global_relative_factor=float(args.motion_global_relative_factor),
    )

    sync, selector, motion, _ = _make_runtime(
        args.anchor_mode,
        args.dominant_hand,
        args.active_hand_policy,
        work_region=WORK_REGION,
        drop_region=DROP_REGION,
        operation_mode=args.operation_mode,
        motion_fsm_version="v2",
        stage_descriptors=stage_descs,
        use_motion_triggers=True,
    )
    motion.set_stage_index(stage_index)
    reference_stage_autostart = bool(
        args.operation_mode == "dismantle"
        and args.stage_name
        and initial_stage_desc.get("manual_start_allowed", True)
        and not (initial_stage_desc.get("start_tool_regions", []) or [])
        and not (initial_stage_desc.get("start_source_regions", []) or [])
    )

    focus_regions = _build_focus_regions(initial_stage_desc, motion=motion)
    allowed_regions = _allowed_motion_regions(initial_stage_desc, args.operation_mode, "start", motion=motion, stage_descs=stage_descs)

    prev_state = motion.state
    active_stage_name = None
    active_start_frame = None
    active_start_time = None
    completion_reason = None
    summary_rows: list[dict] = []
    packets = _offline_frame_packets(Path(args.video), pre, dominant_hand=args.dominant_hand)
    for pkt in packets:
        frame = pkt["frame"]
        frame_id = pkt["frame_id"]
        timestamp = pkt["timestamp"]
        hand_data = pkt["hand_data"]

        frame_data = sync.sync(
            hand_data=hand_data,
            tool_data=_none_tool(timestamp),
            timestamp=timestamp,
            frame_id=frame_id,
        )
        if reference_stage_autostart:
            frame_data["reference_stage_autostart"] = True
        _ = selector.select(frame_data)

        current_stage_desc = _current_stage_descriptor(motion, stage_descs, initial_stage_desc)
        motion_results = motion_trigger.update(frame)
        phase = "start" if str(motion.state) == "WAIT_TRIGGER" else "completion"
        allowed_regions = _allowed_motion_regions(current_stage_desc, args.operation_mode, phase, motion=motion, stage_descs=stage_descs)
        if motion_results and allowed_regions:
            motion_results = {rid: res for rid, res in motion_results.items() if rid in allowed_regions}
        if motion_results:
            motion_hits = _build_motion_region_hits(
                region_registry=region_registry,
                operation_mode=args.operation_mode,
                motion_results=motion_results,
            )
            frame_data["motion_region_hits"] = motion_hits
            if args.motion_log_interval_frames > 0:
                try:
                    if int(frame_id) % int(args.motion_log_interval_frames) == 0:
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

        motion.update(frame_data)
        if hasattr(motion, "consume_transition_reasons"):
            for rr in getattr(motion, "consume_transition_reasons")():
                if rr.get("kind") == "completion":
                    completion_reason = rr
        if motion.state != prev_state:
            logger.info("FSM %s -> %s at frame %s (t=%.2fs)", prev_state, motion.state, frame_id, timestamp)
            if motion.state == "RECORDING":
                active_stage_name = str(current_stage_desc.get("stage_name", "UNKNOWN"))
                active_start_frame = int(frame_id)
                active_start_time = float(timestamp)
            prev_state = motion.state

        segment_closed = bool(getattr(motion, "consume_segment_closed", lambda: False)())
        if segment_closed:
            closed_stage_name = str((completion_reason or {}).get("stage_name") or active_stage_name or current_stage_desc.get("stage_name", "UNKNOWN"))
            logger.info(
                "SEGMENT CLOSED for stage %s at frame %s (t=%.2fs); next stage index=%s",
                closed_stage_name,
                frame_id,
                timestamp,
                getattr(motion, "current_stage_index", 0),
            )
            summary_rows.append(
                {
                    "stage_name": closed_stage_name,
                    "start_frame": active_start_frame,
                    "end_frame": int(frame_id),
                    "duration_s": (
                        float(timestamp) - float(active_start_time)
                        if active_start_time is not None
                        else None
                    ),
                    "closure_type": "segment_closed",
                }
            )
            active_stage_name = None
            active_start_frame = None
            active_start_time = None
            completion_reason = None
            if args.stop_after_first_segment:
                break

        current_stage_desc = _current_stage_descriptor(motion, stage_descs, initial_stage_desc)
        focus_regions = _build_focus_regions(current_stage_desc, motion=motion)

        _draw_focus_regions(frame, pre, region_registry, focus_regions)
        draw_fsm_overlay(frame, motion.state, frame_id, timestamp)
        cv2.putText(
            frame,
            f"Stage: {current_stage_desc.get('stage_name', args.stage_name)}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Stage Trigger Check", frame)
        if cv2.waitKey(int(args.delay_ms)) & 0xFF == 27:
            break

    if str(getattr(motion, "state", "")) == "RECORDING":
        final_stage_desc = _current_stage_descriptor(motion, stage_descs, initial_stage_desc)
        logger.info(
            "VIDEO ENDED while still RECORDING stage %s; treating this as an open final segment.",
            final_stage_desc.get("stage_name", "UNKNOWN"),
        )
        summary_rows.append(
            {
                "stage_name": active_stage_name or str(final_stage_desc.get("stage_name", "UNKNOWN")),
                "start_frame": active_start_frame,
                "end_frame": None,
                "duration_s": None,
                "closure_type": "video_end",
            }
        )

    _log_segment_summary(summary_rows, header="Run Summary")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
