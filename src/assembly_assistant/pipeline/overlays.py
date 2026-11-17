from __future__ import annotations

import cv2
import numpy as np

from assembly_assistant.trajectory.frame_processor import FramePreprocessor


def draw_fsm_overlay(frame, motion_state: str, frame_id: int, timestamp: float) -> None:
    cv2.rectangle(frame, (10, 10), (360, 90), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"FSM: {motion_state}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"frame={frame_id} t={timestamp:.2f}s",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _project_norm_to_image(H_inv: np.ndarray, x: float, y: float):
    p = H_inv @ np.array([float(x), float(y), 1.0], dtype=float)
    if abs(float(p[2])) < 1e-9:
        return None
    return int(round(float(p[0] / p[2]))), int(round(float(p[1] / p[2])))


def draw_workspace_and_drop_overlay(
    frame,
    pre: FramePreprocessor,
    work_region,
    drop_region,
) -> None:
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        return

    def draw_region(region, color, label):
        x0, y0, x1, y1 = region
        pts_n = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        pts_i = []
        for x, y in pts_n:
            pt = _project_norm_to_image(H_inv, x, y)
            if pt is None:
                return
            pts_i.append(pt)
        poly = np.array([pts_i], dtype=np.int32)
        cv2.polylines(frame, poly, True, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            pts_i[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Draw calibrated workspace boundary (exact [0,1] corners from homography).
    draw_region((0.0, 0.0, 1.0, 1.0), (255, 0, 255), "Workspace")
    # Draw active work region used by MotionFSM.
    draw_region(work_region, (180, 80, 255), "Work")
    draw_region(drop_region, (0, 165, 255), "Drop")


def draw_region_registry_overlay(frame, pre: FramePreprocessor, region_registry) -> None:
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        H_inv = None

    colors = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
        (128, 255, 0),
        (0, 255, 128),
        (255, 0, 128),
    ]
    for i, (rid, region) in enumerate(region_registry.regions.items()):
        color = colors[i % len(colors)]
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
        cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2)
        lx, ly = int(pts[0][0]), int(pts[0][1])
        cv2.putText(frame, rid, (lx + 6, ly + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_depth_trigger_overlay(frame, depth_events: list | None) -> None:
    if not depth_events:
        return
    active = sorted({e.get("region_id") for e in depth_events if e.get("region_id")})
    if not active:
        return
    cv2.rectangle(frame, (10, 95), (520, 135), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Depth Trigger Active: {', '.join(active)}",
        (20, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_depth_roi_overlay(frame, pre: FramePreprocessor, region_registry, depth_results: dict | None) -> None:
    if not depth_results or region_registry is None:
        return
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        H_inv = None

    for rid, res in depth_results.items():
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
        color = (0, 255, 0) if res.active else (0, 0, 255)
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        if res.active:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [arr], color)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2)
        lx, ly = int(pts[0][0]), int(pts[0][1])
        cv2.putText(
            frame,
            f"{rid} d={res.delta_mm:.1f}mm w={int(res.dwell_frames)}",
            (lx + 4, ly + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_color_roi_overlay(frame, pre: FramePreprocessor, region_registry, color_results: dict | None) -> None:
    if not color_results or region_registry is None:
        return
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        H_inv = None

    for rid, res in color_results.items():
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
        color = (0, 200, 255) if res.active else (90, 90, 90)
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2)
        lx, ly = int(pts[0][0]), int(pts[0][1])
        cv2.putText(
            frame,
            f"{rid} c={res.score:.2f}",
            (lx + 4, ly + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_motion_roi_overlay(frame, pre: FramePreprocessor, region_registry, motion_results: dict | None) -> None:
    if not motion_results or region_registry is None:
        return
    try:
        H_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        H_inv = None

    for rid, res in motion_results.items():
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
        color = (0, 255, 0) if res.active else (120, 120, 120)
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        if res.active:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [arr], color)
            cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
        cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2)
        lx, ly = int(pts[0][0]), int(pts[0][1])
        cv2.putText(
            frame,
            f"{rid} m={res.motion_score:.2f}",
            (lx + 4, ly + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_anchor_overlay(frame, keypoints_px: np.ndarray, anchor_mode: str) -> None:
    if keypoints_px is None or len(keypoints_px) <= 5:
        return

    wrist = keypoints_px[0]
    index_mcp = keypoints_px[5]
    if anchor_mode == "wrist":
        anchor = wrist
    else:
        anchor = 0.5 * (wrist + index_mcp)

    wx, wy = int(round(float(wrist[0]))), int(round(float(wrist[1])))
    ix, iy = int(round(float(index_mcp[0]))), int(round(float(index_mcp[1])))
    ax, ay = int(round(float(anchor[0]))), int(round(float(anchor[1])))

    cv2.circle(frame, (wx, wy), 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), 4, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "A", (ax + 6, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


def draw_detection_overlay(frame, frame_data: dict | None) -> None:
    if not frame_data:
        return
    parts = frame_data.get("parts") or []
    tools = frame_data.get("tools") or []
    active_part = (frame_data.get("active_part") or {}).get("label")
    active_tool = (frame_data.get("active_tool") or {}).get("label")

    def _draw(items, color, active_label):
        for det in items:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            label = str(det.get("label", ""))
            conf = float(det.get("confidence", 0.0))
            thickness = 3 if label == active_label else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    _draw(parts, (0, 255, 255), active_part)
    _draw(tools, (255, 128, 0), active_tool)

    pose_candidate = frame_data.get("pose_pilot_candidate") or {}
    if pose_candidate:
        bbox = pose_candidate.get("bbox_xyxy")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            label = str(pose_candidate.get("part_label", "pose"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"pose-candidate {label}",
                (x1, min(frame.shape[0] - 8, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    pose_result = frame_data.get("pose_pilot_result") or {}
    if pose_result:
        verdict = str(pose_result.get("orientation_verdict", "pose")).strip()
        confidence = pose_result.get("alignment_confidence")
        text = f"pose={verdict}"
        if confidence is not None:
            text += f" conf={float(confidence):.2f}"
        cv2.putText(
            frame,
            text,
            (20, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
