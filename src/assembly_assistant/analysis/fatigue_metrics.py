from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_phase1_fatigue_metrics(
    traj: List[dict],
    *,
    pause_speed_threshold: float = 0.02,
    micro_stop_min_duration: float = 0.05,
    micro_stop_max_duration: float = 0.25,
) -> Dict[str, float]:
    """
    Computes phase-1 fatigue-related kinematic metrics from normalized trajectory.
    Expects points with keys: x, y, t.
    """
    if traj is None or len(traj) < 2:
        return {
            "num_points": int(len(traj) if traj is not None else 0),
            "stage_completion_time": 0.0,
            "path_length": 0.0,
            "displacement": 0.0,
            "endpoint_displacement": 0.0,
            "workspace_extent_displacement": 0.0,
            "displacement_definition": "max(endpoint_displacement, workspace_extent_diagonal)",
            "path_inefficiency_ratio": None,
            "mean_speed": 0.0,
            "speed_variance": 0.0,
            "acceleration_variance": 0.0,
            "jerk_index": 0.0,
            "pause_ratio": 0.0,
            "micro_stop_count": 0,
        }

    x = np.array([float(p["x"]) for p in traj], dtype=float)
    y = np.array([float(p["y"]) for p in traj], dtype=float)
    t = np.array([float(p["t"]) for p in traj], dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    valid = dt > 1e-9
    if not np.any(valid):
        return {
            "num_points": int(len(traj)),
            "stage_completion_time": 0.0,
            "path_length": 0.0,
            "displacement": 0.0,
            "endpoint_displacement": 0.0,
            "workspace_extent_displacement": 0.0,
            "displacement_definition": "max(endpoint_displacement, workspace_extent_diagonal)",
            "path_inefficiency_ratio": None,
            "mean_speed": 0.0,
            "speed_variance": 0.0,
            "acceleration_variance": 0.0,
            "jerk_index": 0.0,
            "pause_ratio": 0.0,
            "micro_stop_count": 0,
        }

    dx = dx[valid]
    dy = dy[valid]
    dt = dt[valid]

    seg_dist = np.sqrt(dx * dx + dy * dy)
    speed = seg_dist / dt

    duration = float(max(0.0, t[-1] - t[0]))
    path_length = float(np.sum(seg_dist))
    endpoint_displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    bbox_diagonal = float(np.sqrt((np.max(x) - np.min(x)) ** 2 + (np.max(y) - np.min(y)) ** 2))
    # Endpoint displacement collapses for circular or return-to-start actions.
    # Use the covered workspace extent as a stable fallback scale for inefficiency.
    displacement = float(max(endpoint_displacement, bbox_diagonal))
    if displacement > 1e-9:
        inefficiency = float(path_length / displacement)
    else:
        inefficiency = 1.0 if path_length <= 1e-9 else None

    mean_speed = float(np.mean(speed)) if len(speed) else 0.0
    speed_var = float(np.var(speed)) if len(speed) else 0.0

    if len(speed) >= 2:
        dt_acc = np.diff(t)
        dt_acc = dt_acc[1:][dt_acc[1:] > 1e-9] if len(dt_acc) > 1 else np.array([], dtype=float)
        acc = np.diff(speed)
        if len(acc) and len(dt_acc):
            m = min(len(acc), len(dt_acc))
            acc = acc[:m] / dt_acc[:m]
        else:
            acc = np.array([], dtype=float)
    else:
        acc = np.array([], dtype=float)

    acc_var = float(np.var(acc)) if len(acc) else 0.0

    if len(acc) >= 2:
        dt_jerk = np.diff(t)
        dt_jerk = dt_jerk[2:][dt_jerk[2:] > 1e-9] if len(dt_jerk) > 2 else np.array([], dtype=float)
        jerk = np.diff(acc)
        if len(jerk) and len(dt_jerk):
            m = min(len(jerk), len(dt_jerk))
            jerk = jerk[:m] / dt_jerk[:m]
        else:
            jerk = np.array([], dtype=float)
    else:
        jerk = np.array([], dtype=float)

    jerk_index = float(np.mean(np.abs(jerk))) if len(jerk) else 0.0

    pause_mask = speed < pause_speed_threshold
    pause_time = float(np.sum(dt[pause_mask])) if len(dt) else 0.0
    pause_ratio = float(pause_time / duration) if duration > 1e-9 else 0.0

    micro_stop_count = 0
    i = 0
    while i < len(pause_mask):
        if not pause_mask[i]:
            i += 1
            continue
        start = i
        while i < len(pause_mask) and pause_mask[i]:
            i += 1
        stop_duration = float(np.sum(dt[start:i]))
        if micro_stop_min_duration <= stop_duration <= micro_stop_max_duration:
            micro_stop_count += 1

    return {
        "num_points": int(len(traj)),
        "stage_completion_time": duration,
        "path_length": path_length,
        "displacement": displacement,
        "endpoint_displacement": endpoint_displacement,
        "workspace_extent_displacement": bbox_diagonal,
        "displacement_definition": "max(endpoint_displacement, workspace_extent_diagonal)",
        "path_inefficiency_ratio": inefficiency,
        "mean_speed": mean_speed,
        "speed_variance": speed_var,
        "acceleration_variance": acc_var,
        "jerk_index": jerk_index,
        "pause_ratio": pause_ratio,
        "micro_stop_count": int(micro_stop_count),
    }
