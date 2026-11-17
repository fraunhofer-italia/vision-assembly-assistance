from __future__ import annotations

from typing import Dict, List, Optional


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _boundary_ratio(observed_traj: List[dict], edge_eps: float = 0.02) -> float:
    if not observed_traj:
        return 0.0
    edge_hits = 0
    for pt in observed_traj:
        x = _safe_float(pt.get("x"), 0.5)
        y = _safe_float(pt.get("y"), 0.5)
        if x <= edge_eps or x >= (1.0 - edge_eps) or y <= edge_eps or y >= (1.0 - edge_eps):
            edge_hits += 1
    return edge_hits / max(1, len(observed_traj))


def classify_failure(
    *,
    observed_traj: List[dict],
    fatigue_metrics: Dict[str, float],
    component_scores: Dict[str, float],
    final_score: float,
    stage_threshold: float,
    raw_oob_count: int = 0,
    reference_len: Optional[int] = None,
    temporal_consistency: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Rule-based taxonomy for failed stage attempts.
    Labels:
      - wrong_path
      - incomplete_action
      - pause_heavy
      - jittery
      - boundary_error
      - wrong_stage_boundary
    """
    temporal_consistency = temporal_consistency or {}
    n_points = int(len(observed_traj) if observed_traj is not None else 0)
    duration = _safe_float(fatigue_metrics.get("stage_completion_time"), 0.0)
    displacement = _safe_float(fatigue_metrics.get("displacement"), 0.0)
    pause_ratio = _safe_float(fatigue_metrics.get("pause_ratio"), 0.0)
    micro_stops = int(fatigue_metrics.get("micro_stop_count", 0) or 0)
    jerk_index = _safe_float(fatigue_metrics.get("jerk_index"), 0.0)
    speed_var = _safe_float(fatigue_metrics.get("speed_variance"), 0.0)
    mean_speed = _safe_float(fatigue_metrics.get("mean_speed"), 0.0)
    dtw_score = _safe_float(component_scores.get("dtw"), 0.0)
    boundary_ratio = _boundary_ratio(observed_traj)
    early_to_late_delta = _safe_float(temporal_consistency.get("early_to_late_delta"), 0.0)

    ref_len_ratio = None
    if reference_len is not None and reference_len > 0:
        ref_len_ratio = n_points / float(reference_len)

    labels: List[str] = []

    # 1) Boundary quality issues (strongest signal)
    oob_threshold = max(3, int(0.05 * max(1, n_points)))
    if int(raw_oob_count) >= oob_threshold or boundary_ratio >= 0.20:
        labels.append("boundary_error")

    # 2) Segment boundary mismatch (split too early/late)
    if ref_len_ratio is not None and (ref_len_ratio < 0.55 or ref_len_ratio > 1.80):
        labels.append("wrong_stage_boundary")
    elif early_to_late_delta <= -0.20 and n_points < 50:
        labels.append("wrong_stage_boundary")

    # 3) Incomplete action
    if n_points < 30 or duration < 1.0 or displacement < 0.08:
        labels.append("incomplete_action")
    elif ref_len_ratio is not None and ref_len_ratio < 0.65:
        labels.append("incomplete_action")

    # 4) Pause-heavy behavior
    if pause_ratio >= 0.25 or micro_stops >= 6 or (mean_speed <= 0.03 and duration >= 1.5):
        labels.append("pause_heavy")

    # 5) High instability/jitter
    if jerk_index >= 2.0 or speed_var >= 0.02:
        labels.append("jittery")

    # 6) Wrong path geometry
    dtw_floor = max(0.70, stage_threshold - 0.10)
    if dtw_score < dtw_floor and "incomplete_action" not in labels:
        labels.append("wrong_path")

    # Fallback: if failed but none triggered, classify as wrong_path.
    if not labels and final_score < stage_threshold:
        labels = ["wrong_path"]

    priority = [
        "boundary_error",
        "wrong_stage_boundary",
        "incomplete_action",
        "wrong_path",
        "pause_heavy",
        "jittery",
    ]
    primary = next((p for p in priority if p in labels), labels[0] if labels else None)

    return {
        "primary": primary,
        "labels": labels,
        "signals": {
            "num_points": n_points,
            "duration_s": duration,
            "displacement": displacement,
            "pause_ratio": pause_ratio,
            "micro_stop_count": micro_stops,
            "jerk_index": jerk_index,
            "speed_variance": speed_var,
            "mean_speed": mean_speed,
            "dtw_score": dtw_score,
            "final_score": _safe_float(final_score, 0.0),
            "stage_threshold": _safe_float(stage_threshold, 0.0),
            "raw_oob_count": int(raw_oob_count),
            "boundary_ratio": boundary_ratio,
            "reference_length_ratio": ref_len_ratio,
            "early_to_late_delta": early_to_late_delta,
        },
    }

