from assembly_assistant.analysis.fatigue_metrics import compute_phase1_fatigue_metrics


def test_phase1_metrics_constant_motion():
    traj = [
        {"x": 0.0, "y": 0.0, "t": 0.0},
        {"x": 0.5, "y": 0.0, "t": 0.5},
        {"x": 1.0, "y": 0.0, "t": 1.0},
    ]
    m = compute_phase1_fatigue_metrics(traj)
    assert m["stage_completion_time"] == 1.0
    assert abs(m["path_length"] - 1.0) < 1e-9
    assert abs(m["displacement"] - 1.0) < 1e-9
    assert abs(m["path_inefficiency_ratio"] - 1.0) < 1e-9
    assert m["pause_ratio"] == 0.0
    assert m["micro_stop_count"] == 0


def test_phase1_metrics_empty_or_short():
    m0 = compute_phase1_fatigue_metrics([])
    assert m0["num_points"] == 0
    assert m0["stage_completion_time"] == 0.0
    m1 = compute_phase1_fatigue_metrics([{"x": 0.0, "y": 0.0, "t": 0.0}])
    assert m1["num_points"] == 1
    assert m1["stage_completion_time"] == 0.0

