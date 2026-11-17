from assembly_assistant.analysis.error_taxonomy import classify_failure


def _traj(n=20, x=0.5, y=0.5):
    return [{"x": x, "y": y, "t": i * 0.05, "tool": "NONE"} for i in range(n)]


def test_classify_incomplete_action() -> None:
    taxonomy = classify_failure(
        observed_traj=_traj(n=10),
        fatigue_metrics={
            "stage_completion_time": 0.4,
            "displacement": 0.02,
            "pause_ratio": 0.0,
            "micro_stop_count": 0,
            "jerk_index": 0.0,
            "speed_variance": 0.0,
            "mean_speed": 0.1,
        },
        component_scores={"dtw": 0.9},
        final_score=0.5,
        stage_threshold=0.9,
        raw_oob_count=0,
        reference_len=100,
        temporal_consistency={},
    )
    assert "incomplete_action" in taxonomy["labels"]


def test_classify_boundary_error() -> None:
    boundary_traj = _traj(n=40, x=0.01, y=0.99)
    taxonomy = classify_failure(
        observed_traj=boundary_traj,
        fatigue_metrics={
            "stage_completion_time": 2.0,
            "displacement": 0.2,
            "pause_ratio": 0.0,
            "micro_stop_count": 0,
            "jerk_index": 0.0,
            "speed_variance": 0.0,
            "mean_speed": 0.2,
        },
        component_scores={"dtw": 0.85},
        final_score=0.7,
        stage_threshold=0.9,
        raw_oob_count=6,
        reference_len=40,
        temporal_consistency={},
    )
    assert "boundary_error" in taxonomy["labels"]

