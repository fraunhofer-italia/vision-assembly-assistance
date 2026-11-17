# Run:
# python -m pytest -q tests/test_unit_trajectory_preprocessing.py

import numpy as np
from assembly_assistant.trajectory.preprocessing import TrajectoryPreprocessor

def test_time_normalization():
    traj = [
        {"x": 0.5, "y": 0.5, "t":100.0, "tool":"NONE", "tool_id": 0},
        {"x": 0.6, "y": 0.6, "t":101.0, "tool":"NONE", "tool_id": 0},
    ]

    processed = TrajectoryPreprocessor.preprocess(traj)

    assert processed[0]["t"] == 0
    assert processed[1]["t"] == 1

def test_sorted():
    traj = [
        {"x": 0.6, "y": 0.6, "t":2.0, "tool":"NONE", "tool_id": 0},
        {"x": 0.5, "y": 0.5, "t":1.0, "tool":"NONE", "tool_id": 0} , 
    ]

    processed = TrajectoryPreprocessor.preprocess(traj)

    assert processed[0]["t"] == 0

def test_nan_removal():
    traj = [
        {"x": 0.5, "y": 0.5, "t":1.0, "tool":"NONE", "tool_id": 0},
        {"x": np.nan, "y": 0.6, "t":2.0, "tool":"NONE", "tool_id": 0} ,
    ]

    processed = TrajectoryPreprocessor.preprocess(traj)

    assert len(processed) == 1


def test_oob_drop_interpolate_rectifies_points():
    traj = [
        {"x": 0.20, "y": 0.30, "t": 1.0, "tool": "NONE", "tool_id": 0},
        {"x": 0.25, "y": -0.10, "t": 2.0, "tool": "NONE", "tool_id": 0},  # OOB
        {"x": 0.30, "y": 0.40, "t": 3.0, "tool": "NONE", "tool_id": 0},
    ]

    processed = TrajectoryPreprocessor.preprocess(
        traj,
        oob_policy={"mode": "drop_interpolate", "bounds": [0, 1]},
    )

    assert len(processed) == 3
    assert 0.0 <= processed[1]["x"] <= 1.0
    assert 0.0 <= processed[1]["y"] <= 1.0


def test_oob_drop_policy_removes_out_of_bounds():
    traj = [
        {"x": 0.20, "y": 0.30, "t": 1.0, "tool": "NONE", "tool_id": 0},
        {"x": 1.30, "y": 0.40, "t": 2.0, "tool": "NONE", "tool_id": 0},  # OOB
        {"x": 0.30, "y": 0.40, "t": 3.0, "tool": "NONE", "tool_id": 0},
    ]

    processed = TrajectoryPreprocessor.preprocess(
        traj,
        oob_policy={"mode": "drop", "bounds": [0, 1]},
    )

    assert len(processed) == 2


def test_single_frame_spike_is_suppressed():
    traj = [
        {"x": 0.10, "y": 0.10, "t": 1.0, "tool": "NONE", "tool_id": 0},
        {"x": 0.80, "y": 0.80, "t": 2.0, "tool": "NONE", "tool_id": 0},  # isolated spike
        {"x": 0.11, "y": 0.11, "t": 3.0, "tool": "NONE", "tool_id": 0},
    ]

    processed = TrajectoryPreprocessor.preprocess(
        traj,
        oob_policy={"mode": "none"},
    )

    # Middle point should be pulled near neighbor midpoint.
    assert abs(processed[1]["x"] - 0.105) < 1e-6
    assert abs(processed[1]["y"] - 0.105) < 1e-6
