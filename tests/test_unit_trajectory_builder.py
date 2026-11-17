import numpy as np
from assembly_assistant.trajectory.trajectory_builder import TrajectoryBuilder

def test_no_point_if_hand_not_visible():
    tb = TrajectoryBuilder()

    frame = {
        "hand":{"visible":False},
        "tool":{"class":"NONE"},
        "timestamp":1.0
    }

    assert tb.build_trajectory_point(frame) is None

def test_anchor_normalization(valid_hand):
    tb = TrajectoryBuilder()

    frame = {
        "hand":{
            "visible": True,
            "keypoints": valid_hand["keypoints"],
            "wrist":tuple(valid_hand["keypoints"][0])
        },
        "tool": {"class":"NONE","bbox":None},
        "timestamp":1.0
    }

    pt = tb.build_trajectory_point(frame)

    assert "x" in pt
    assert "y" in pt

def test_tool_association(valid_hand):
    tb = TrajectoryBuilder(tool_hand_distance_thresh=1.0)

    frame = {
        "hand":{
            "visible": True,
            "keypoints": valid_hand["keypoints"],
            "wrist":(0.3, 0.3)
        },
        "tool":{
            "class":"ScrewdriverL",
            "bbox":(0.29, 0.29, 0.31, 0.31),
            "confidence": 0.9
        },
        "active_tool": {"label": "screwdriver_large_l"},
        "timestamp":1.0
    }

    pt = tb.build_trajectory_point(frame)

    assert pt["tool"] == "screwdriver_large_l"

def test_reset_functionality(valid_hand):
    tb = TrajectoryBuilder()

    tb.append({"x":1})
    tb.reset()

    assert len(tb.get_trajectory()) == 0


def test_anchor_mode_midpoint_changes_x(valid_hand):
    frame = {
        "hand": {
            "visible": True,
            "keypoints": valid_hand["keypoints"],
            "wrist": tuple(valid_hand["keypoints"][0]),
        },
        "tool": {"class": "NONE", "bbox": None},
        "timestamp": 1.0,
    }

    pt_wrist = TrajectoryBuilder(anchor_mode="wrist").build_trajectory_point(frame)
    pt_mid = TrajectoryBuilder(anchor_mode="wrist_index_midpoint").build_trajectory_point(frame)

    assert pt_wrist["x"] != pt_mid["x"]
