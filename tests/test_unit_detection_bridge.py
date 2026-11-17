import numpy as np

from assembly_assistant.detectors.detection_bridge import select_primary_detection
from assembly_assistant.synchronization.sync_manager import SyncManager
from assembly_assistant.trajectory.trajectory_builder import TrajectoryBuilder


def _single_right_hand():
    keypoints = np.zeros((21, 2), dtype=float)
    keypoints[0] = [0.20, 0.20]
    keypoints[5] = [0.24, 0.22]
    return {"keypoints": keypoints, "velocity": 0.05}


def _single_right_hand_with_px():
    hand = _single_right_hand()
    hand["wrist_px"] = (128.0, 96.0)
    return hand


def test_select_primary_detection_prefers_screwdriver_over_parts():
    detections = [
        {"label": "piston", "bbox": [0.1, 0.1, 0.3, 0.3], "confidence": 0.95, "source_model": "visual_prompt"},
        {"label": "screwdriver_small_l", "bbox": [0.7, 0.2, 0.9, 0.4], "confidence": 0.60, "source_model": "open_vocab"},
    ]
    selected = select_primary_detection(detections)
    assert selected["class"] == "screwdriver_small_l"


def test_sync_manager_adds_detections_and_chooses_closest_part_to_active_hand():
    sync = SyncManager(preferred_hand="RIGHT")
    hand_data = _single_right_hand()
    detections = [
        {"label": "piston", "bbox": [0.15, 0.15, 0.25, 0.25], "confidence": 0.70, "source_model": "visual_prompt"},
        {"label": "rear_cover", "bbox": [0.70, 0.70, 0.90, 0.90], "confidence": 0.90, "source_model": "visual_prompt"},
    ]
    frame_data = sync.sync(hand_data=hand_data, tool_data=None, timestamp=1.0, frame_id=1, detections=detections)
    assert len(frame_data["detections"]) == 2
    assert frame_data["tool"]["class"] == "piston"


def test_trajectory_builder_tolerates_non_tool_detection_labels():
    sync = SyncManager(preferred_hand="RIGHT")
    hand_data = _single_right_hand()
    detections = [
        {"label": "piston_rod", "bbox": [0.15, 0.15, 0.25, 0.25], "confidence": 0.85, "source_model": "visual_prompt"},
    ]
    _ = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.0,
        frame_id=1,
        detections=detections,
        active_part_candidates=["piston_rod"],
    )
    frame_data = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.1,
        frame_id=2,
        detections=detections,
        active_part_candidates=["piston_rod"],
    )
    point = TrajectoryBuilder(anchor_mode="wrist_index_midpoint").build_trajectory_point(frame_data)
    assert point is not None
    assert point["tool"] == "NONE"
    assert point["active_part"] == "piston_rod"
    assert point["tool_id"] == 0


def test_sync_manager_requires_stable_part_before_marking_active():
    sync = SyncManager(preferred_hand="RIGHT")
    hand_data = _single_right_hand()
    detections = [
        {"label": "piston", "bbox": [0.15, 0.15, 0.25, 0.25], "confidence": 0.90, "source_model": "visual_prompt"},
    ]

    frame1 = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.0,
        frame_id=1,
        detections=detections,
        active_part_candidates=["piston"],
    )
    assert frame1["active_part"] is None

    frame2 = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.1,
        frame_id=2,
        detections=detections,
        active_part_candidates=["piston"],
    )
    assert frame2["active_part"]["label"] == "piston"


def test_sync_manager_does_not_activate_far_transient_detection():
    sync = SyncManager(preferred_hand="RIGHT")
    hand_data = _single_right_hand()
    detections = [
        {"label": "rear_cover", "bbox": [0.70, 0.70, 0.90, 0.90], "confidence": 0.95, "source_model": "visual_prompt"},
    ]

    frame = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.0,
        frame_id=1,
        detections=detections,
        active_part_candidates=["rear_cover"],
    )
    assert frame["active_part"] is None


def test_sync_manager_clears_active_part_shortly_after_drop():
    sync = SyncManager(preferred_hand="RIGHT")
    hand_data = _single_right_hand()
    detections = [
        {"label": "piston", "bbox": [0.15, 0.15, 0.25, 0.25], "confidence": 0.90, "source_model": "visual_prompt"},
    ]

    _ = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.0,
        frame_id=1,
        detections=detections,
        active_part_candidates=["piston"],
    )
    held = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.1,
        frame_id=2,
        detections=detections,
        active_part_candidates=["piston"],
    )
    assert held["active_part"]["label"] == "piston"

    linger = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.2,
        frame_id=3,
        detections=[],
        active_part_candidates=["piston"],
    )
    assert linger["active_part"]["label"] == "piston"

    cleared = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.3,
        frame_id=4,
        detections=[],
        active_part_candidates=["piston"],
    )
    assert cleared["active_part"] is None


def test_sync_manager_activates_pixel_space_detection_using_pixel_wrist():
    sync = SyncManager(preferred_hand="RIGHT")
    hand_data = _single_right_hand_with_px()
    detections = [
        {"label": "bttm_tie_rod_nut", "bbox": [100, 80, 150, 120], "confidence": 0.92, "source_model": "visual_prompt"},
    ]

    frame1 = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.0,
        frame_id=1,
        detections=detections,
        active_part_candidates=["bttm_tie_rod_nut"],
    )
    assert frame1["active_part"] is None

    frame2 = sync.sync(
        hand_data=hand_data,
        tool_data=None,
        timestamp=1.1,
        frame_id=2,
        detections=detections,
        active_part_candidates=["bttm_tie_rod_nut"],
    )
    assert frame2["active_part"]["label"] == "bttm_tie_rod_nut"
