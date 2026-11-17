import numpy as np
from assembly_assistant.synchronization.sync_manager import SyncManager

def test_sync_hand_missing(valid_tool):
    sm = SyncManager()

    frame = sm.sync(
        hand_data=None,
        tool_data=valid_tool,
        timestamp=1.0,
        frame_id=1
    )

    assert frame["hand"]["visible"] is False

def test_sync_valid_hand(valid_hand, valid_tool):
    sm = SyncManager()

    frame = sm.sync(
        hand_data=valid_hand,
        tool_data=valid_tool,
        timestamp=1.0,
        frame_id=1
    )

    assert frame["hand"]["visible"] is True
    assert frame["tool"]["class"] == "screwdriver_large_l"

def test_tool_temporal_rejection(valid_hand, valid_tool):
    sm = SyncManager(max_time_diff=0.1)

    valid_tool["timestamp"] = 5.0

    frame = sm.sync(
        hand_data=valid_hand,
        tool_data=valid_tool,
        timestamp=1.0,
        frame_id=1
    )

    assert frame["tool"]["class"] == "NONE"


def test_sync_two_hands_prefers_requested_side(valid_hand):
    sm = SyncManager(preferred_hand="LEFT")
    left = dict(valid_hand)
    right = dict(valid_hand)
    left["keypoints"] = np.array(valid_hand["keypoints"], dtype=float)
    right["keypoints"] = np.array(valid_hand["keypoints"], dtype=float) + 0.4

    frame = sm.sync(
        hand_data={"left": left, "right": right},
        tool_data=None,
        timestamp=1.0,
        frame_id=10,
    )

    assert frame["hands"]["visible_count"] == 2
    assert frame["hands"]["active_label"] == "LEFT"
    assert frame["hand"]["visible"] is True
    assert frame["hand"]["wrist"] == tuple(left["keypoints"][0])


def test_sync_two_hands_fallback_to_other_side(valid_hand):
    sm = SyncManager(preferred_hand="RIGHT")
    frame = sm.sync(
        hand_data={"left": valid_hand, "right": None},
        tool_data=None,
        timestamp=1.0,
        frame_id=11,
    )

    assert frame["hands"]["active_label"] == "LEFT"
    assert frame["hand"]["visible"] is True
