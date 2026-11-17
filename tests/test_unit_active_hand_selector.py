from assembly_assistant.trajectory.active_hand_selector import ActiveHandSelector


def _hand(x, y, v=0.0):
    return {
        "visible": True,
        "wrist": (x, y),
        "velocity": v,
        "keypoints": None,
    }


def _frame(left=None, right=None, tool=None):
    return {
        "hands": {
            "left": left or {"visible": False, "wrist": None, "velocity": 0.0, "keypoints": None},
            "right": right or {"visible": False, "wrist": None, "velocity": 0.0, "keypoints": None},
            "active_label": None,
            "visible_count": 0,
        },
        "hand": {"visible": False, "wrist": None, "velocity": 0.0, "keypoints": None},
        "tool": tool or {"class": "NONE", "bbox": None, "confidence": 0.0},
    }


def test_selector_dominant_prefers_right():
    sel = ActiveHandSelector(policy="dominant", dominant_hand="RIGHT")
    f = _frame(left=_hand(0.1, 0.1), right=_hand(0.8, 0.8))
    label = sel.select(f)
    assert label == "RIGHT"
    assert f["hands"]["active_label"] == "RIGHT"
    assert f["hand"]["wrist"] == (0.8, 0.8)


def test_selector_closest_to_tool():
    sel = ActiveHandSelector(policy="closest_to_tool", dominant_hand="RIGHT")
    f = _frame(
        left=_hand(0.10, 0.10),
        right=_hand(0.80, 0.80),
        tool={"class": "ScrewdriverL", "bbox": (0.0, 0.0, 0.2, 0.2), "confidence": 0.9},
    )
    label = sel.select(f)
    assert label == "LEFT"
    assert f["hand"]["wrist"] == (0.10, 0.10)


def test_selector_best_match_prefers_continuity():
    sel = ActiveHandSelector(policy="best_match", dominant_hand="RIGHT")
    f1 = _frame(left=_hand(0.15, 0.15, v=0.2), right=_hand(0.85, 0.85, v=0.2))
    assert sel.select(f1) == "RIGHT"

    # keep both visible; continuity should keep RIGHT even with close velocities
    f2 = _frame(left=_hand(0.16, 0.16, v=0.25), right=_hand(0.84, 0.84, v=0.22))
    assert sel.select(f2) == "RIGHT"


def test_selector_best_match_switches_when_other_hand_becomes_active():
    sel = ActiveHandSelector(policy="best_match", dominant_hand="RIGHT")
    f1 = _frame(left=_hand(0.20, 0.20, v=0.01), right=_hand(0.80, 0.80, v=0.20))
    assert sel.select(f1) == "RIGHT"

    # Right becomes nearly static while left becomes active.
    # Hysteresis requires sustained evidence before switching.
    f2 = _frame(left=_hand(0.22, 0.23, v=0.08), right=_hand(0.80, 0.80, v=0.001))
    assert sel.select(f2) == "RIGHT"

    f3 = _frame(left=_hand(0.24, 0.26, v=0.09), right=_hand(0.80, 0.80, v=0.001))
    assert sel.select(f3) == "LEFT"
