from assembly_assistant.pipeline.event_extractor import RegionEventExtractor
from assembly_assistant.pipeline.region_registry import RegionRegistry


def _registry():
    return RegionRegistry.from_dict(
        {
            "registry_id": "evt",
            "modes": ["assembly", "dismantle"],
            "regions": [
                {
                    "region_id": "pick_box",
                    "coordinate_frame": "image_px",
                    "polygon_px": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "roles": ["part_pick"],
                    "mode_semantics": {
                        "assembly": {"part_pick": True},
                        "dismantle": {"part_pick": False},
                    },
                },
                {
                    "region_id": "drop_box",
                    "coordinate_frame": "image_px",
                    "polygon_px": [[20, 0], [30, 0], [30, 10], [20, 10]],
                    "roles": ["part_drop"],
                    "mode_semantics": {
                        "assembly": {"part_drop": False},
                        "dismantle": {"part_drop": True},
                    },
                },
            ],
            "stage_bindings": {},
        }
    )


def _frame(active_regions_left=None, *, vel_left=0.01, left_visible=True, active_label="LEFT"):
    active_regions_left = active_regions_left or []
    return {
        "hands": {
            "left": {"visible": left_visible, "velocity": vel_left},
            "right": {"visible": False, "velocity": 0.0},
        },
        "region_hits": {
            "active_label": active_label,
            "hands": {
                "LEFT": {"regions": active_regions_left, "roles": []},
                "RIGHT": {"regions": [], "roles": []},
            },
            "active": {"regions": active_regions_left, "roles": []},
            "any": {"regions": active_regions_left, "roles": []},
        },
    }


def _frame_both(
    *,
    left_regions=None,
    right_regions=None,
    active_label="RIGHT",
    vel_left=0.01,
    vel_right=0.01,
):
    left_regions = left_regions or []
    right_regions = right_regions or []
    return {
        "hands": {
            "left": {"visible": True, "velocity": vel_left},
            "right": {"visible": True, "velocity": vel_right},
        },
        "region_hits": {
            "active_label": active_label,
            "hands": {
                "LEFT": {"regions": left_regions, "roles": []},
                "RIGHT": {"regions": right_regions, "roles": []},
            },
            "active": {
                "regions": right_regions if active_label == "RIGHT" else left_regions,
                "roles": [],
            },
            "any": {"regions": sorted(set(left_regions + right_regions)), "roles": []},
        },
    }


def test_event_extractor_enter_dwell_pick_candidate():
    ex = RegionEventExtractor(
        _registry(),
        operation_mode="assembly",
        enter_confirm_frames=2,
        dwell_confirm_frames=2,
        event_gap_frames=0,
    )
    e0 = ex.update(_frame(["pick_box"], vel_left=0.02), frame_id=1, timestamp=0.1)
    assert e0 == []
    e1 = ex.update(_frame(["pick_box"], vel_left=0.02), frame_id=2, timestamp=0.2)
    assert any(e["type"] == "enter_region" for e in e1)
    assert any(e["type"] == "pick_candidate" for e in e1)
    e2 = ex.update(_frame(["pick_box"], vel_left=0.01), frame_id=4, timestamp=0.4)
    assert any(e["type"] == "dwell_region" for e in e2)


def test_event_extractor_drop_candidate_on_dwell():
    ex = RegionEventExtractor(
        _registry(),
        operation_mode="dismantle",
        enter_confirm_frames=1,
        dwell_confirm_frames=1,
        event_gap_frames=0,
        drop_velocity_max=0.02,
    )
    e0 = ex.update(_frame(["drop_box"], vel_left=0.005), frame_id=1, timestamp=0.1)
    assert any(e["type"] == "enter_region" for e in e0)
    e1 = ex.update(_frame(["drop_box"], vel_left=0.001), frame_id=2, timestamp=0.2)
    assert any(e["type"] == "dwell_region" for e in e1)
    assert any(e["type"] == "drop_candidate" for e in e1)


def test_event_extractor_hysteresis_blocks_brief_toggle_exit():
    ex = RegionEventExtractor(
        _registry(),
        operation_mode="assembly",
        enter_confirm_frames=1,
        exit_confirm_frames=2,
        occlusion_tolerance_frames=1,
        event_gap_frames=0,
    )
    _ = ex.update(_frame(["pick_box"]), frame_id=1, timestamp=0.1)
    # brief one-frame outside should not produce exit due hysteresis
    e1 = ex.update(_frame([]), frame_id=2, timestamp=0.2)
    assert not any(e["type"] == "exit_region" for e in e1)
    # still back inside
    e2 = ex.update(_frame(["pick_box"]), frame_id=3, timestamp=0.3)
    assert not any(e["type"] == "exit_region" for e in e2)


def test_event_extractor_scope_active_ignores_non_active_hand_events():
    ex = RegionEventExtractor(
        _registry(),
        operation_mode="assembly",
        enter_confirm_frames=1,
        event_gap_frames=0,
    )
    events = ex.update_with_scope(
        _frame_both(left_regions=["pick_box"], right_regions=[], active_label="RIGHT", vel_left=0.02),
        frame_id=1,
        timestamp=0.1,
        hand_scope="active",
    )
    assert events == []


def test_event_extractor_scope_any_keeps_non_active_hand_events():
    ex = RegionEventExtractor(
        _registry(),
        operation_mode="assembly",
        enter_confirm_frames=1,
        event_gap_frames=0,
    )
    events = ex.update_with_scope(
        _frame_both(left_regions=["pick_box"], right_regions=[], active_label="RIGHT", vel_left=0.02),
        frame_id=1,
        timestamp=0.1,
        hand_scope="any",
    )
    assert any(e["type"] == "enter_region" for e in events)
    assert any(e["hand_label"] == "LEFT" for e in events)


def test_event_extractor_build_stable_hits_keeps_region_during_brief_toggle():
    ex = RegionEventExtractor(
        _registry(),
        operation_mode="assembly",
        enter_confirm_frames=1,
        exit_confirm_frames=2,
        occlusion_tolerance_frames=1,
        event_gap_frames=0,
    )
    f_in = _frame(["pick_box"], vel_left=0.02)
    _ = ex.update_with_scope(f_in, frame_id=1, timestamp=0.1, hand_scope="any")
    stable_in = ex.build_stable_region_hits(f_in, hand_scope="any")
    assert "pick_box" in stable_in["any"]["regions"]

    # One-frame miss should not clear stable region due hysteresis.
    f_out = _frame([], vel_left=0.02)
    _ = ex.update_with_scope(f_out, frame_id=2, timestamp=0.2, hand_scope="any")
    stable_out = ex.build_stable_region_hits(f_out, hand_scope="any")
    assert "pick_box" in stable_out["any"]["regions"]
