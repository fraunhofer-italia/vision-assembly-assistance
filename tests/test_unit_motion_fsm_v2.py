from assembly_assistant.fsm.motion_fsm_v2 import AssemblyPolicy, DismantlePolicy, MotionFSMV2


def _frame(*, in_work=True, active_roles=None, velocity=0.02, visible=True):
    active_roles = active_roles or []
    wrist = (0.3, 0.3) if in_work else (1.2, 0.3)
    return {
        "hands": {
            "left": {"visible": visible, "wrist": wrist, "velocity": velocity},
            "right": {"visible": False, "wrist": None, "velocity": 0.0},
        },
        "hand": {"visible": visible, "wrist": wrist, "velocity": velocity},
        "region_hits": {
            "active": {"roles": list(active_roles), "regions": []},
            "any": {"roles": list(active_roles), "regions": []},
        },
    }


def _frame_split(*, in_work=True, active_roles=None, active_regions=None, any_roles=None, any_regions=None, velocity=0.02, visible=True):
    active_roles = active_roles or []
    active_regions = active_regions or []
    any_roles = any_roles or []
    any_regions = any_regions or []
    wrist = (0.3, 0.3) if in_work else (1.2, 0.3)
    return {
        "hands": {
            "left": {"visible": visible, "wrist": wrist, "velocity": velocity},
            "right": {"visible": False, "wrist": None, "velocity": 0.0},
        },
        "hand": {"visible": visible, "wrist": wrist, "velocity": velocity},
        "region_hits": {
            "active": {"roles": list(active_roles), "regions": list(active_regions)},
            "any": {"roles": list(any_roles), "regions": list(any_regions)},
        },
    }


def test_motion_fsm_v2_dismantle_completes_on_drop_role():
    stage_desc = [
        {
            "stage_name": "Stage1",
            "manual_start_allowed": True,
            "start_tool_regions": [],
            "completion_regions": ["drop_box_A"],
            "completion_roles": ["part_drop"],
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        cooldown_frames=0,
    )
    fsm.update(_frame(in_work=True))
    assert fsm.state == MotionFSMV2.RECORDING

    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {
                "active": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
                "any": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
            },
        }
    )
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {
                "active": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
                "any": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
            },
        }
    )
    assert fsm.state == MotionFSMV2.COOLDOWN
    assert fsm.consume_segment_closed() is True


def test_motion_fsm_v2_assembly_can_start_from_pick_role_outside_work_region():
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        cooldown_frames=0,
    )
    # Outside work region but explicit pick role is present.
    fsm.update(_frame(in_work=False, active_roles=["part_pick"], velocity=0.02))
    assert fsm.state == MotionFSMV2.RECORDING


def test_motion_fsm_v2_start_recent_hit_window_allows_non_consecutive_pick_start():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "start_recent_hit_window": 4,
            "start_recent_hit_count": 2,
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=3,
        start_no_activity_frames=10,
    )

    # First validated pick hit.
    fsm.update(_frame_split(active_roles=["part_pick"], active_regions=["box_a"], any_roles=["part_pick"], any_regions=["box_a"]))
    assert fsm.state == MotionFSMV2.WAIT_TRIGGER

    # A weak/noisy frame breaks strict consecutive counting.
    fsm.update(_frame_split(active_roles=[], active_regions=[], any_roles=[], any_regions=[]))
    assert fsm.state == MotionFSMV2.WAIT_TRIGGER

    # Second validated hit inside the recent window should still trigger.
    fsm.update(_frame_split(active_roles=["part_pick"], active_regions=["box_a"], any_roles=["part_pick"], any_regions=["box_a"]))
    assert fsm.state == MotionFSMV2.RECORDING
    reasons = fsm.consume_transition_reasons()
    assert any(r.get("reason_code") == "OK_START_RECENT_WINDOW" for r in reasons)


def test_motion_fsm_v2_assembly_completes_on_next_pick_event():
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        cooldown_frames=0,
    )
    fsm.update(_frame(in_work=True, active_roles=["part_pick"]))
    assert fsm.state == MotionFSMV2.RECORDING
    fsm.update(_frame(in_work=True, active_roles=["part_pick"]))
    fsm.update(_frame(in_work=True, active_roles=["part_pick"]))
    assert fsm.state == MotionFSMV2.COOLDOWN
    assert fsm.consume_segment_closed() is True


def test_assembly_start_requires_stage_correct_source_region():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
            "cooldown_frames": 1,
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
    )
    # Wrong source region -> should not start.
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_x"]}, "any": {"roles": ["part_pick"], "regions": ["box_x"]}},
        }
    )
    assert fsm.state == MotionFSMV2.WAIT_TRIGGER

    # Correct source region -> should start.
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_a"]}, "any": {"roles": ["part_pick"], "regions": ["box_a"]}},
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING


def test_assembly_consecutive_stage_transition_without_hand_exit():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
            "cooldown_frames": 1,
        },
        {
            "stage_name": "A2",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_b"],
            "completion_mode": "completion_proxy",
            "completion_roles": ["assembly_target"],
            "completion_regions": ["assembly_target_zone"],
            "cooldown_frames": 1,
        },
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
    )
    # Stage1 start from box_a
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_a"]}, "any": {"roles": ["part_pick"], "regions": ["box_a"]}},
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING


def test_assembly_handoff_starts_next_stage_even_if_following_frames_are_quiet():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
            "cooldown_frames": 1,
        },
        {
            "stage_name": "A2",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_b"],
            "completion_mode": "terminal_end",
            "cooldown_frames": 1,
        },
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        start_handoff_frames=2,
    )
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_a"]}, "any": {"roles": ["part_pick"], "regions": ["box_a"]}},
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING

    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_b"]}, "any": {"roles": ["part_pick"], "regions": ["box_b"]}},
        }
    )
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_b"]}, "any": {"roles": ["part_pick"], "regions": ["box_b"]}},
        }
    )
    assert fsm.consume_segment_closed() is True
    assert fsm.state == MotionFSMV2.COOLDOWN

    # cooldown frame
    fsm.update(_frame_split(active_roles=[], active_regions=[], any_roles=[], any_regions=[]))
    # handoff should carry A1 completion evidence into A2 start
    fsm.update(_frame_split(active_roles=[], active_regions=[], any_roles=[], any_regions=[]))
    assert fsm.state == MotionFSMV2.RECORDING
    reasons = fsm.consume_transition_reasons()
    assert any(r.get("reason_code") == "OK_START_HANDOFF" for r in reasons)


def test_terminal_assembly_stage_stays_recording_despite_completion_proxy_hits():
    stage_desc = [
        {
            "stage_name": "A11",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_piston_rod_nut"],
            "completion_mode": "terminal_end",
            "completion_regions": ["assembly_target"],
            "completion_roles": ["assembly_target"],
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
    )
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_piston_rod_nut"]}, "any": {"roles": ["part_pick"], "regions": ["box_piston_rod_nut"]}},
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING


def test_completion_waits_for_first_anchor_point_after_valid_trigger():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
            "cooldown_frames": 1,
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        require_anchor_before_completion=True,
    )

    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02, "keypoints": [(0.3, 0.3)] * 21},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_a"]}, "any": {"roles": ["part_pick"], "regions": ["box_a"]}},
            "anchor_valid": False,
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING

    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02, "keypoints": [(0.3, 0.3)] * 21},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_b"]}, "any": {"roles": ["part_pick"], "regions": ["box_b"]}},
            "anchor_valid": False,
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING
    assert fsm.consume_segment_closed() is False

    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02, "keypoints": [(0.3, 0.3)] * 21},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_b"]}, "any": {"roles": ["part_pick"], "regions": ["box_b"]}},
            "anchor_valid": True,
        }
    )
    assert fsm.state == MotionFSMV2.WAIT_COMPLETION_CONFIRM
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02, "keypoints": [(0.3, 0.3)] * 21},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_b"]}, "any": {"roles": ["part_pick"], "regions": ["box_b"]}},
            "anchor_valid": True,
        }
    )
    assert fsm.state == MotionFSMV2.COOLDOWN
    assert fsm.consume_segment_closed() is True


def test_motion_trigger_mode_does_not_fallback_to_hand_region_hits_when_strict():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        use_motion_triggers=True,
        strict_motion_hits_only=True,
    )
    fsm.update(
        {
            "hands": {"left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02}, "right": {"visible": False, "wrist": None, "velocity": 0.0}},
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_a"]}, "any": {"roles": ["part_pick"], "regions": ["box_a"]}},
            "fsm_region_hits": {"active": {"roles": ["part_pick"], "regions": ["box_a"]}, "any": {"roles": ["part_pick"], "regions": ["box_a"]}},
            "motion_region_hits": {"active": {"roles": [], "regions": []}, "any": {"roles": [], "regions": []}},
        }
    )
    assert fsm.state == MotionFSMV2.WAIT_TRIGGER


def test_dismantle_tool_stage_requires_tool_pick_when_manual_disabled():
    stage_desc = [
        {
            "stage_name": "StageTool",
            "manual_start_allowed": False,
            "start_tool_regions": ["box_tools_pair"],
            "completion_regions": ["part_drop_A"],
            "completion_roles": ["part_drop"],
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
    )
    # motion in workspace without tool pick should NOT start
    fsm.update(_frame(in_work=True, active_roles=[]))
    assert fsm.state == MotionFSMV2.WAIT_TRIGGER

    # tool pick in allowed region should start
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {
                "active": {"roles": ["tool_pick"], "regions": ["box_tools_pair"]},
                "any": {"roles": ["tool_pick"], "regions": ["box_tools_pair"]},
            },
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING


def test_dismantle_completion_rejects_wrong_drop_region():
    stage_desc = [
        {
            "stage_name": "StageDrop",
            "manual_start_allowed": True,
            "start_tool_regions": [],
            "completion_regions": ["correct_drop_box"],
            "completion_roles": ["part_drop"],
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
    )
    fsm.update(_frame(in_work=True))
    assert fsm.state == MotionFSMV2.RECORDING
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {
                "active": {"roles": ["part_drop"], "regions": ["wrong_drop_box"]},
                "any": {"roles": ["part_drop"], "regions": ["wrong_drop_box"]},
            },
        }
    )
    assert fsm.state == MotionFSMV2.RECORDING
    rejects = fsm.consume_contract_rejections()
    assert len(rejects) >= 1
    assert any(r["reason_code"] == "COMPLETION_WRONG_REGION" for r in rejects)


def test_motion_fsm_v2_emits_trigger_and_completion_reasons():
    stage_desc = [
        {
            "stage_name": "Stage1",
            "manual_start_allowed": True,
            "start_tool_regions": [],
            "completion_regions": ["drop_box_A"],
            "completion_roles": ["part_drop"],
            "cooldown_frames": 0,
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        cooldown_frames=0,
    )
    fsm.update(_frame(in_work=True))
    reasons = fsm.consume_transition_reasons()
    assert any(r.get("kind") == "trigger" for r in reasons)

    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {
                "active": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
                "any": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
            },
        }
    )
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "region_hits": {
                "active": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
                "any": {"roles": ["part_drop"], "regions": ["drop_box_A"]},
            },
        }
    )
    reasons2 = fsm.consume_transition_reasons()
    assert any(r.get("kind") == "completion" for r in reasons2)


def test_motion_fsm_v2_dismantle_start_accepts_non_active_tool_pick_by_default():
    stage_desc = [
        {
            "stage_name": "StageTool",
            "manual_start_allowed": False,
            "start_tool_regions": ["box_tools_pair"],
            "completion_regions": ["drop_box_A"],
            "completion_roles": ["part_drop"],
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
    )
    # Active hand has no event; non-active hand contributes via `any`.
    fsm.update(
        _frame_split(
            active_roles=[],
            active_regions=[],
            any_roles=["tool_pick"],
            any_regions=["box_tools_pair"],
        )
    )
    assert fsm.state == MotionFSMV2.RECORDING


def test_motion_fsm_v2_assembly_completion_accepts_non_active_next_pick_by_default():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
            "cooldown_frames": 0,
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        cooldown_frames=0,
    )
    # Start correctly.
    fsm.update(
        _frame_split(
            active_roles=["part_pick"],
            active_regions=["box_a"],
            any_roles=["part_pick"],
            any_regions=["box_a"],
        )
    )
    assert fsm.state == MotionFSMV2.RECORDING
    # Completion only appears via `any` (active hand toggled away).
    fsm.update(
        _frame_split(
            active_roles=[],
            active_regions=[],
            any_roles=["part_pick"],
            any_regions=["box_b"],
        )
    )
    fsm.update(
        _frame_split(
            active_roles=[],
            active_regions=[],
            any_roles=["part_pick"],
            any_regions=["box_b"],
        )
    )
    assert fsm.state == MotionFSMV2.COOLDOWN


def test_motion_fsm_v2_clear_rearm_uses_stage_completion_contract():
    stage_desc = [
        {
            "stage_name": "StageDrop",
            "manual_start_allowed": True,
            "completion_roles": ["part_drop"],
            "completion_regions": ["drop_box_A"],
            "cooldown_frames": 0,
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        clear_rearm_frames=2,
    )
    # Simulate post-cooldown rearm phase.
    fsm.state = MotionFSMV2.WAIT_TRIGGER
    fsm._clear_rearm_left = 2

    # Wrong drop region should NOT reset rearm counter.
    fsm.update(
        _frame_split(
            active_roles=["part_drop"],
            active_regions=["drop_box_B"],
            any_roles=["part_drop"],
            any_regions=["drop_box_B"],
        )
    )
    assert fsm._clear_rearm_left == 1


def test_motion_fsm_v2_prefers_event_stable_hits_for_start_gating():
    stage_desc = [
        {
            "stage_name": "A1",
            "manual_start_allowed": False,
            "start_roles": ["part_pick"],
            "start_source_regions": ["box_a"],
            "completion_mode": "completion_proxy",
            "completion_roles": ["assembly_target"],
            "completion_regions": ["assembly_target_zone"],
        }
    ]
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
    )
    frame = _frame_split(
        active_roles=[],
        active_regions=[],
        any_roles=[],
        any_regions=[],
    )
    frame["fsm_region_hits"] = {
        "active": {"roles": ["part_pick"], "regions": ["box_a"]},
        "any": {"roles": ["part_pick"], "regions": ["box_a"]},
    }
    fsm.update(frame)
    assert fsm.state == MotionFSMV2.RECORDING


def test_motion_fsm_v2_throttles_repeated_wrong_region_reject_logs():
    stage_desc = [
        {
            "stage_name": "StageDrop",
            "manual_start_allowed": True,
            "completion_regions": ["correct_drop_box"],
            "completion_roles": ["part_drop"],
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        reject_log_min_gap_frames=10,
    )
    fsm.update(_frame(in_work=True))
    assert fsm.state == MotionFSMV2.RECORDING

    wrong = {
        "hands": {
            "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "right": {"visible": False, "wrist": None, "velocity": 0.0},
        },
        "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
        "region_hits": {
            "active": {"roles": ["part_drop"], "regions": ["wrong_drop_box"]},
            "any": {"roles": ["part_drop"], "regions": ["wrong_drop_box"]},
        },
    }
    for _ in range(4):
        fsm.update(wrong)
    rejects = fsm.consume_contract_rejections()
    wrong_region = [r for r in rejects if r.get("reason_code") == "COMPLETION_WRONG_REGION"]
    assert len(wrong_region) == 1


def test_abort_recording_keep_stage_preserves_current_stage():
    fsm = MotionFSMV2(
        policy=AssemblyPolicy(),
        stage_descriptors=[{"stage_name": "A1"}, {"stage_name": "A2"}],
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
    )
    fsm.current_stage_index = 1
    fsm.state = MotionFSMV2.RECORDING
    fsm._recording_anchor_points = 0
    fsm._anchor_startup_grace_left = 7
    fsm._recording_frames = 5

    fsm.abort_recording_keep_stage()

    assert fsm.state == MotionFSMV2.WAIT_TRIGGER
    assert fsm.current_stage_index == 1
    assert fsm._recording_anchor_points == 0
    assert fsm._anchor_startup_grace_left == 0


def test_dismantle_first_manual_stage_autostarts_without_roi():
    stage_desc = [
        {
            "stage_name": "D01_remove_piston_rod_nut_drop",
            "manual_start_allowed": True,
            "completion_regions": ["box_piston_rod_nut"],
            "completion_roles": ["part_drop"],
        }
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=3,
        start_no_activity_frames=10,
    )

    fsm.update(_frame(in_work=False, active_roles=[], visible=False, velocity=0.0))
    assert fsm.state == MotionFSMV2.RECORDING
    reasons = fsm.consume_transition_reasons()
    assert any(r.get("reason_code") == "OK_START_AUTO" for r in reasons)


def test_dismantle_reference_stage_autostarts_without_previous_boundary():
    stage_desc = [
        {
            "stage_name": "D01_remove_piston_rod_nut_drop",
            "manual_start_allowed": True,
            "completion_regions": ["box_piston_rod_nut"],
            "completion_roles": ["part_drop"],
        },
        {
            "stage_name": "D02_remove_tie_rod_assemblies_drop",
            "manual_start_allowed": True,
            "completion_regions": ["box_tie_rod_assemblies"],
            "completion_roles": ["part_drop"],
        },
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=3,
        start_no_activity_frames=10,
    )
    fsm.set_stage_index(1)

    frame = _frame(in_work=False, active_roles=[], visible=False, velocity=0.0)
    frame["reference_stage_autostart"] = True
    fsm.update(frame)

    assert fsm.state == MotionFSMV2.RECORDING
    reasons = fsm.consume_transition_reasons()
    assert any(r.get("reason_code") == "OK_START_AUTO" for r in reasons)


def test_dismantle_next_manual_stage_uses_previous_completion_boundary():
    stage_desc = [
        {
            "stage_name": "D01_remove_piston_rod_nut_drop",
            "manual_start_allowed": True,
            "completion_regions": ["box_piston_rod_nut"],
            "completion_roles": ["part_drop"],
            "cooldown_frames": 0,
        },
        {
            "stage_name": "D02_remove_tie_rod_assemblies_drop",
            "manual_start_allowed": True,
            "completion_regions": ["box_tie_rod_assemblies"],
            "completion_roles": ["part_drop"],
            "cooldown_frames": 0,
        },
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=3,
        start_no_activity_frames=10,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
        start_handoff_frames=2,
    )

    # D01 auto-starts.
    fsm.update(_frame(in_work=False, active_roles=[], visible=False, velocity=0.0))
    assert fsm.state == MotionFSMV2.RECORDING

    # D01 closes on its drop ROI.
    drop_frame = {
        "hands": {
            "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "right": {"visible": False, "wrist": None, "velocity": 0.0},
        },
        "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
        "region_hits": {
            "active": {"roles": ["part_drop"], "regions": ["box_piston_rod_nut"]},
            "any": {"roles": ["part_drop"], "regions": ["box_piston_rod_nut"]},
        },
    }
    fsm.update(drop_frame)
    fsm.update(drop_frame)
    assert fsm.state == MotionFSMV2.COOLDOWN
    assert fsm.current_stage_index == 1

    # Next quiet frame after cooldown should start D02 from the previous completion boundary.
    fsm.update(_frame(in_work=False, active_roles=[], visible=False, velocity=0.0))
    assert fsm.state == MotionFSMV2.RECORDING
    reasons = fsm.consume_transition_reasons()
    assert any(r.get("reason_code") == "OK_START_HANDOFF" for r in reasons)


def test_unordered_group_completion_uses_matching_member_not_current_index():
    stage_desc = [
        {
            "stage_name": "D08a_drop_mounting_screw",
            "manual_start_allowed": True,
            "completion_regions": ["box_mounting_screw"],
            "completion_roles": ["part_drop"],
            "unordered_group_id": "g1",
            "unordered_group_members": ["D08a_drop_mounting_screw", "D08b_drop_piston"],
        },
        {
            "stage_name": "D08b_drop_piston",
            "manual_start_allowed": True,
            "completion_regions": ["box_piston"],
            "completion_roles": ["part_drop"],
            "unordered_group_id": "g1",
            "unordered_group_members": ["D08a_drop_mounting_screw", "D08b_drop_piston"],
        },
        {
            "stage_name": "D09_remove_cylinder_tube_drop",
            "manual_start_allowed": True,
            "completion_regions": ["box_cylinder_tube"],
            "completion_roles": ["part_drop"],
        },
    ]
    fsm = MotionFSMV2(
        policy=DismantlePolicy(),
        stage_descriptors=stage_desc,
        warmup_frames=0,
        min_consecutive_visible=1,
        start_no_activity_frames=1,
        min_consecutive_completion_hits=1,
        completion_confirm_frames=1,
        min_recording_frames=0,
    )
    fsm.set_stage_index(0)
    fsm.update(_frame(in_work=False, active_roles=[], visible=False, velocity=0.0))
    assert fsm.state == MotionFSMV2.RECORDING

    piston_drop = {
        "hands": {
            "left": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
            "right": {"visible": False, "wrist": None, "velocity": 0.0},
        },
        "hand": {"visible": True, "wrist": (0.3, 0.3), "velocity": 0.02},
        "region_hits": {
            "active": {"roles": ["part_drop"], "regions": ["box_piston"]},
            "any": {"roles": ["part_drop"], "regions": ["box_piston"]},
        },
    }
    fsm.update(piston_drop)
    fsm.update(piston_drop)
    reasons = fsm.consume_transition_reasons()
    completion_reasons = [r for r in reasons if r.get("kind") == "completion"]
    assert any(r.get("stage_name") == "D08b_drop_piston" for r in completion_reasons)
