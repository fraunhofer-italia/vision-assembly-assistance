from assembly_assistant.fsm.motion_fsm import MotionFSM

def test_idle_to_recording():
    fsm = MotionFSM(
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_drop_hits=1,
        min_recording_frames=0,
    )

    frame = {
        "hand" :{
            "visible":True,
            "wrist": (0.3, 0.3)
        }
    }

    fsm.update(frame)

    assert fsm.state == MotionFSM.RECORDING


def test_recording_to_done():
    fsm = MotionFSM(
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_drop_hits=1,
        min_recording_frames=0,
    )
    fsm.state = MotionFSM.RECORDING

    frame ={
        "hand":{
            "visible":True,
            "wrist":(0.8, 0.8)
        }
    }

    fsm.update(frame)

    assert fsm.state == MotionFSM.DONE

def test_no_transition_if_hand_missing():
    fsm = MotionFSM(warmup_frames=0)

    frame = {"hand":{"visible":False}}

    fsm.update(frame)

    assert fsm.state == MotionFSM.IDLE

def test_reset():
    fsm = MotionFSM(warmup_frames=0)
    fsm.state = MotionFSM.DONE

    fsm.reset()

    assert fsm.state == MotionFSM.IDLE


def test_done_requires_drop_exit_clear_before_retrigger():
    fsm = MotionFSM(
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_drop_hits=1,
        min_recording_frames=0,
        drop_region=(0.6, 0.0, 1.0, 0.5),
        max_hand_jump=0.0,
        done_cooldown_frames=1,
        rearm_clear_frames=2,
        activity_v_min=0.001,
    )
    fsm.state = MotionFSM.RECORDING
    fsm.update(
        {
            "hands": {
                "left": {"visible": False, "wrist": None, "velocity": 0.0},
                "right": {"visible": True, "wrist": (0.8, 0.2), "velocity": 0.01},
            },
            "hand": {"visible": True, "wrist": (0.8, 0.2), "velocity": 0.01},
        }
    )
    assert fsm.state == MotionFSM.DONE

    # cooldown tick -> IDLE but must clear first
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.8, 0.2), "velocity": 0.01},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.8, 0.2), "velocity": 0.01},
        }
    )
    assert fsm.state == MotionFSM.IDLE

    # still in drop, should not retrigger.
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.8, 0.2), "velocity": 0.01},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "hand": {"visible": True, "wrist": (0.8, 0.2), "velocity": 0.01},
        }
    )
    assert fsm.state == MotionFSM.IDLE

    # provide clear frames (exit drop region)
    for _ in range(2):
        fsm.update(
            {
                "hands": {
                    "left": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
                    "right": {"visible": False, "wrist": None, "velocity": 0.0},
                },
                "hand": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
            }
        )
    # now a work/activity frame can start recording
    fsm.update(
        {
            "hands": {
                "left": {"visible": False, "wrist": None, "velocity": 0.0},
                "right": {"visible": True, "wrist": (0.25, 0.25), "velocity": 0.02},
            },
            "hand": {"visible": True, "wrist": (0.25, 0.25), "velocity": 0.02},
        }
    )
    assert fsm.state == MotionFSM.RECORDING


def test_assembly_done_trigger_from_region_roles_active_hand():
    fsm = MotionFSM(
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_drop_hits=1,
        min_recording_frames=0,
        operation_mode="assembly",
        done_region_roles=["part_pick"],
    )
    fsm.state = MotionFSM.RECORDING
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
                "right": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
            },
            "hand": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
            "region_hits": {
                "active": {"roles": ["part_pick"], "regions": ["piston_rod_box"]},
                "any": {"roles": ["part_pick"], "regions": ["piston_rod_box"]},
            },
        }
    )
    assert fsm.state == MotionFSM.DONE


def test_done_role_uses_active_hand_scope_by_default():
    fsm = MotionFSM(
        warmup_frames=0,
        min_consecutive_visible=1,
        min_consecutive_drop_hits=1,
        min_recording_frames=0,
        operation_mode="assembly",
        done_region_roles=["part_pick"],
    )
    fsm.state = MotionFSM.RECORDING
    fsm.update(
        {
            "hands": {
                "left": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
                "right": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
            },
            "hand": {"visible": True, "wrist": (0.2, 0.2), "velocity": 0.01},
            "region_hits": {
                "active": {"roles": [], "regions": []},
                "any": {"roles": ["part_pick"], "regions": ["piston_rod_box"]},
            },
        }
    )
    # Since active scope is default, any-scope alone should not trigger DONE.
    assert fsm.state == MotionFSM.RECORDING
