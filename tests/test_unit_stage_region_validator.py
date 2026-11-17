from assembly_assistant.fsm.motion_fsm_v2 import FrameContext
from assembly_assistant.pipeline.stage_region_validator import StageRegionContractValidator


def _ctx(active_roles=None, active_regions=None):
    return FrameContext(
        in_work=True,
        in_drop=False,
        visible=True,
        activity=0.02,
        active_regions=set(active_regions or []),
        any_regions=set(active_regions or []),
        active_roles=set(active_roles or []),
        any_roles=set(active_roles or []),
    )


def _ctx_split(active_roles=None, active_regions=None, any_roles=None, any_regions=None):
    return FrameContext(
        in_work=True,
        in_drop=False,
        visible=True,
        activity=0.02,
        active_regions=set(active_regions or []),
        any_regions=set(any_regions or []),
        active_roles=set(active_roles or []),
        any_roles=set(any_roles or []),
    )


def test_dismantle_wrong_completion_region_rejected():
    v = StageRegionContractValidator()
    r = v.validate_completion(
        mode="dismantle",
        ctx=_ctx(active_roles=["part_drop"], active_regions=["wrong_box"]),
        stage_descriptor={"completion_regions": ["correct_box"], "completion_roles": ["part_drop"]},
    )
    assert r.allowed is False
    assert r.reason_code == "COMPLETION_WRONG_REGION"


def test_assembly_start_wrong_source_rejected():
    v = StageRegionContractValidator()
    r = v.validate_start(
        mode="assembly",
        ctx=_ctx(active_roles=["part_pick"], active_regions=["box_x"]),
        stage_descriptor={"start_roles": ["part_pick"], "start_source_regions": ["box_a"], "manual_start_allowed": False},
    )
    assert r.allowed is False
    assert r.reason_code == "START_WRONG_SOURCE_REGION"


def test_assembly_next_stage_pick_completion_allowed():
    v = StageRegionContractValidator()
    r = v.validate_completion(
        mode="assembly",
        ctx=_ctx(active_roles=["part_pick"], active_regions=["box_b"]),
        stage_descriptor={
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
        },
    )
    assert r.allowed is True


def test_dismantle_start_uses_any_hand_scope_by_default():
    v = StageRegionContractValidator()
    r = v.validate_start(
        mode="dismantle",
        ctx=_ctx_split(
            active_roles=[],
            active_regions=[],
            any_roles=["tool_pick"],
            any_regions=["box_tools_pair"],
        ),
        stage_descriptor={"manual_start_allowed": False, "start_tool_regions": ["box_tools_pair"]},
    )
    assert r.allowed is True
    assert r.reason_code == "OK_TOOL_START"


def test_assembly_completion_next_pick_uses_any_hand_scope_by_default():
    v = StageRegionContractValidator()
    r = v.validate_completion(
        mode="assembly",
        ctx=_ctx_split(
            active_roles=[],
            active_regions=[],
            any_roles=["part_pick"],
            any_regions=["box_b"],
        ),
        stage_descriptor={
            "completion_mode": "next_stage_pick",
            "next_stage_trigger_roles": ["part_pick"],
            "next_stage_source_regions": ["box_b"],
        },
    )
    assert r.allowed is True
    assert r.reason_code == "OK_NEXT_STAGE_PICK"
