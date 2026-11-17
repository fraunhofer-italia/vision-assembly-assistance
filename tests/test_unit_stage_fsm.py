from data.mocks.trajectory_generator import generate_circular_traj
from assembly_assistant.fsm.stage_fsm import StageFSM

########################################################
#------------ STAGE A SUCCESS --------------------------
########################################################

def test_stageA_pass(stage_fsm, perfect_scores, tool_perfect_scores):
    traj = generate_circular_traj(tool="NONE")

    result = stage_fsm.evaluate(
        observed_traj=traj,
        dtw_scores=perfect_scores,
        tool_score=tool_perfect_scores
    )

    assert result["stage_passed"] is True
    assert result["next_stage"] == "StageB_Linear"

#########################################################
#---------------- TOOL FAILURE --------------------------
#########################################################

def test_stageA_tool_fail(stage_fsm, perfect_scores, tool_weak_scores):
    traj = generate_circular_traj(tool="ScrewdriverL")    # Wrong Tool

    result = stage_fsm.evaluate(
        observed_traj=traj,
        dtw_scores=perfect_scores,
        tool_score=tool_weak_scores
    )

    assert result["stage_passed"] is False


#########################################################
#---------------- Low Score FAILURE ---------------------
#########################################################

def test_stageA_tool_fail(stage_fsm, weak_scores, tool_perfect_scores):
    traj = generate_circular_traj(tool="NONE")    

    result = stage_fsm.evaluate(
        observed_traj=traj,
        dtw_scores= weak_scores,
        tool_score=tool_perfect_scores
    )

    assert result["stage_passed"] is False


#########################################################
#------------ SEQUENTIAL STAGE PROGRESSION --------------
#########################################################

def test_full_stage_progression(stage_fsm, perfect_scores, tool_perfect_scores):
    
    # Stage A
    trajA = generate_circular_traj(tool="NONE")
    stage_fsm.evaluate(trajA, perfect_scores, tool_perfect_scores)

    # Stage B
    trajB = generate_circular_traj(tool="ScrewdriverL")
    stage_fsm.evaluate(trajB, perfect_scores, tool_perfect_scores)

    # Stage C
    trajC = generate_circular_traj(tool="NONE")
    result = stage_fsm.evaluate(trajC, perfect_scores, tool_perfect_scores)

    assert result["task_complete"] is True


def _unordered_traj(x: float, n: int = 10):
    return [{"x": x, "y": 0.2 + 0.01 * i, "t": 0.05 * i, "tool": "NONE"} for i in range(n)]


def _unordered_stage_fsm(required_count=2, lockout=True):
    cfg = [
        {
            "name": "StageU",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": required_count,
            "lockout_enabled": lockout,
        },
        {
            "name": "StageNext",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "ordered",
        },
    ]
    fsm = StageFSM(stage_config=cfg, load_references=False)
    fsm.stage_references = {
        "StageU": [_unordered_traj(0.1), _unordered_traj(0.9)],
        "StageNext": [_unordered_traj(0.5)],
    }
    return fsm


def test_unordered_stage_order_invariant_completion():
    fsm = _unordered_stage_fsm(required_count=2, lockout=True)
    # Reverse order on purpose: ref2 then ref1.
    r1 = fsm.evaluate(_unordered_traj(0.9))
    assert r1["stage_name"] == "StageU"
    assert r1["group_progress"]["completed_count"] == 1
    assert r1["next_stage"] is None
    assert fsm.current_stage_index == 0

    r2 = fsm.evaluate(_unordered_traj(0.1))
    assert r2["stage_name"] == "StageU"
    assert r2["group_progress"]["completed_count"] == 2
    assert r2["group_progress"]["group_complete"] is True
    assert r2["next_stage"] == "StageNext"
    assert fsm.current_stage_index == 1


def test_unordered_stage_duplicate_prevention_with_lockout():
    fsm = _unordered_stage_fsm(required_count=2, lockout=True)
    _ = fsm.evaluate(_unordered_traj(0.1))
    dup = fsm.evaluate(_unordered_traj(0.1))

    assert dup["stage_passed"] is False
    assert dup["stage_name"] == "StageU"
    if "group_progress" in dup:
        assert dup["group_progress"]["completed_count"] == 1
    assert fsm.current_stage_index == 0


def test_unordered_stage_advances_only_at_required_count():
    fsm = _unordered_stage_fsm(required_count=2, lockout=True)
    one = fsm.evaluate(_unordered_traj(0.1))
    assert one["group_progress"]["group_complete"] is False
    assert fsm.current_stage_index == 0

    two = fsm.evaluate(_unordered_traj(0.9))
    assert two["group_progress"]["group_complete"] is True
    assert fsm.current_stage_index == 1


def test_unordered_stage_margin_gate_logging():
    cfg = [
        {
            "name": "StageU",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 1,
            "lockout_enabled": True,
            "margin_min": 0.2,
        }
    ]
    fsm = StageFSM(stage_config=cfg, load_references=False)
    # Similar references to force a small top-2 margin.
    ref1 = _unordered_traj(0.4)
    ref2 = _unordered_traj(0.42)
    fsm.stage_references = {"StageU": [ref1, ref2]}

    result = fsm.evaluate(_unordered_traj(0.41))
    assert result["stage_name"] == "StageU"
    assert "best_match_score" in result
    assert "second_best_score" in result
    assert "margin" in result
    gp = result.get("group_progress", {})
    assert gp.get("margin_min") == 0.2
    assert gp.get("margin_gate_ok") is False
    assert gp.get("reject_reason") is not None


def test_unordered_stage_roi_guard_blocks_wrong_area():
    cfg = [
        {
            "name": "StageU",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 1,
            "lockout_enabled": True,
            "subtask_rois": {0: [0.0, 0.0, 0.2, 0.2], 1: [0.8, 0.0, 1.0, 0.2]},
        }
    ]
    fsm = StageFSM(stage_config=cfg, load_references=False)
    fsm.stage_references = {"StageU": [_unordered_traj(0.1), _unordered_traj(0.9)]}

    result = fsm.evaluate(_unordered_traj(0.5))
    gp = result.get("group_progress", {})
    assert gp.get("roi_gate_required") is True
    assert gp.get("roi_gate_ok") is False
    assert gp.get("group_complete") is False


def test_unordered_stage_per_subtask_cooldown():
    cfg = [
        {
            "name": "StageU",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 2,
            "lockout_enabled": False,
            "cooldown_attempts": 1,
        }
    ]
    fsm = StageFSM(stage_config=cfg, load_references=False)
    # Use one dominant reference and keep repeating same motion.
    fsm.stage_references = {"StageU": [_unordered_traj(0.1), _unordered_traj(0.9)]}

    r1 = fsm.evaluate(_unordered_traj(0.1))
    assert r1["group_progress"]["completed_count"] == 1

    r2 = fsm.evaluate(_unordered_traj(0.1))
    assert r2["group_progress"]["guard_blocked"] is True
    assert r2["group_progress"]["completed_count"] == 1


def test_unordered_stage_subtask_group_lockout_excludes_all_refs_in_group():
    cfg = [
        {
            "name": "StageU",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 2,
            "lockout_enabled": True,
            "subtask_groups": {"A1": [0, 1], "A2": [2]},
        }
    ]
    fsm = StageFSM(stage_config=cfg, load_references=False)
    # ref0/ref1 belong to same physical subtask A1
    fsm.stage_references = {"StageU": [_unordered_traj(0.10), _unordered_traj(0.12), _unordered_traj(0.9)]}

    first = fsm.evaluate(_unordered_traj(0.10))
    assert first["matched_subtask_id"] == "A1"
    assert first["matched_reference_id"] in [0, 1]
    assert first["completed_subtasks"] == ["A1"]

    second = fsm.evaluate(_unordered_traj(0.90))
    # A1 should lock out both ref0 and ref1; next match should move to A2.
    assert 0 in second["excluded_reference_indices"]
    assert 1 in second["excluded_reference_indices"]
    assert second["matched_subtask_id"] == "A2"
    assert second["group_progress"]["group_complete"] is True
