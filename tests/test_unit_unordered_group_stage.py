from assembly_assistant.fsm.stage_fsm import StageFSM


def _traj(x: float, n: int = 8):
    return [{"x": x, "y": 0.2 + 0.01 * i, "t": 0.05 * i, "tool": "NONE"} for i in range(n)]


def test_unordered_group_progress_and_advance():
    stage_config = [
        {
            "name": "StageA",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 2,
            "lockout_enabled": True,
        },
        {
            "name": "StageB",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "ordered",
        },
    ]
    fsm = StageFSM(stage_config=stage_config, load_references=False)
    fsm.stage_references = {"StageA": [_traj(0.1), _traj(0.9)], "StageB": [_traj(0.5)]}

    first = fsm.evaluate(_traj(0.1))
    assert first["stage_name"] == "StageA"
    assert first["stage_passed"] is True
    assert first["group_progress"]["completed_count"] == 1
    assert first["group_progress"]["group_complete"] is False
    assert fsm.current_stage_index == 0

    second = fsm.evaluate(_traj(0.9))
    assert second["stage_name"] == "StageA"
    assert second["stage_passed"] is True
    assert second["group_progress"]["completed_count"] == 2
    assert second["group_progress"]["group_complete"] is True
    assert second["next_stage"] == "StageB"
    assert fsm.current_stage_index == 1


def test_unordered_group_lockout_prevents_recount():
    stage_config = [
        {
            "name": "StageA",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 2,
            "lockout_enabled": True,
        }
    ]
    fsm = StageFSM(stage_config=stage_config, load_references=False)
    fsm.stage_references = {"StageA": [_traj(0.1), _traj(0.9)]}

    _ = fsm.evaluate(_traj(0.1))
    repeat = fsm.evaluate(_traj(0.1))
    assert repeat["stage_passed"] is False
    assert "recovery" in repeat


def test_unordered_group_mode1_dtw_scores_resolves_reference_and_progresses():
    stage_config = [
        {
            "name": "StageA",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 2,
            "lockout_enabled": True,
        },
        {
            "name": "StageB",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "ordered",
        },
    ]
    fsm = StageFSM(stage_config=stage_config, load_references=False)
    fsm.stage_references = {"StageA": [_traj(0.1), _traj(0.9)], "StageB": [_traj(0.5)]}

    # Mode-1 input: precomputed similarity only (no reference_id).
    dtw_scores = {"similarity_fastdtw": 0.95}
    tool_score = {"tool_match_score": 1.0}

    first = fsm.evaluate(_traj(0.1), dtw_scores=dtw_scores, tool_score=tool_score)
    assert first["stage_passed"] is True
    assert first["reference_id"] in (0, 1)
    assert first["matched_subtask_id"] is not None
    assert first["group_progress"]["completed_count"] == 1
    assert first["group_progress"]["group_complete"] is False
    assert fsm.current_stage_index == 0

    second = fsm.evaluate(_traj(0.9), dtw_scores=dtw_scores, tool_score=tool_score)
    assert second["stage_passed"] is True
    assert second["reference_id"] in (0, 1)
    assert second["matched_subtask_id"] is not None
    assert second["group_progress"]["completed_count"] == 2
    assert second["group_progress"]["group_complete"] is True
    assert second["next_stage"] == "StageB"
    assert fsm.current_stage_index == 1
