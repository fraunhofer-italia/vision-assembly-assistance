from assembly_assistant.similarity.dtw_engine import DTWEngine
from assembly_assistant.fsm.stage_fsm import StageFSM

from data.mocks.trajectory_generator import (
    generate_circular_traj,
    add_spatial_noise, add_temporal_jitter, add_tool_noise
)

def test_stageA_dtw_integration(stage_fsm):

    dtw_engine = DTWEngine()

    reference = generate_circular_traj(tool="NONE")
    observed = add_tool_noise(add_spatial_noise(add_temporal_jitter(reference)), 0.3)

    scores = dtw_engine.compare_multiple(
        observed_traj=observed,
        reference_traj=reference,
        use_tool=True
    )

    result = stage_fsm.evaluate(
        observed_traj=observed,
        dtw_scores={"similarity_fastdtw": scores["similarity_fastdtw"]},
        tool_score={"tool_match_score": scores["tool_match_score"]},
    )

    assert result["stage_passed"] is False


def test_unordered_group_integration_unique_completion_flow():
    cfg = [
        {
            "name": "StageU",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "unordered_group",
            "required_count": 2,
            "lockout_enabled": True,
        },
        {
            "name": "StageN",
            "expected_tools": ["NONE"],
            "score_threshold": 0.7,
            "stage_type": "ordered",
        },
    ]
    fsm = StageFSM(stage_config=cfg, load_references=False)
    ref_a = generate_circular_traj(tool="NONE")
    ref_b = generate_circular_traj(tool="NONE")
    # Spatially shift one reference so the best-match identity is deterministic.
    for p in ref_b:
        p["x"] = min(1.0, p["x"] + 0.25)
    fsm.stage_references = {"StageU": [ref_a, ref_b], "StageN": [generate_circular_traj(tool="NONE")]}

    # Use exact references so both attempts are accepted and progression logic is tested.
    obs_b = ref_b
    obs_a = ref_a

    r1 = fsm.evaluate(observed_traj=obs_b)
    assert r1["stage_name"] == "StageU"
    assert r1["group_progress"]["completed_count"] == 1
    assert fsm.current_stage_index == 0

    r2 = fsm.evaluate(observed_traj=obs_a)
    assert r2["stage_name"] == "StageU"
    assert r2["group_progress"]["completed_count"] == 2
    assert r2["group_progress"]["group_complete"] is True
    assert r2["next_stage"] == "StageN"
    assert fsm.current_stage_index == 1
