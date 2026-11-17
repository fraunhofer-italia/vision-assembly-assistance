from scripts.main_full_pipeline import _per_stage_run_should_continue


def test_per_stage_continue_for_ordered_pass():
    feedback = {"stage_passed": True, "stage_type": "ordered"}
    assert _per_stage_run_should_continue(feedback) is True


def test_per_stage_stop_for_unordered_pass_but_incomplete():
    feedback = {
        "stage_passed": True,
        "stage_type": "unordered_group",
        "stage_complete": False,
        "group_progress": {"group_complete": False},
    }
    assert _per_stage_run_should_continue(feedback) is False


def test_per_stage_continue_for_unordered_group_complete():
    feedback = {
        "stage_passed": True,
        "stage_type": "unordered_group",
        "stage_complete": True,
        "group_progress": {"group_complete": True},
    }
    assert _per_stage_run_should_continue(feedback) is True
