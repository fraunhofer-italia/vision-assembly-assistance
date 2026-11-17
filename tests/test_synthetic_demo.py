from scripts.main import run_demo


def test_camera_free_demo_segments_and_verifies_both_stages():
    result = run_demo()
    assert result["synthetic"] is True
    assert result["task_complete"] is True
    assert len(result["stages"]) == 2
    for stage in result["stages"]:
        assert stage["passed"] is True
        assert stage["points"] == 40
        assert stage["dtw_similarity"] > 0.99
        assert stage["transitions"] == [
            {"from": "IDLE", "to": "RECORDING"},
            {"from": "RECORDING", "to": "DONE"},
        ]
