from assembly_assistant.similarity.dtw_engine import DTWEngine

def test_extract_points_no_tool(reference_traj):
    dtw = DTWEngine()

    pts = dtw._extract_points(reference_traj, use_tool=False)

    assert pts.shape[1] == 2

def test_extract_points_with_tool(reference_traj):
    dtw = DTWEngine()

    pts = dtw._extract_points(reference_traj, use_tool=True)

    assert pts.shape[1] == 3

def test_compare_multiple(reference_traj):
    dtw =DTWEngine()

    scores = dtw.compare_multiple(reference_traj, reference_traj)

    assert "fastdtw" in scores
    assert scores["similarity_fastdtw"] is not None
    assert scores["tool_match_score"] is not None


def test_tool_match_score_uses_alignment_path():
    dtw = DTWEngine()
    # Observed is shorter; alignment maps both reference points to one observed point.
    observed = [
        {"x": 0.0, "y": 0.0, "t": 0.0, "tool": "ScrewdriverL"},
    ]
    reference = [
        {"x": 0.0, "y": 0.0, "t": 0.0, "tool": "ScrewdriverL"},
        {"x": 0.1, "y": 0.1, "t": 0.1, "tool": "ScrewdriverL"},
    ]
    score = dtw._tool_match_score(observed, reference, alignment_path=[(0, 0), (0, 1)])
    assert score == 1.0
