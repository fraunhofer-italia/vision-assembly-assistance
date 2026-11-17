
import os
import tempfile
from assembly_assistant.utils.experiment_logger import ExperimentLogger

TEST_ESP = "test_experiment_logger"

def setup_function():
    test_root = tempfile.mkdtemp(prefix="sloa_exp_logger_")
    os.environ["SLOA_EXPERIMENTS_ROOT"] = test_root
    ExperimentLogger.reset_instance()

def test_logger_creation():

    logger = ExperimentLogger.get_instance()
    logger.start_experiment(TEST_ESP)

    root = os.environ["SLOA_EXPERIMENTS_ROOT"]
    assert os.path.exists(f"{root}/{TEST_ESP}/experiment_log.json")

def test_log_stage_entry():

    logger = ExperimentLogger.get_instance()
    logger.start_experiment(TEST_ESP)

    logger.log_stage_decision(
        trajectory_id="traj1",
        stage_name="StageA",
        reference_id="ref1",
        dtw_distance=0.3,
        stage_passed=True,
        fsm_transitions=["IDLE -> REC", "REC -> DONE"],
        tool_confidence_timeline=[],
        dtw_alignment_path=[(0,0), (1,1)]
    )

    assert "traj1" in logger.logs["trajectories"]
    assert len(logger.logs["trajectories"]["traj1"]["stages"]) == 1

def test_persistence():

    logger = ExperimentLogger.get_instance()
    logger.start_experiment(TEST_ESP)

    logger.log_stage_decision(
        "traj1", "StageA", "ref1",
        0.2, True, [], []
    )

    # Reload
    logger2 = ExperimentLogger.get_instance()
    logger2.start_experiment(TEST_ESP)

    assert "traj1" in logger2.logs["trajectories"]
