import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest
from assembly_assistant.fsm.stage_fsm import StageFSM
from assembly_assistant.fsm.stage_config import SYNTHETIC_STAGE_CONFIG
from data.mocks.trajectory_generator import generate_circular_traj
from assembly_assistant.utils.metadata_manager import MetadataManager
from assembly_assistant.utils.experiment_logger import ExperimentLogger


@pytest.fixture(autouse=True)
def _isolate_experiment_output(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOA_EXPERIMENTS_ROOT", str(tmp_path / "experiments"))


@pytest.fixture(autouse=True)
def _reset_singletons():
    MetadataManager.reset_instance()
    ExperimentLogger.reset_instance()
    yield
    MetadataManager.reset_instance()
    ExperimentLogger.reset_instance()


@pytest.fixture
def valid_hand():
    keypoints = np.zeros((21, 2))
    keypoints[0] = [0.3, 0.3]   #wrist
    keypoints[5] = [0.35, 0.35]
    return {
        "keypoints": keypoints,
        "velocity": 0.02
    }

@pytest.fixture
def valid_tool():
    return {
        "class": "ScrewdriverL",
        "bbox": (0.28, 0.28, 0.32, 0.32),
        "confidence": 0.9,
        "timestamp": 1.0
    }

@pytest.fixture
def reference_traj():
    return [
        {"x": 0.1, "y": 0.2, "t": 0.0, "tool":"NONE"},
        {"x": 0.2, "y": 0.3, "t": 0.1, "tool":"NONE"},
        {"x": 0.3, "y": 0.4, "t": 0.2, "tool":"NONE"}
    ]

@pytest.fixture
def stage_fsm():
    stage_fsm = StageFSM(SYNTHETIC_STAGE_CONFIG, load_references=False)

    # inject references manually
    stage_fsm.stage_references = {
        "StageA_circular" : [generate_circular_traj(tool="NONE")],
        "StageB_Linear" : [generate_circular_traj(tool="ScrewdriverL")],
        "StageC_Sinusoidal" : [generate_circular_traj(tool="NONE")]
    }
    return stage_fsm

@pytest.fixture
def perfect_scores():
    return {
        "similarity_fastdtw": 0.95
    }

@pytest.fixture
def weak_scores():
    return {
        "similarity_fastdtw": 0.3
    }

@pytest.fixture
def tool_perfect_scores():
    return {
        "tool_match_score": 0.95
    }

@pytest.fixture
def tool_weak_scores():
    return {
        "tool_match_score": 0.25
    }
