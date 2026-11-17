import tempfile
import os
from pathlib import Path

from assembly_assistant.utils.metadata_manager import MetadataManager

def setup_function():
    test_root = tempfile.mkdtemp(prefix="sloa_exp_meta_")
    os.environ["SLOA_EXPERIMENTS_ROOT"] = test_root
    MetadataManager.reset_instance()

def test_create_experiment():
    manager = MetadataManager.get_instance()

    manager.create_experiment(
        experiment_name="test_exp",
        user_id="user1",
        session_id="sess1",
        references=["ref1.npy"],
    )

    exp_dir = Path(os.environ["SLOA_EXPERIMENTS_ROOT"])
    assert exp_dir.exists()

    folders = list(exp_dir.iterdir())
    assert len(folders) == 1

    metadata_file = folders[0] / "metadata.json"
    assert metadata_file.exists()

def test_versioning():
    manager = MetadataManager.get_instance()

    manager.create_experiment(
        "exp",
        "u1",
        "s1",
        ["ref.npy"]
    )

    # Reset singleton to simulate new run

    MetadataManager.reset_instance()
    manager2 = MetadataManager.get_instance()

    manager2.create_experiment(
        "exp",
        "u1",
        "s2",
        ["ref.npy"]
    )

    folders = sorted(Path(os.environ["SLOA_EXPERIMENTS_ROOT"]).iterdir())

    assert folders[0].name.endswith("_v1")
    assert folders[1].name.endswith("_v2")

def test_add_fsm_decision():
    manager = MetadataManager.get_instance()

    manager.create_experiment(
        "fsm_test",
        "user1",
        "sess1",
        ["ref.npy"]
    )

    manager.add_fsm_decision({
        "stage": "unscrew",
        "result": "success"
    })

    assert "fsm_decisions" in manager.metadata
    assert len(manager.metadata["fsm_decisions"]) == 1

def test_finalize_blocks_updates():
    manager = MetadataManager.get_instance()

    manager.create_experiment(
        "finalize_test",
        "user1",
        "sess1",
        ["ref.npy"]
    )

    manager.finalize()

    try: 
        manager.update("new_key", 123)
        assert False
    except RuntimeError:
        assert True
