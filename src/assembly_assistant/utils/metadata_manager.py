######################################################
######  Store Experiment Context + Configurations ####
######  Needed because :                     #########
######  1) Experiments must be reproducible  #########
######  2) DTW parameters/smoothing setting change ###
######  3) Tool detection confidence thresholds change
######  4) Need traceability                    ######
######################################################

import json
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from assembly_assistant.utils.paths import get_experiments_root

class MetadataManager:
    _instance = None

    #------------------------------------------------
    #---  SINGLETON ACCESS --------------------------
    #------------------------------------------------

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MetadataManager()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance.reset_state()
        cls._instance = None
    
    #------------------------------------------------
    #------  CONSTRUCTOR ----------------------------
    #------------------------------------------------

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.experiment_path: Optional[Path] = None
        self.metadata = {}
        self._finalized = False

    #------------------------------------------------
    #------ CREATE EXPERIMENT (versioning) ----------
    #------------------------------------------------

    def create_experiment(
            self,
            experiment_name: str,
            user_id: str,
            session_id: str,
            references: List[str],
            config: Optional[dict] = None,
            session_meta: Optional[dict] = None,
            protocol_meta: Optional[dict] = None,
            root: Optional[str] = None,
    ):
        if self.experiment_path is not None:
            raise RuntimeError("Experiment already created.")
        
        base_dir = Path(root) if root is not None else get_experiments_root()
        base_dir.mkdir(exist_ok=True)

        exp_path = self._create_versioned_folder(base_dir, experiment_name)

        self.experiment_path = exp_path

        session_payload = {
            "user_id": user_id,
            "session_id": session_id,
        }
        if session_meta:
            session_payload.update(session_meta)

        self.metadata = {
            "experiment_name": exp_path.name,
            "created_at": datetime.utcnow().isoformat(),
            "session": session_payload,
            "references": references,
            "system_config": config or {},
            "protocol": protocol_meta or {},
        }

        self._save()

    #----------------------------------------------------
    #------ VERSIONING LOGIC ----------------------------
    #----------------------------------------------------

    def _create_versioned_folder(self, base_dir: Path, name: str) -> Path:
        version = 1
        while True:
            candidate = base_dir / f"{name}_v{version}"
            if not candidate.exists():
                candidate.mkdir()
                return candidate
            version += 1

    #----------------------------------------------------
    #--------------- UPDATE METADATA --------------------
    #----------------------------------------------------

    def update(self, key: str, value):
        if self._finalized:
            raise RuntimeError("Metadata already finalized.")
        
        self.metadata[key] = value
        self._save()

    #----------------------------------------------------
    #--------- ADD FSM DECISION METADATA ----------------
    #----------------------------------------------------

    def add_fsm_decision(self, decision_data: dict):
        if self._finalized:
            raise RuntimeError("Metadata already finalized")

        # Keep alignment path only in experiment_log.json, not in metadata.json.
        sanitized = copy.deepcopy(decision_data)
        feedback = sanitized.get("feedback")
        if isinstance(feedback, dict):
            feedback.pop("alignment_path", None)

        # Normalize unordered-group tracking fields for stable JSON schema.
        for k in ("completed_subtasks", "remaining_subtasks"):
            if k in sanitized and sanitized[k] is None:
                sanitized[k] = []
            if isinstance(sanitized.get(k), set):
                sanitized[k] = sorted(list(sanitized[k]))
            if k in sanitized and not isinstance(sanitized[k], list):
                sanitized[k] = [sanitized[k]]

        if "group_progress" in sanitized and sanitized["group_progress"] is not None:
            if not isinstance(sanitized["group_progress"], dict):
                sanitized["group_progress"] = {"value": sanitized["group_progress"]}

        self.metadata.setdefault("fsm_decisions", []).append(sanitized)
        self._save()

    #-----------------------------------------------------
    #-------------------- FINALIZE METADATA --------------
    #-----------------------------------------------------
    
    def finalize(self):
        self.metadata["finalized_at"] = datetime.utcnow().isoformat()
        self._finalized = True
        self._save()

    #------------------------------------------------------
    #------------ SAVE FILE -------------------------------
    #------------------------------------------------------

    def _save(self):
        if self.experiment_path is None:
            return
        
        path = self.experiment_path / "metadata.json"

        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=4)
