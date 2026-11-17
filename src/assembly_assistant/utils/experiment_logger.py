######################################################
######  Records runtime evaluation data      #########
######  Needed because :                     #########
######  1) Compare DTW thresholds            #########
######  2) Tune FSM transitions              #########
######  3) Analyse failures                  #########
######  4) Provide quantative research results  ######
######  5) Debug false positives             #########
#################################################################
###### MOTION FSM --> BUILDS TRAJECTORY                 #########
###### STAGE FSM --> EVALUATES TRAJECTORY               #########
###### STAGE FSM --> SENDS RESULTS --> EXPERIMENTLOGGER       ###
###### EXPERIMENTLOGGER --> WRITES JSON                 #########
###### METADATA MANAGER --> STORES EXPERIMENTS --> LEVEL INFO ###
#################################################################

import json
import os
from datetime import datetime
from assembly_assistant.utils.paths import get_experiments_root

class ExperimentLogger:

    _instance = None

    #----------------------------------------
    #------- SINGLETON ACCESS ---------------
    #----------------------------------------

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance.reset_state()
        cls._instance = None
    
    #----------------------------------------
    #------- CONSTRUCTOR --------------------
    #----------------------------------------

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.experiment_name = None
        self.log_path = None
        self.logs = {}

    #----------------------------------------
    #------- START EXPERIMENT ---------------
    #----------------------------------------

    def start_experiment(self, experiment_name, root=None):
        self.experiment_name = experiment_name

        if root is None:
            root = str(get_experiments_root())
        exp_dir = os.path.join(root, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        self.log_path = os.path.join(exp_dir, "experiment_log.json")

        # Load existing if present
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                self.logs = json.load(f)

        else:
            self.logs = {
                "experiment_name": experiment_name,
                "trajectories": {}
            }
            self._save()


    #----------------------------------------
    #------- LOG STAGE DECISION -------------
    #----------------------------------------

    def log_stage_decision(
            self,
            trajectory_id,
            stage_name,
            reference_id,
            dtw_distance,
            stage_passed,
            fsm_transitions,
            tool_confidence_timeline,
            dtw_alignment_path=None,
            observed_traj=None,
            reference_traj=None,
            user_id=None,
    ):
        timestamp = datetime.utcnow().isoformat()

        observed_path = None
        reference_path = None

        if self.log_path is not None and (observed_traj is not None or reference_traj is not None):
            from uuid import uuid4
            import numpy as np

            exp_dir = os.path.dirname(self.log_path)
            short_id = str(uuid4())[:8]
            safe_user = user_id or "unknown"

            if observed_traj is not None:
                observed_path = os.path.join(
                    exp_dir,
                    f"trajectory_{stage_name}_{safe_user}_{short_id}_observed.npy"
                )
                np.save(observed_path, observed_traj, allow_pickle=True)

            if reference_traj is not None:
                reference_path = os.path.join(
                    exp_dir,
                    f"trajectory_{stage_name}_{safe_user}_{short_id}_reference.npy"
                )
                np.save(reference_path, reference_traj, allow_pickle=True)

        # Ensure trajectory exists
        if trajectory_id not in self.logs["trajectories"]:
            self.logs["trajectories"][trajectory_id] = {
                "stages": []
            }

        entry = {
            "stage_name": stage_name,
            "reference_id": reference_id,
            "dtw_distance": dtw_distance,          
            "stage_passed": stage_passed,
            "fsm_transitions": fsm_transitions,
            "tool_confidence_timeline": tool_confidence_timeline,
            "dtw_alignment_path": dtw_alignment_path,
            "timestamp": timestamp,
            "observed_traj_path": observed_path,
            "reference_traj_path": reference_path
        }

        self.logs["trajectories"][trajectory_id]["stages"].append(entry)

        self._save()

    #----------------------------------------
    #------- SAVE IMMEDIATELY ---------------
    #----------------------------------------

    def _save(self):

        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=4)
