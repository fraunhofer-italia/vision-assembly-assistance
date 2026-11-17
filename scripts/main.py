"""Deterministic camera-free MotionFSM segmentation and DTW scoring example.

This synthetic example is not a reproduction of the study's perception path.
It writes its summary to stdout and does not load data or model assets.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from assembly_assistant.fsm.motion_fsm import MotionFSM
from assembly_assistant.fsm.stage_fsm import StageFSM
from assembly_assistant.synchronization.sync_manager import SyncManager
from assembly_assistant.trajectory.trajectory_builder import TrajectoryBuilder
from data.mocks.synthetic_trajectories import generate_expert_trajectory


def run_demo() -> dict:
    stages = [{"name": f"Synthetic_{i}", "expected_tools": ["NONE"], "score_threshold": 0.8}
              for i in (1, 2)]
    references = {stage["name"]: [generate_expert_trajectory(n_points=40, tool="NONE")]
                  for stage in stages}
    stage_fsm = StageFSM(stages, weights={"dtw": 1.0}, stage_references=references,
                         load_references=False)
    summaries = []
    for stage in stages:
        motion = MotionFSM(work_region=(0.05, 0.05, 0.95, 0.95),
                           drop_region=(0.85, 0.0, 1.0, 0.2), warmup_frames=0,
                           min_consecutive_visible=1, start_no_activity_frames=1,
                           min_consecutive_drop_hits=1, min_recording_frames=3,
                           max_hand_jump=1.0)
        sync = SyncManager()
        builder = TrajectoryBuilder()
        transitions = []
        reference = references[stage["name"]][0]
        packets = reference + [{"x": 0.9, "y": 0.1, "t": reference[-1]["t"] + 0.05}]
        for frame_id, point in enumerate(packets, start=1):
            keypoints = np.zeros((21, 2))
            keypoints[0] = keypoints[5] = (point["x"], point["y"])
            frame = sync.sync(hand_data={"keypoints": keypoints, "velocity": 0.02},
                              tool_data=None, timestamp=point["t"], frame_id=frame_id)
            previous = motion.state
            motion.update(frame)
            if motion.state != previous:
                transitions.append({"from": previous, "to": motion.state})
            if motion.state == motion.RECORDING:
                trajectory_point = builder.build_trajectory_point(frame)
                if trajectory_point is not None:
                    builder.append(trajectory_point)
            if motion.state == motion.DONE:
                break
        trajectory = builder.get_trajectory()
        if motion.state != motion.DONE or not trajectory:
            raise RuntimeError("Synthetic motion did not produce a complete segment")
        feedback = stage_fsm.evaluate(trajectory, use_tool=False)
        if not feedback.get("stage_passed"):
            raise RuntimeError("Synthetic reference did not pass the demo stage")
        summaries.append({"stage": stage["name"], "points": len(trajectory),
                          "dtw_similarity": feedback["dtw_similarity"],
                          "passed": feedback["stage_passed"], "transitions": transitions})
    return {"synthetic": True, "task_complete": stage_fsm.is_task_complete(), "stages": summaries}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
