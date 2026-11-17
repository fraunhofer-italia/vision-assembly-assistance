import numpy as np
from assembly_assistant.detection_types.label_taxonomy import canonical_label
from assembly_assistant.analysis.error_taxonomy import classify_failure
from assembly_assistant.fsm.reference_manager import ReferenceManager
from assembly_assistant.fsm.recovery_manager import RecoveryManager
from assembly_assistant.analysis.fatigue_metrics import compute_phase1_fatigue_metrics
from assembly_assistant.trajectory.preprocessing import TrajectoryPreprocessor

class StageFSM:

    def __init__(self, stage_config, weights=None, stage_references=None, load_references=True):
        """
        stage_config: list of stage dictionaries
        weights: {"dtw": float, "tool": float}
        """

        self.stages = stage_config
        self.current_stage_index = 0
        self.load_references = load_references

        self.weights = weights or {
            "dtw": 0.65,
            "tool": 0.0,
            "active_part_match": 0.10,
            "active_tool_match": 0.10,
            "expected_part_match": 0.05,
            "expected_tool_match": 0.05,
            "smoothness": 0.03,
            "efficiency": 0.01,
            "temporal_stability": 0.01,
        }

        self.reference_manager = ReferenceManager()
        self.recovery_manager = RecoveryManager()
        self.group_progress = {}

        # cache references
        self.stage_references = stage_references or {}
        if self.load_references:
            self.stage_references = self._load_all_references()

    #---------------------------------------------------------#

    def _load_all_references(self):

        refs = {}

        for stage in self.stages:
            ref_paths = stage.get("reference_paths") or [stage["reference_path"]]

            raw_refs = self.reference_manager.load_references(ref_paths)

            processed_refs = [
                TrajectoryPreprocessor.preprocess(
                    ref,
                    oob_policy={"mode": "none"},
                )
                for ref in raw_refs
            ]

            refs[stage["name"]] = processed_refs

        return refs

    @staticmethod
    def _trajectory_centroid(observed_traj):
        if not observed_traj:
            return None
        xs = [float(p.get("x", 0.0)) for p in observed_traj if "x" in p]
        ys = [float(p.get("y", 0.0)) for p in observed_traj if "y" in p]
        if not xs or not ys:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)))

    @staticmethod
    def _normalize_label_set(values):
        if values is None:
            return set()
        if isinstance(values, str):
            values = [values]
        return {str(v).strip() for v in values if str(v).strip()}

    @classmethod
    def _active_match_score(cls, observed_traj, *, field_name: str, candidates) -> float:
        candidate_set = cls._normalize_label_set(candidates)
        if not candidate_set:
            return 1.0
        labels = [str(pt.get(field_name, "NONE")).strip() for pt in (observed_traj or [])]
        if not labels:
            return 1.0 if candidate_set == {"NONE"} else 0.0
        if candidate_set == {"NONE"}:
            non_none = [label for label in labels if label and label != "NONE"]
            return 1.0 if not non_none else 0.0
        return 1.0 if any(label in candidate_set for label in labels) else 0.0

    @classmethod
    def _expected_inventory_match_score(
        cls,
        observed_traj,
        *,
        expected_labels,
        expected_counts=None,
    ) -> float:
        label_set = cls._normalize_label_set(expected_labels)
        counts = {str(k).strip(): int(v) for k, v in (expected_counts or {}).items() if str(k).strip()}
        if not label_set and not counts:
            return 1.0

        # Special case: non-tool/non-part expected absence.
        if label_set == {"NONE"} and not counts:
            for pt in observed_traj or []:
                inventory = dict(pt.get("workspace_inventory") or {})
                positives = {k: int(v) for k, v in inventory.items() if int(v) > 0 and str(k).strip() != "NONE"}
                if not positives:
                    return 1.0
            return 0.0

        best = 0.0
        for pt in observed_traj or []:
            inventory = dict(pt.get("workspace_inventory") or {})
            if counts:
                total = 0
                matched = 0
                for label, need in counts.items():
                    if label == "NONE":
                        continue
                    total += max(0, int(need))
                    matched += min(max(0, int(need)), max(0, int(inventory.get(label, 0))))
                score = 1.0 if total <= 0 else float(matched) / float(total)
            else:
                relevant = [label for label in label_set if label != "NONE"]
                if not relevant:
                    score = 1.0
                else:
                    present = sum(1 for label in relevant if int(inventory.get(label, 0)) > 0)
                    score = float(present) / float(len(relevant))
            best = max(best, float(score))
        return float(best)

    @staticmethod
    def _windowed_points(observed_traj, temporal_window: str):
        points = list(observed_traj or [])
        if not points:
            return []
        window = str(temporal_window or "late").strip().lower()
        if window == "all":
            return points
        n = len(points)
        third = max(1, n // 3)
        if window == "early":
            return points[:third]
        if window == "middle":
            return points[third : min(n, 2 * third)]
        return points[min(n, 2 * third) :]

    @classmethod
    def _expected_pose_match_score(cls, observed_traj, expected_poses) -> tuple[float, list[dict]]:
        expectations = list(expected_poses or [])
        if not expectations:
            return 1.0, []

        results = []
        for exp in expectations:
            part_label = canonical_label(exp.get("part_label", exp.get("part", "")))
            expected_pose = str(exp.get("expected_pose", exp.get("pose", ""))).strip()
            temporal_window = str(exp.get("temporal_window", "late")).strip().lower()
            min_confidence = float(exp.get("min_confidence", 0.50))
            min_match_ratio = float(exp.get("min_match_ratio", 0.50))
            window_points = cls._windowed_points(observed_traj, temporal_window)
            considered = []
            matched = 0
            for pt in window_points:
                for pred in pt.get("pose_predictions", []) or []:
                    pred_part = canonical_label(pred.get("part_label", pred.get("part", "")))
                    if pred_part != part_label:
                        continue
                    try:
                        conf = float(pred.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0
                    if conf < min_confidence:
                        continue
                    class_name = str(pred.get("class_name", pred.get("pose", ""))).strip()
                    considered.append(
                        {
                            "class_name": class_name,
                            "confidence": conf,
                            "frame_id": pred.get("frame_id", pt.get("frame_id")),
                        }
                    )
                    if class_name == expected_pose:
                        matched += 1

            total = len(considered)
            ratio = float(matched) / float(total) if total else 0.0
            results.append(
                {
                    "part_label": part_label,
                    "expected_pose": expected_pose,
                    "temporal_window": temporal_window,
                    "min_confidence": min_confidence,
                    "min_match_ratio": min_match_ratio,
                    "observed_predictions": total,
                    "matched_predictions": int(matched),
                    "match_ratio": ratio,
                    "passed": bool(ratio >= min_match_ratio),
                    "samples": considered[:10],
                }
            )

        if not results:
            return 1.0, []
        score = float(np.mean([float(r["match_ratio"]) for r in results]))
        return score, results

    @staticmethod
    def _point_in_roi(point, roi):
        if point is None or roi is None:
            return False
        x, y = point
        x_min, y_min, x_max, y_max = roi
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    @staticmethod
    def _normalize_subtask_rois(subtask_rois):
        """
        Accepts:
        - dict: {0:[x1,y1,x2,y2], "1":[...]}
        - list: [[...], [...], ...] where index is subtask/reference id
        """
        if subtask_rois is None:
            return {}
        out = {}
        if isinstance(subtask_rois, dict):
            iterator = subtask_rois.items()
        elif isinstance(subtask_rois, list):
            iterator = enumerate(subtask_rois)
        else:
            return {}
        for k, v in iterator:
            try:
                # keep both numeric and string keys (e.g. "S1")
                key = int(k) if str(k).isdigit() else str(k)
                if v is None:
                    continue
                roi = [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
                out[key] = roi
            except Exception:
                continue
        return out

    @staticmethod
    def _normalize_subtask_groups(stage, num_refs):
        """
        Returns:
        - subtask_to_refs: {subtask_id(str): [ref_idx,...]}
        - ref_to_subtask: {ref_idx: subtask_id}
        """
        raw = stage.get("subtask_groups")
        if not raw:
            subtask_to_refs = {f"S{i + 1}": [i] for i in range(num_refs)}
        else:
            subtask_to_refs = {}
            if isinstance(raw, dict):
                items = raw.items()
            else:
                raise ValueError("subtask_groups must be a dict{subtask_id: [reference_indices]}.")
            for sid, refs in items:
                sid_norm = str(sid)
                if not isinstance(refs, (list, tuple)) or len(refs) == 0:
                    raise ValueError(f"subtask_groups[{sid_norm}] must be a non-empty list.")
                idxs = [int(x) for x in refs]
                subtask_to_refs[sid_norm] = idxs

        ref_to_subtask = {}
        for sid, refs in subtask_to_refs.items():
            for r in refs:
                if r < 0 or r >= num_refs:
                    raise ValueError(f"subtask_groups reference index {r} out of range [0,{num_refs - 1}]")
                if r in ref_to_subtask:
                    raise ValueError(f"subtask_groups overlap at reference index {r}")
                ref_to_subtask[r] = sid

        missing = [i for i in range(num_refs) if i not in ref_to_subtask]
        if missing:
            raise ValueError(f"subtask_groups missing reference indices: {missing}")
        return subtask_to_refs, ref_to_subtask
    
    #------------------------------------------------------------#

    def evaluate(
        self,
        observed_traj,
        dtw_scores=None,
        tool_score=None,
        use_tool=True,
        raw_oob_count: int = 0,
    ):
        """
        !================================================================================================!
        Evaluate observed trajectory against current stage

        Supports 2 modes:
        1) Can work after DTW has been executed wherein dtw_scores and tool_scores will have been populated
        2) Calls DTW inside when dtw_scores and tool_score aren't present. 
           Will be used on this mode (mode-2) in main function to ensure proper loading of reference traj for DTW scores
        
        !=!=!=!=!=!=====================================================================================!
        """

        if self.is_task_complete():
            return {"task_complete": True}
        
        stage = self.stages[self.current_stage_index]

        observed_traj = TrajectoryPreprocessor.preprocess(
            observed_traj,
            oob_policy={
                "mode": "drop_interpolate",
                "bounds": [0, 1],
                "max_gap_threshold": 20,
            },
        )

        reference_list = self.stage_references[stage["name"]]
        subtask_to_refs, ref_to_subtask = self._normalize_subtask_groups(stage, len(reference_list))
        stage_type = str(stage.get("stage_type", "ordered")).lower()
        required_count = int(stage.get("required_count", 1))
        lockout_enabled = bool(stage.get("lockout_enabled", True))
        cooldown_attempts = int(max(0, stage.get("cooldown_attempts", 0)))
        region_guard_min_dist = float(max(0.0, stage.get("region_guard_min_dist", 0.0)))
        margin_min = float(max(0.0, stage.get("margin_min", 0.0)))
        subtask_rois = self._normalize_subtask_rois(stage.get("subtask_rois"))

        alignment_path = None
        temporal_window_scores = []
        temporal_consistency = {}
        reference_len = None
        reference_id = None
        best_match = None
        second_best_match = None

        #--------------------------------------------------------------
        #------------- MODE 1 -> UNIT TESTS / PRECOMPUTED SCORES ------
        #--------------------------------------------------------------

        if dtw_scores is not None:
            dtw_similarity = dtw_scores.get("similarity_fastdtw", 0.0)
            tool_match_score = (tool_score or {}).get("tool_match_score", 0.0)
            alignment_path = dtw_scores.get("alignment_path")
            reference_id = dtw_scores.get("reference_id", None)
            best_match = dtw_scores.get("best_match", None)
            second_best_match = dtw_scores.get("second_best_match", None)

            # Mode-1 (precomputed scores): unordered stages still require matched
            # reference identity for subtask progression/lockout bookkeeping.
            if stage_type == "unordered_group" and (reference_id is None or int(reference_id) < 0):
                excluded = []
                if lockout_enabled:
                    state = self.group_progress.setdefault(
                        stage["name"],
                        {
                            "completed_subtasks": set(),
                            "accepted_events": [],
                            "accepted_events_by_subtask": {},
                            "attempt_index": 0,
                        },
                    )
                    for sid in state["completed_subtasks"]:
                        excluded.extend(subtask_to_refs.get(str(sid), []))
                _sim, _tool, reference_id, _path, best_match, second_best_match = \
                    self.reference_manager.match_best_reference(
                        observed_traj,
                        reference_list,
                        exclude_indices=excluded,
                        return_top2=True,
                    )
                if reference_id is not None and int(reference_id) >= 0:
                    reference_used = reference_list[int(reference_id)]
                    reference_len = len(reference_used)
                else:
                    return {
                        "stage_name": stage["name"],
                        "stage_passed": False,
                        "final_score": 0.0,
                        "task_complete": False,
                        "task_status": "IN_PROGRESS",
                        "recovery": {
                            "action": "RETRY_STAGE",
                            "reason": ["No remaining reference available for matching in this stage."],
                        },
                    }

        #------------------------------------------------------------------
        #------------- MODE 2 -> MAIN FUNCTION CALL / REAL DTW EXECUTION---
        #------------------------------------------------------------------

        elif dtw_scores is None:

            if not self.load_references and stage["name"] not in self.stage_references:
                raise RuntimeError(
                    "Reference loading disabled but DTW required"
                )

            excluded = []
            if stage_type == "unordered_group" and lockout_enabled:
                state = self.group_progress.setdefault(
                    stage["name"],
                    {
                        "completed_subtasks": set(),
                        "accepted_events": [],
                        "accepted_events_by_subtask": {},
                        "attempt_index": 0,
                    },
                )
                excluded = []
                for sid in state["completed_subtasks"]:
                    excluded.extend(subtask_to_refs.get(str(sid), []))

            dtw_similarity, tool_match_score, reference_id, alignment_path, best_match, second_best_match = \
                self.reference_manager.match_best_reference(
                    observed_traj, 
                    reference_list,
                    exclude_indices=excluded,
                    return_top2=True,
                )
            if reference_id < 0:
                return {
                    "stage_name": stage["name"],
                    "stage_passed": False,
                    "final_score": 0.0,
                    "task_complete": False,
                    "task_status": "IN_PROGRESS",
                    "recovery": {
                        "action": "RETRY_STAGE",
                        "reason": ["No remaining reference available for matching in this stage."],
                    },
                }
            if reference_id is not None and reference_id >= 0:
                reference_used = reference_list[reference_id]
                reference_len = len(reference_used)
                temporal_window_scores = self._compute_temporal_window_scores(
                    observed_traj=observed_traj,
                    reference_traj=reference_used,
                    alignment_path=alignment_path,
                    stage=stage,
                    n_windows=3,
                )
                temporal_consistency = self._summarize_temporal_consistency(temporal_window_scores)
            
        else: 
            return {"error": "Either dtw_scores or dtw_engine must be provided"}

        #------------ TOOL VALIDATION --------------------------

        observed_tools = {pt.get("tool", "NONE") for pt in observed_traj}
        excepted_tools = set(stage["expected_tools"])
        expected_parts = set(stage.get("expected_parts", []))
        active_part_match_score = self._active_match_score(
            observed_traj,
            field_name="active_part",
            candidates=stage.get("active_part_candidates", []),
        )
        active_tool_match_score = self._active_match_score(
            observed_traj,
            field_name="active_tool",
            candidates=stage.get("active_tool_candidates", stage.get("expected_tools", [])),
        )
        expected_part_match_score = self._expected_inventory_match_score(
            observed_traj,
            expected_labels=stage.get("expected_parts", []),
            expected_counts=stage.get("required_part_counts", {}) or stage.get("expected_inventory", {}),
        )
        expected_tool_match_score = self._expected_inventory_match_score(
            observed_traj,
            expected_labels=stage.get("expected_tools", []),
            expected_counts=stage.get("required_tool_counts", {}),
        )
        expected_pose_match_score, pose_expectation_results = self._expected_pose_match_score(
            observed_traj,
            stage.get("expected_poses", []),
        )

        tool_gate_enforced = bool(stage.get("enforce_active_tool_match", False))
        part_gate_enforced = bool(stage.get("enforce_active_part_match", False))
        pose_gate_enforced = bool(stage.get("enforce_expected_pose_match", bool(stage.get("expected_poses", []))))
        tool_valid = (active_tool_match_score > 0.0) or not tool_gate_enforced
        part_valid = (
            active_part_match_score > 0.0
            or not stage.get("active_part_candidates")
            or not part_gate_enforced
        )
        pose_valid = (
            all(bool(r.get("passed", False)) for r in pose_expectation_results)
            if pose_gate_enforced and pose_expectation_results
            else True
        )

        #------------ FINAL SCORE ------------------------------
        fatigue = compute_phase1_fatigue_metrics(observed_traj)
        smoothness_score = self._smoothness_score(fatigue)
        efficiency_score = self._efficiency_score(fatigue)
        temporal_stability_score = self._temporal_stability_score(fatigue)

        component_scores = {
            "dtw": float(dtw_similarity),
            "tool": float(tool_match_score),
            "active_part_match": float(active_part_match_score),
            "active_tool_match": float(active_tool_match_score),
            "expected_part_match": float(expected_part_match_score),
            "expected_tool_match": float(expected_tool_match_score),
            "expected_pose_match": float(expected_pose_match_score),
            "smoothness": float(smoothness_score),
            "efficiency": float(efficiency_score),
            "temporal_stability": float(temporal_stability_score),
        }
        final_score = self._weighted_score(component_scores)

        stage_pass = (
            final_score >= stage["score_threshold"] and
            tool_valid and
            part_valid and
            pose_valid
        )

        total_stages = len(self.stages)
        stage_index = self.current_stage_index

        feedback = {
            "stage_name": stage["name"],
            "stage_passed": stage_pass,
            "dtw_similarity": dtw_similarity,
            "tool_match_score": tool_match_score,
            "active_part_match_score": float(active_part_match_score),
            "active_tool_match_score": float(active_tool_match_score),
            "expected_part_match_score": float(expected_part_match_score),
            "expected_tool_match_score": float(expected_tool_match_score),
            "expected_pose_match_score": float(expected_pose_match_score),
            "pose_expectation_results": pose_expectation_results,
            "expected_poses": list(stage.get("expected_poses", [])),
            "smoothness_score": smoothness_score,
            "efficiency_score": efficiency_score,
            "temporal_stability_score": temporal_stability_score,
            "fatigue_metrics": fatigue,
            "alignment_path": alignment_path,
            "temporal_window_scores": temporal_window_scores,
            "temporal_consistency": temporal_consistency,
            "final_score": final_score,
            "expected_parts": list(expected_parts),
            "expected_tools": list(excepted_tools),
            "observed_tools": list(observed_tools),
            "active_part_candidates": list(stage.get("active_part_candidates", [])),
            "active_tool_candidates": list(stage.get("active_tool_candidates", [])),
            "enforce_active_tool_match": tool_gate_enforced,
            "enforce_active_part_match": part_gate_enforced,
            "enforce_expected_pose_match": pose_gate_enforced,
            "pose_valid": pose_valid,
            "reference_id": reference_id,
            "best_match_score": float(best_match["score"]) if best_match else None,
            "best_match_reference_id": int(best_match["index"]) if best_match else None,
            "second_best_score": float(second_best_match["score"]) if second_best_match else None,
            "second_best_reference_id": int(second_best_match["index"]) if second_best_match else None,
            "margin": (
                float(best_match["score"] - second_best_match["score"])
                if best_match and second_best_match
                else None
            ),
            "stage_type": stage_type,
            "required_count": required_count,
            "lockout_enabled": lockout_enabled,
            "margin_min": margin_min,
            "next_stage": None,
            "task_complete": False,
            "task_status": "IN_PROGRESS",
            "stage_index": stage_index,
            "total_stages": total_stages,
            "stages_left": total_stages - stage_index
        }

        #------------- FAILURE RECOVERY -----------------------------#

        if not stage_pass:
            feedback["failure_taxonomy"] = classify_failure(
                observed_traj=observed_traj,
                fatigue_metrics=fatigue,
                component_scores=component_scores,
                final_score=final_score,
                stage_threshold=stage["score_threshold"],
                raw_oob_count=raw_oob_count,
                reference_len=reference_len,
                temporal_consistency=temporal_consistency,
            )
            feedback["recovery"] = self.recovery_manager.generate_feedback(
                stage,
                final_score,
                tool_valid,
                part_valid
            )
            if pose_gate_enforced and not pose_valid:
                feedback["recovery"]["reason"].append("Expected object pose not observed in required temporal window.")
            primary_label = feedback["failure_taxonomy"].get("primary")
            if primary_label:
                feedback["recovery"]["reason"].append(f"Failure type: {primary_label}")
            return feedback

        #------------- STAGE PROGRESSION -----------------------------#
        if stage_type == "unordered_group":
            if lockout_enabled and required_count > len(subtask_to_refs):
                raise ValueError(
                    f"Stage '{stage['name']}' requires {required_count} unique completions "
                    f"but only {len(subtask_to_refs)} subtasks are available."
                )
            state = self.group_progress.setdefault(
                stage["name"],
                {
                    "completed_subtasks": set(),
                    "accepted_events": [],
                    "accepted_events_by_subtask": {},
                    "attempt_index": 0,
                },
            )
            state["attempt_index"] = int(state.get("attempt_index", 0)) + 1
            excluded_reference_indices = []
            if lockout_enabled:
                for sid in state["completed_subtasks"]:
                    excluded_reference_indices.extend(subtask_to_refs.get(str(sid), []))
            matched_subtask_id = ref_to_subtask.get(int(reference_id)) if reference_id is not None and reference_id >= 0 else None

            guard_block_reason = None
            guard_blocked = False
            centroid = self._trajectory_centroid(observed_traj)
            if matched_subtask_id is not None:
                same_subtask_hist = state.get("accepted_events_by_subtask", {}).get(str(matched_subtask_id), [])
                last_evt = same_subtask_hist[-1] if same_subtask_hist else None
            else:
                last_evt = None
            if last_evt is not None:
                if cooldown_attempts > 0:
                    attempts_since = state["attempt_index"] - int(last_evt.get("attempt_index", 0))
                    if attempts_since <= cooldown_attempts:
                        guard_blocked = True
                        guard_block_reason = (
                            f"cooldown_guard(subtask): attempts_since_last={attempts_since} <= cooldown_attempts={cooldown_attempts}"
                        )
                if not guard_blocked and region_guard_min_dist > 0.0 and centroid is not None:
                    last_centroid = last_evt.get("centroid")
                    if last_centroid is not None:
                        dx = float(centroid[0]) - float(last_centroid[0])
                        dy = float(centroid[1]) - float(last_centroid[1])
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < region_guard_min_dist:
                            guard_blocked = True
                            guard_block_reason = (
                                f"region_guard(subtask): centroid_distance={dist:.4f} < min_dist={region_guard_min_dist:.4f}"
                            )

            roi_gate_required = False
            roi_gate_ok = None
            roi = subtask_rois.get(str(matched_subtask_id))
            if roi is None and reference_id is not None and reference_id >= 0:
                roi = subtask_rois.get(int(reference_id))
            if roi is not None:
                roi_gate_required = True
                roi_gate_ok = self._point_in_roi(centroid, roi)
                if not roi_gate_ok and not guard_blocked:
                    guard_blocked = True
                    guard_block_reason = "roi_guard: centroid outside subtask ROI"

            if reference_id is not None and reference_id >= 0:
                margin = (
                    float(best_match["score"] - second_best_match["score"])
                    if best_match and second_best_match
                    else None
                )
                margin_gate_ok = True
                margin_gate_reason = "not_applicable"
                if margin_min > 0.0:
                    if margin is None:
                        # Last remaining candidate (or single-ref stage): do not block.
                        margin_gate_ok = True
                        margin_gate_reason = "single_candidate"
                    else:
                        margin_gate_ok = margin >= margin_min
                        margin_gate_reason = (
                            f"margin={margin:.4f} >= margin_min={margin_min:.4f}"
                            if margin_gate_ok
                            else f"margin={margin:.4f} < margin_min={margin_min:.4f}"
                        )

                if not guard_blocked and margin_gate_ok:
                    if matched_subtask_id is not None:
                        state["completed_subtasks"].add(str(matched_subtask_id))
                    state["accepted_events"].append(
                        {
                            "attempt_index": int(state["attempt_index"]),
                            "reference_id": int(reference_id),
                            "subtask_id": str(matched_subtask_id) if matched_subtask_id is not None else None,
                            "centroid": centroid,
                        }
                    )
                    state.setdefault("accepted_events_by_subtask", {}).setdefault(
                        str(matched_subtask_id) if matched_subtask_id is not None else "__none__", []
                    ).append(
                        {
                            "attempt_index": int(state["attempt_index"]),
                            "centroid": centroid,
                        }
                    )
                    accept_reason = f"accepted: stage_pass and gates passed ({margin_gate_reason})"
                    reject_reason = None
                else:
                    accept_reason = None
                    if guard_blocked:
                        reject_reason = f"rejected: GateB failed ({guard_block_reason})"
                    else:
                        reject_reason = f"rejected: GateA failed ({margin_gate_reason})"
            completed_count = len(state["completed_subtasks"])
            group_complete = completed_count >= required_count
            feedback["matched_subtask_id"] = matched_subtask_id
            feedback["matched_reference_id"] = reference_id
            feedback["completed_subtasks"] = sorted(state["completed_subtasks"])
            feedback["excluded_reference_indices"] = sorted(set(int(x) for x in excluded_reference_indices))
            feedback["group_progress"] = {
                "completed_count": int(completed_count),
                "required_count": int(required_count),
                "remaining_count": int(max(0, required_count - completed_count)),
                "group_complete": bool(group_complete),
                "total_subtasks": int(len(subtask_to_refs)),
                "cooldown_attempts": int(cooldown_attempts),
                "region_guard_min_dist": float(region_guard_min_dist),
                "guard_blocked": bool(guard_blocked),
                "guard_block_reason": guard_block_reason,
                "margin_min": float(margin_min),
                "margin_gate_ok": bool(margin_gate_ok) if "margin_gate_ok" in locals() else None,
                "roi_gate_required": bool(roi_gate_required),
                "roi_gate_ok": roi_gate_ok,
                "accept_reason": accept_reason if "accept_reason" in locals() else None,
                "reject_reason": reject_reason if "reject_reason" in locals() else None,
            }
            feedback["stage_complete"] = bool(group_complete)

            if group_complete:
                self.current_stage_index += 1
                if not self.is_task_complete():
                    feedback["next_stage"] = self.stages[self.current_stage_index]["name"]
                    feedback["task_complete"] = False
                    feedback["task_status"] = "IN_PROGRESS"
                    feedback["stages_left"] = total_stages - self.current_stage_index
                else:
                    feedback["task_complete"] = True
                    feedback["task_status"] = "COMPLETE"
                    feedback["stages_left"] = 0
            return feedback

        self.current_stage_index += 1

        if not self.is_task_complete():
            feedback["next_stage"] = self.stages[self.current_stage_index]["name"]
            feedback["task_complete"] = False
            feedback["task_status"] = "IN_PROGRESS"
            feedback["stages_left"] = total_stages - self.current_stage_index
        else:
            feedback["task_complete"] = True
            feedback["task_status"] = "COMPLETE"
            feedback["stages_left"] = 0

        return feedback
    
    #------------------------------------------------------------------

    def is_task_complete(self):
        return self.current_stage_index >= len(self.stages)
    
    #------------------------------------------------------------------

    def reset(self):
        self.current_stage_index = 0
        self.group_progress = {}

    def _weighted_score(self, component_scores):
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, score in component_scores.items():
            w = float(self.weights.get(name, 0.0))
            if w <= 0.0:
                continue
            weighted_sum += w * float(score)
            weight_sum += w
        if weight_sum <= 1e-9:
            return float(component_scores.get("dtw", 0.0))
        return weighted_sum / weight_sum

    @staticmethod
    def _smoothness_score(fatigue_metrics):
        # Lower jerk is smoother; map to [0,1] via exponential decay.
        jerk = float(fatigue_metrics.get("jerk_index", 0.0))
        return float(np.exp(-2.0 * max(0.0, jerk)))

    @staticmethod
    def _efficiency_score(fatigue_metrics):
        # Ideal path inefficiency is ~1.0. Larger means extra motion.
        ratio = fatigue_metrics.get("path_inefficiency_ratio", None)
        if ratio is None:
            return 0.0
        ratio = float(max(1.0, ratio))
        return 1.0 / ratio

    @staticmethod
    def _temporal_stability_score(fatigue_metrics):
        # Lower speed variance and lower pause ratio imply stable execution.
        speed_var = float(max(0.0, fatigue_metrics.get("speed_variance", 0.0)))
        pause_ratio = float(min(1.0, max(0.0, fatigue_metrics.get("pause_ratio", 0.0))))
        speed_term = float(np.exp(-3.0 * speed_var))
        pause_term = 1.0 - pause_ratio
        return 0.5 * speed_term + 0.5 * pause_term

    def _compute_temporal_window_scores(
        self,
        *,
        observed_traj,
        reference_traj,
        alignment_path,
        stage,
        n_windows=3,
    ):
        if not observed_traj or not reference_traj:
            return []
        if len(observed_traj) < n_windows:
            return []

        window_names = ["early", "mid", "late"] if n_windows == 3 else [f"window_{i}" for i in range(n_windows)]
        index_windows = np.array_split(np.arange(len(observed_traj)), n_windows)
        align_map = {}
        if alignment_path:
            for i, j in alignment_path:
                align_map.setdefault(int(i), []).append(int(j))

        windows = []
        for win_idx, obs_idx_arr in enumerate(index_windows):
            if len(obs_idx_arr) == 0:
                continue
            obs_start = int(obs_idx_arr[0])
            obs_end = int(obs_idx_arr[-1])
            obs_segment = observed_traj[obs_start : obs_end + 1]
            if len(obs_segment) < 2:
                continue

            ref_indices = []
            for i in range(obs_start, obs_end + 1):
                ref_indices.extend(align_map.get(i, []))

            if ref_indices:
                ref_start = max(0, min(ref_indices))
                ref_end = min(len(reference_traj) - 1, max(ref_indices))
            else:
                # Fallback proportional slicing when alignment mapping is sparse.
                ratio_start = obs_start / max(1, len(observed_traj) - 1)
                ratio_end = obs_end / max(1, len(observed_traj) - 1)
                ref_start = int(round(ratio_start * (len(reference_traj) - 1)))
                ref_end = int(round(ratio_end * (len(reference_traj) - 1)))

            if ref_end < ref_start:
                ref_start, ref_end = ref_end, ref_start
            ref_segment = reference_traj[ref_start : ref_end + 1]
            if len(ref_segment) < 2:
                continue

            try:
                scores = self.reference_manager.dtw_engine.compare_multiple(
                    observed_traj=obs_segment,
                    reference_traj=ref_segment,
                    use_tool=self.reference_manager.use_tool,
                )
            except ValueError:
                continue

            fatigue = compute_phase1_fatigue_metrics(obs_segment)
            component_scores = {
                "dtw": float(scores.get("similarity_fastdtw", 0.0)),
                "tool": float(scores.get("tool_match_score", 0.0)),
                "active_part_match": float(
                    self._active_match_score(
                        obs_segment,
                        field_name="active_part",
                        candidates=stage.get("active_part_candidates", []),
                    )
                ),
                "active_tool_match": float(
                    self._active_match_score(
                        obs_segment,
                        field_name="active_tool",
                        candidates=stage.get("active_tool_candidates", stage.get("expected_tools", [])),
                    )
                ),
                "expected_part_match": float(
                    self._expected_inventory_match_score(
                        obs_segment,
                        expected_labels=stage.get("expected_parts", []),
                        expected_counts=stage.get("required_part_counts", {}) or stage.get("expected_inventory", {}),
                    )
                ),
                "expected_tool_match": float(
                    self._expected_inventory_match_score(
                        obs_segment,
                        expected_labels=stage.get("expected_tools", []),
                        expected_counts=stage.get("required_tool_counts", {}),
                    )
                ),
                "smoothness": float(self._smoothness_score(fatigue)),
                "efficiency": float(self._efficiency_score(fatigue)),
                "temporal_stability": float(self._temporal_stability_score(fatigue)),
            }
            window_final = self._weighted_score(component_scores)
            windows.append(
                {
                    "window": window_names[win_idx] if win_idx < len(window_names) else f"window_{win_idx}",
                    "obs_index_range": [obs_start, obs_end],
                    "ref_index_range": [int(ref_start), int(ref_end)],
                    "num_obs_points": int(len(obs_segment)),
                    "num_ref_points": int(len(ref_segment)),
                    "dtw_similarity": float(scores.get("similarity_fastdtw", 0.0)),
                    "tool_match_score": float(scores.get("tool_match_score", 0.0)),
                    "active_part_match_score": float(component_scores.get("active_part_match", 0.0)),
                    "active_tool_match_score": float(component_scores.get("active_tool_match", 0.0)),
                    "expected_part_match_score": float(component_scores.get("expected_part_match", 0.0)),
                    "expected_tool_match_score": float(component_scores.get("expected_tool_match", 0.0)),
                    "final_score": float(window_final),
                }
            )
        return windows

    @staticmethod
    def _summarize_temporal_consistency(window_scores):
        if not window_scores:
            return {}
        finals = [float(w["final_score"]) for w in window_scores]
        labels = [w["window"] for w in window_scores]
        min_idx = int(np.argmin(finals))
        max_idx = int(np.argmax(finals))
        summary = {
            "num_windows": len(window_scores),
            "mean_final_score": float(np.mean(finals)),
            "std_final_score": float(np.std(finals)),
            "best_window": labels[max_idx],
            "worst_window": labels[min_idx],
            "worst_window_score": float(finals[min_idx]),
            "best_window_score": float(finals[max_idx]),
        }
        early = next((w for w in window_scores if w.get("window") == "early"), None)
        late = next((w for w in window_scores if w.get("window") == "late"), None)
        if early and late:
            summary["early_to_late_delta"] = float(late["final_score"] - early["final_score"])
        return summary
