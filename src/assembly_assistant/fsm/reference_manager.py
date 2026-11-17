#############################################################################
##### PURPOSE : ------------------ ##########################################
##### In real skill learning there's no single correct expert traj ##########
##### Experts differ in speed, style and micro-movement #####################
##### So each stage must support multiple reference trajectories ############
##### And choose BEST DTW MATCH #############################################
#############################################################################


import numpy as np
from assembly_assistant.similarity.dtw_engine import DTWEngine

class ReferenceManager:

    def __init__(self, use_tool = False):
        self.dtw_engine = DTWEngine()
        self.use_tool = use_tool

    #------------------------------------------------------------------------

    def load_references(self, reference_path):
        """
        Loads List of reference trajectories for one stage
        
        """

        refs = []
        for path in reference_path:
            refs.append(np.load(path, allow_pickle=True))
        return refs
    
    #-------------------------------------------------------------------------

    def match_best_reference(
        self,
        observed_traj,
        reference_list,
        exclude_indices=None,
        return_top2=False,
    ):
        """
        Compare observed trajectory with multiple references
        Returns best similarity score
        """

        best_score = -1
        best_tool_score = 0
        best_index = -1
        best_alignment_path = None

        exclude = set(exclude_indices or [])
        candidates = []

        for idx, ref in enumerate(reference_list):
            if idx in exclude:
                continue

            scores = self.dtw_engine.compare_multiple(
                observed_traj=observed_traj,
                reference_traj= ref,
                use_tool= self.use_tool
            )

            similarity = scores["similarity_fastdtw"]

            tool_score = scores["tool_match_score"]

            alignment_path = scores.get("alignment_path")

            if similarity > best_score:
                best_score = similarity
                best_tool_score = tool_score
                best_index = idx
                best_alignment_path = alignment_path
            candidates.append(
                {
                    "index": int(idx),
                    "score": float(similarity),
                    "tool_score": float(tool_score),
                    "alignment_path": alignment_path,
                }
            )
        if not return_top2:
            return best_score, best_tool_score, best_index, best_alignment_path

        candidates_sorted = sorted(candidates, key=lambda c: c["score"], reverse=True)
        best = candidates_sorted[0] if candidates_sorted else None
        second = candidates_sorted[1] if len(candidates_sorted) > 1 else None
        return best_score, best_tool_score, best_index, best_alignment_path, best, second
    
