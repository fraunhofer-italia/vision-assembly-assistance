import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from math import exp
from assembly_assistant.utils.logger import get_logger

logger = get_logger(__name__)

TOOL_ENCODING = {
    "NONE" : 0,
    "ScrewdriverL" : 1,
    "ScrewdriverS" : 2
}

class DTWEngine:
    """
    Compare two trajectories using Dynamic Time Warping
    Trajectories are lists of dicts: [{"x":....., "y":......, "t":......, "tool":.....},.....]
    """
    def __init__(self):
        pass

    def _extract_points(self, traj, use_tool=True):
        """
        Converts trajectory dicts to np.array for DTW
        Optionally encodes tool presence as an extra dimension
        """
        points = []

        if traj is None :
            logger.warning("Empty trajectory passed to DTWEngine")
            return np.empty((0, 3 if use_tool else 2))
            
        for i, pt in enumerate(traj):
            try:
                x, y = pt["x"], pt["y"]
            except KeyError as e:
                logger.error("Trajectory point %d missing key: %s", i, e)
                continue

            if use_tool:
                if "tool_id" in pt:
                    tool_flag = pt["tool_id"]
                else:
                    tool_name = pt.get("tool", "NONE")
                    if tool_name not in TOOL_ENCODING:
                        logger.warning(
                            "Unknown tool '%s' at point %d, treating as NONE",
                            tool_name, i
                        )
                        tool_flag = TOOL_ENCODING["NONE"]
                    else:
                        tool_flag = TOOL_ENCODING[tool_name]

                points.append([x, y, tool_flag])
            else:
                points.append([x, y])
        return np.asarray(points, dtype=float)
    
    def compare_multiple(self, observed_traj, reference_traj, use_tool = False, return_alignment=True):
        """
        Compare trajectories using FastDTW and return similarity/tool scores

        :param observed_traj: Description
        :param reference_traj: Description
        :param use_tool: Description
        """

        if  len(observed_traj) == 0 or  len(reference_traj) == 0:
            raise ValueError("Observed or reference trajectory is empty")
        
        #if  observed_traj is None or  reference_traj.size is None:
           # raise ValueError("Observed or reference trajectory is None")

        obs_points = self._extract_points(observed_traj, use_tool=use_tool)
        ref_points = self._extract_points(reference_traj, use_tool=use_tool)

        scores = {}

        # fastdtw
        distance, path = fastdtw(obs_points, ref_points, dist = euclidean)
        scores["fastdtw"] = distance
        dtw_norm = distance / (len(observed_traj) + len(reference_traj))
        similarity = exp(-dtw_norm / 0.5)
        scores["similarity_fastdtw"] = similarity                  ## in future use this for acceptance or rejection in stage FSM
        if return_alignment:
            scores["alignment_path"] = path
        scores["tool_match_score"] = self._tool_match_score(
            observed_traj,
            reference_traj,
            alignment_path=path,
        )


        return scores
    
    def _tool_match_score(self, observed_traj, reference_traj, alignment_path=None):
        """
        Returns a score in [0,1] measuring tool correctness
        """
        if not observed_traj or not reference_traj:
            return 0.0

        obs_tools = [pt.get("tool", "NONE") for pt in observed_traj]
        ref_tools = [pt.get("tool", "NONE") for pt in reference_traj]

        # Prefer DTW-aligned tool comparison so unequal trajectory lengths
        # and temporal warping are evaluated on matched positions.
        if alignment_path:
            matches = 0
            compared = 0
            obs_len = len(obs_tools)
            ref_len = len(ref_tools)
            for i, j in alignment_path:
                if 0 <= i < obs_len and 0 <= j < ref_len:
                    compared += 1
                    if obs_tools[i] == ref_tools[j]:
                        matches += 1
            if compared > 0:
                return matches / compared

        # Fallback if no alignment path is available.
        compared = min(len(obs_tools), len(ref_tools))
        if compared == 0:
            return 0.0
        matches = sum(o == r for o, r in zip(obs_tools, ref_tools))
        return matches / compared


        
