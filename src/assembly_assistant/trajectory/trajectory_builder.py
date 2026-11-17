import numpy as np
import math

TOOL_ENCODING = {
    "NONE" : 0,
    "ScrewdriverL" : 1,
    "ScrewdriverS" : 2
}

class TrajectoryBuilder:
    def __init__(
            self,
            dominant_hand = "RIGHT",
            tool_hand_distance_thresh = 0.10,
            tool_min_confidence = 0.0,
            anchor_mode="wrist",
            ):
        
        """
        :param dominant_hand: 'LEFT' or 'RIGHT'
        :param tool_hand_distance_thresh: The minimum threshold between the tool and hand to be considered as held by hand
        :param anchor_mode: 'wrist' or 'wrist_index_midpoint'
        """
        self.trajectory = []
        self.DOMINANT_HAND = dominant_hand
        self.TOOL_HAND_DISTANCE_THRESHOLD = tool_hand_distance_thresh
        self.TOOL_MIN_CONFIDENCE = float(tool_min_confidence)
        allowed_anchor_modes = {"wrist", "wrist_index_midpoint"}
        if anchor_mode not in allowed_anchor_modes:
            raise ValueError(
                f"Unsupported anchor_mode={anchor_mode}. "
                f"Expected one of {sorted(list(allowed_anchor_modes))}."
            )
        self.ANCHOR_MODE = anchor_mode


    #-------------------------------------------------------------
    #-------------- PUBLIC API -----------------------------------
    #-------------------------------------------------------------

    def build_trajectory_point(self, frame_data):
        """
        Builds a single normalized trajecteory point from synchronized frame data
        
        :param self: Description
        :param frame_data: Description
        """

        hand_data = frame_data["hand"]

        if not hand_data["visible"]:
            return None
        
        keypoints = hand_data["keypoints"]

        anchor = self._hand_anchor(keypoints)
        x_norm, y_norm = self._normalize_position(anchor)

        tool_state = self._extract_relevant_tool(
            wrist = hand_data["wrist"],
            tool_data = frame_data["tool"]
        )
        active_tool = frame_data.get("active_tool") or {}
        active_part = frame_data.get("active_part") or {}
        active_tool_label = str(active_tool.get("label") or "NONE")
        active_part_label = str(active_part.get("label") or "NONE")

        traj_point = {
            "x" : x_norm,
            "y" : y_norm,
            "t" : frame_data["timestamp"],
            "tool" : active_tool_label,
            "tool_id" : TOOL_ENCODING.get(active_tool_label, TOOL_ENCODING["NONE"]),
            "active_tool": active_tool_label,
            "active_part": active_part_label,
            "expected_tools": list(frame_data.get("expected_tools") or []),
            "expected_parts": list(frame_data.get("expected_parts") or []),
            "expected_poses": list(frame_data.get("expected_poses") or []),
            "pose_predictions": list(frame_data.get("pose_predictions") or []),
            "workspace_inventory": dict(frame_data.get("workspace_inventory") or {}),
            "expected_inventory": dict(frame_data.get("expected_inventory") or {}),
        }

        return traj_point
    

    #---------------------------------------------------------------
    #--------------- HAND PROCESSING -------------------------------
    #---------------------------------------------------------------

    def _hand_anchor(self, keypoints):
        """
        Uses configured anchor definition.
        
        :param self: Description
        :param keypoints: Description
        """

        WRIST = 0
        INDEX_MCP = 5

        wrist = keypoints[WRIST]
        if self.ANCHOR_MODE == "wrist":
            return wrist
        if self.ANCHOR_MODE == "wrist_index_midpoint":
            index_mcp = keypoints[INDEX_MCP]
            return (wrist + index_mcp) / 2.0
        raise ValueError(f"Unhandled anchor mode: {self.ANCHOR_MODE}")
    
    def _normalize_position(self, anchor):
        """
        Anchor is expected to already be in normalized workspace coordinates
        after homography warp (unit square [0,1] x [0,1]).
        
        :param self: Description
        :param anchor: Description
        """
        x_norm = float(anchor[0])
        y_norm = float(anchor[1])

        # Optional hand-invariant horizontal mirroring.
        if self.DOMINANT_HAND == "LEFT":
            x_norm = 1.0 - x_norm

        return x_norm, y_norm
    

    #----------------------------------------------------------------
    #-------- TOOL ASSOCIATION --------------------------------------
    #----------------------------------------------------------------

    def _extract_relevant_tool(self, wrist, tool_data):
        """
        Euclidean-distance-based hand-tool association
        
        :param self: Description
        :param wrist: Description
        :param tool_data: Description
        """

        if tool_data["class"] == "NONE":
            return {
                "class" : "NONE",
                "position" : None
            }

        confidence = float(tool_data.get("confidence", 0.0))
        if confidence < self.TOOL_MIN_CONFIDENCE:
            return {
                "class": "NONE",
                "position": None,
            }

        bbox = tool_data.get("bbox")
        if bbox is None or len(bbox) != 4:
            return {
                "class": "NONE",
                "position": None,
            }
        
        # Tool bbox center
        x1, y1, x2, y2 = bbox
        box_center = np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ])

        dist = self._euclidean(wrist, box_center)

        if dist < self.TOOL_HAND_DISTANCE_THRESHOLD:
            return {
                "class" : tool_data["class"],
                "position" : box_center
            }
        else:
            return {
                "class" : "NONE",
                "position" : None
            }
        
    @staticmethod
    def _euclidean(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def append(self, traj_point):
        self.trajectory.append(traj_point)

    def get_trajectory(self):
        return self.trajectory
    
    def reset(self):
        self.trajectory = []

    def load_reference(self):
        #return np.load("data/reference_trajectory.npy", allow_pickle=True)
        return [
            {"x": 0.1, "y": 0.2, "t": 0.0},
            {"x": 0.2, "y": 0.3, "t": 0.1},
            {"x": 0.3, "y": 0.4, "t": 0.2},
        ]
