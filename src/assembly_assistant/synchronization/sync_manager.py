import numpy as np

from assembly_assistant.detectors.detection_bridge import (
    filter_workspace_detections,
    partition_detections,
    select_active_part,
    select_active_tool,
    select_primary_detection,
)
from assembly_assistant.detection_types.label_taxonomy import build_inventory, canonical_label

class SyncManager:
    def __init__(self, max_time_diff=0.5, preferred_hand="RIGHT"):
        """
        max_time_diff: maximum alllowed time difference(seconds) between hand frame and tool frame

        """
        self.max_time_diff = max_time_diff
        preferred_hand = str(preferred_hand).upper()
        if preferred_hand not in {"LEFT", "RIGHT"}:
            raise ValueError("preferred_hand must be LEFT or RIGHT")
        self.preferred_hand = preferred_hand
        self._prev_active_part = None
        self._prev_active_tool = None
        self._prev_active_part_missing = 0
        self._prev_active_tool_missing = 0
        self._active_memory_frames = 1
        self._active_acquire_frames = 2
        self._pending_active_part = None
        self._pending_active_tool = None
        self._pending_active_part_streak = 0
        self._pending_active_tool_streak = 0
        self._last_stage_name = None

    
    def sync(
        self,
        hand_data,
        tool_data,
        timestamp,
        frame_id,
        detections=None,
        expected_parts=None,
        expected_tools=None,
        expected_inventory=None,
        active_part_candidates=None,
        active_tool_candidates=None,
        stage_name=None,
        workspace_filter=None,
    ):
        """
        Builds a synchronized FrameData Dictionary
        
        :param hand_data: Output of HandDetector.detect()
        :param tool_data: Output of ToolDetector.detect()
        :param timestamp: float (monotonic time)
        :param frame_id: int
        """
        stage_name = str(stage_name).strip() or None
        if stage_name != self._last_stage_name:
            self._prev_active_part = None
            self._prev_active_tool = None
            self._prev_active_part_missing = 0
            self._prev_active_tool_missing = 0
            self._pending_active_part = None
            self._pending_active_tool = None
            self._pending_active_part_streak = 0
            self._pending_active_tool_streak = 0
            self._last_stage_name = stage_name

        #------------------- Hand Block -----------------------
        hands_block = self._resolve_hands(hand_data)
        hand_block = hands_block["active"]

        #------------------- Detection + Tool Block -----------------------
        detection_block = self._resolve_detections(detections)
        tools_block, parts_block, objects_block = partition_detections(detection_block)
        workspace_objects = filter_workspace_detections(objects_block, workspace_filter=workspace_filter)
        workspace_tools, workspace_parts, _ = partition_detections(workspace_objects)
        inventory = build_inventory(objects_block)
        workspace_inventory = build_inventory(workspace_objects)
        tool_block = self._resolve_tool(hand_data, tool_data, timestamp)
        if detection_block:
            tool_block = select_primary_detection(detection_block, hand_block)
        expected_tools_block = self._normalize_expected_labels(expected_tools)
        expected_parts_block = self._normalize_expected_labels(expected_parts)
        expected_inventory_block = self._normalize_expected_inventory(expected_inventory)
        active_tool_candidates_block = self._normalize_expected_labels(active_tool_candidates)
        active_part_candidates_block = self._normalize_expected_labels(active_part_candidates)
        active_tool_source = self._select_active_source_detections(
            full_detections=tools_block,
            workspace_detections=workspace_tools,
            candidate_labels=active_tool_candidates_block,
            allow_none=True,
        )
        active_part_source = self._select_active_source_detections(
            full_detections=parts_block,
            workspace_detections=workspace_parts,
            candidate_labels=active_part_candidates_block,
            allow_none=False,
        )
        active_tool = select_active_tool(
            active_tool_source,
            hand_block,
            preferred_labels=active_tool_candidates_block,
            previous_active=self._prev_active_tool,
        )
        active_part = select_active_part(
            active_part_source if active_part_source else workspace_parts,
            hand_block,
            preferred_labels=active_part_candidates_block,
            previous_active=self._prev_active_part,
        )
        active_tool = self._stabilize_active_selection(
            active_tool,
            previous_active=self._prev_active_tool,
            missing_counter_attr="_prev_active_tool_missing",
            allowed_labels=active_tool_candidates_block,
            hand_block=hand_block,
            pending_attr="_pending_active_tool",
            pending_streak_attr="_pending_active_tool_streak",
        )
        active_part = self._stabilize_active_selection(
            active_part,
            previous_active=self._prev_active_part,
            missing_counter_attr="_prev_active_part_missing",
            allowed_labels=active_part_candidates_block,
            hand_block=hand_block,
            pending_attr="_pending_active_part",
            pending_streak_attr="_pending_active_part_streak",
        )
        self._prev_active_tool = dict(active_tool) if isinstance(active_tool, dict) else None
        self._prev_active_part = dict(active_part) if isinstance(active_part, dict) else None

        #------------------- Final FrameDate -----------------------
        frame_data = {
            "frame_id" : frame_id,
            "timestamp" : timestamp,
            "hand" : hand_block,
            "hands": {
                "left": hands_block["left"],
                "right": hands_block["right"],
                "active_label": hands_block["active_label"],
                "visible_count": hands_block["visible_count"],
            },
            "tool" : tool_block,
            "tools": tools_block,
            "parts": parts_block,
            "objects": objects_block,
            "inventory": inventory,
            "workspace_tools": workspace_tools,
            "workspace_parts": workspace_parts,
            "workspace_objects": workspace_objects,
            "workspace_inventory": workspace_inventory,
            "expected_tools": expected_tools_block,
            "expected_parts": expected_parts_block,
            "expected_inventory": expected_inventory_block,
            "active_tool_candidates": active_tool_candidates_block,
            "active_part_candidates": active_part_candidates_block,
            "active_tool": active_tool,
            "active_part": active_part,
            "detections": detection_block,
        }

        return frame_data
    

    ######################################################################
    ################  Helper Function ####################################
    ######################################################################


    def _resolve_detections(self, detections):
        if detections is None:
            return []
        if isinstance(detections, list):
            return [dict(d) for d in detections if isinstance(d, dict)]
        if isinstance(detections, dict):
            return [dict(detections)]
        return []

    def _normalize_expected_labels(self, labels):
        if labels is None:
            return []
        if isinstance(labels, str):
            labels = [labels]
        return [canonical_label(x) for x in labels if str(x).strip()]

    def _normalize_expected_inventory(self, inventory):
        if not isinstance(inventory, dict):
            return {}
        out = {}
        for raw_key, raw_value in inventory.items():
            key = canonical_label(raw_key)
            try:
                out[key] = int(raw_value)
            except Exception:
                continue
        return out

    def _select_active_source_detections(
        self,
        *,
        full_detections,
        workspace_detections,
        candidate_labels,
        allow_none: bool,
    ):
        labels = self._normalize_expected_labels(candidate_labels)
        label_set = set(labels)
        if allow_none and label_set == {"NONE"}:
            return []
        if label_set and label_set != {"NONE"}:
            preferred = [
                dict(det)
                for det in (full_detections or [])
                if canonical_label(det.get("label", det.get("class", "NONE"))) in label_set
            ]
            if preferred:
                return preferred
        return [dict(det) for det in (workspace_detections or []) if isinstance(det, dict)]

    def _stabilize_active_selection(
        self,
        current_active,
        *,
        previous_active,
        missing_counter_attr: str,
        allowed_labels,
        hand_block,
        pending_attr: str,
        pending_streak_attr: str,
    ):
        allowed_set = set(self._normalize_expected_labels(allowed_labels))
        current = dict(current_active) if isinstance(current_active, dict) else None
        previous = dict(previous_active) if isinstance(previous_active, dict) else None
        pending = dict(getattr(self, pending_attr, None) or {}) or None
        pending_streak = int(getattr(self, pending_streak_attr, 0))

        def _label(item):
            return canonical_label((item or {}).get("label", (item or {}).get("class", "NONE")))

        if current is not None:
            current_label = _label(current)
            previous_label = _label(previous) if previous is not None else "NONE"

            if previous is not None and current_label == previous_label:
                setattr(self, missing_counter_attr, 0)
                setattr(self, pending_attr, None)
                setattr(self, pending_streak_attr, 0)
                return current

            pending_label = _label(pending) if pending is not None else "NONE"
            if pending is not None and current_label == pending_label:
                pending_streak += 1
            else:
                pending = current
                pending_streak = 1
            setattr(self, pending_attr, pending)
            setattr(self, pending_streak_attr, pending_streak)
            setattr(self, missing_counter_attr, 0)

            if previous is None:
                if pending_streak >= self._active_acquire_frames:
                    setattr(self, pending_attr, None)
                    setattr(self, pending_streak_attr, 0)
                    return current
                return None

            if pending_streak >= self._active_acquire_frames:
                setattr(self, pending_attr, None)
                setattr(self, pending_streak_attr, 0)
                return current
            return previous

        if previous is None or not hand_block.get("visible"):
            setattr(self, missing_counter_attr, 0)
            setattr(self, pending_attr, None)
            setattr(self, pending_streak_attr, 0)
            return None

        previous_label = canonical_label(previous.get("label", previous.get("class", "NONE")))
        if allowed_set and previous_label not in allowed_set:
            setattr(self, missing_counter_attr, 0)
            setattr(self, pending_attr, None)
            setattr(self, pending_streak_attr, 0)
            return None

        misses = int(getattr(self, missing_counter_attr, 0))
        if misses >= self._active_memory_frames:
            setattr(self, missing_counter_attr, 0)
            setattr(self, pending_attr, None)
            setattr(self, pending_streak_attr, 0)
            return None

        setattr(self, missing_counter_attr, misses + 1)
        return previous

    def _resolve_tool(self, hand_data, tool_data, hand_timestamp):
        """
        Resolves the tool state using temporal validity
        
        :param hand_data: Output of HandDetector.detect()
        :param tool_data: Output of ToolDetector.detect()
        :param hand_timestamp: float (monotonic time in this case mainly associated with the hand frames)
        """

        if tool_data is None:
            return self._no_tool()
        
        tool_timestamp = tool_data.get("timestamp", hand_timestamp)
        dt = abs(tool_timestamp - hand_timestamp)

        # Temporal gating
        if dt > self.max_time_diff:
            return self._no_tool()
        
        # Extract tool info
        return {
            "class" : canonical_label(tool_data.get("class", "NONE")),
            "bbox" : tool_data.get("bbox", None),
            "confidence" : float(tool_data.get("confidence", 0.0))
        }
    
    def _no_tool(self):
        """Standardized NO_TOOL state"""

        return{
            "class" : "NONE",
            "bbox" : None,
            "confidence" : 0.0
        }

    def _resolve_hands(self, hand_data):
        """
        Supports:
        - None
        - legacy single-hand format: {"keypoints": ..., "velocity": ...}
        - two-hand format: {"left": {...}|None, "right": {...}|None}
        """
        left = self._empty_hand()
        right = self._empty_hand()

        if hand_data is None:
            return {
                "left": left,
                "right": right,
                "active": self._empty_hand(),
                "active_label": None,
                "visible_count": 0,
            }

        if isinstance(hand_data, dict) and ("left" in hand_data or "right" in hand_data):
            left = self._to_hand_block(hand_data.get("left"))
            right = self._to_hand_block(hand_data.get("right"))
        else:
            # Backward-compatible single-hand input.
            single = self._to_hand_block(hand_data)
            if self.preferred_hand == "LEFT":
                left = single
            else:
                right = single

        active_label = None
        if self.preferred_hand == "RIGHT":
            if right["visible"]:
                active_label = "RIGHT"
            elif left["visible"]:
                active_label = "LEFT"
        else:
            if left["visible"]:
                active_label = "LEFT"
            elif right["visible"]:
                active_label = "RIGHT"

        if active_label == "LEFT":
            active = left
        elif active_label == "RIGHT":
            active = right
        else:
            active = self._empty_hand()

        visible_count = int(left["visible"]) + int(right["visible"])
        return {
            "left": left,
            "right": right,
            "active": active,
            "active_label": active_label,
            "visible_count": visible_count,
        }

    def _to_hand_block(self, raw):
        if raw is None:
            return self._empty_hand()
        keypoints = np.asarray(raw.get("keypoints"), dtype=float)
        if keypoints.shape != (21, 2):
            return self._empty_hand()
        wrist = keypoints[0]
        return {
            "visible": True,
            "keypoints": keypoints,
            "keypoints_px": np.asarray(raw.get("keypoints_px"), dtype=float) if raw.get("keypoints_px") is not None else None,
            "wrist": (float(wrist[0]), float(wrist[1])),
            "wrist_px": tuple(float(v) for v in raw.get("wrist_px")) if raw.get("wrist_px") is not None else None,
            "velocity": float(raw.get("velocity", 0.0)),
        }

    def _empty_hand(self):
        return {
            "visible": False,
            "keypoints": None,
            "keypoints_px": None,
            "wrist": None,
            "wrist_px": None,
            "velocity": 0.0,
        }
