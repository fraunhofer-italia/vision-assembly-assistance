from __future__ import annotations

import math
from typing import Optional, Tuple


class ActiveHandSelector:
    """
    Selects which hand should drive FSM/trajectory at frame level.

    Policies:
    - dominant: prefer configured dominant hand, fallback to other visible hand.
    - closest_to_tool: pick visible hand with smallest wrist-to-tool-center distance.
      Falls back to dominant when tool is unavailable.
    - best_match: weighted hybrid policy for active-manipulation continuity.
      score = w1*dominant + w2*velocity + w3*tool_proximity, with switch hysteresis.
    """

    def __init__(
        self,
        policy: str = "dominant",
        dominant_hand: str = "RIGHT",
        tool_min_confidence: float = 0.0,
        static_velocity_max: float = 0.005,
        switch_min_velocity: float = 0.015,
        switch_velocity_delta: float = 0.008,
        weight_dominant: float = 0.30,
        weight_velocity: float = 0.60,
        weight_tool: float = 0.10,
        velocity_norm_scale: float = 0.06,
        tool_distance_norm: float = 0.30,
        switch_score_margin: float = 0.10,
        switch_confirm_frames: int = 2,
    ):
        self.policy = str(policy).strip().lower()
        if self.policy not in {"dominant", "closest_to_tool", "best_match"}:
            raise ValueError("active hand policy must be one of: dominant, closest_to_tool, best_match")

        dominant = str(dominant_hand).upper()
        if dominant not in {"LEFT", "RIGHT"}:
            raise ValueError("dominant_hand must be LEFT or RIGHT")
        self.dominant_hand = dominant
        self.tool_min_confidence = float(tool_min_confidence)
        self.static_velocity_max = float(max(0.0, static_velocity_max))
        self.switch_min_velocity = float(max(0.0, switch_min_velocity))
        self.switch_velocity_delta = float(max(0.0, switch_velocity_delta))
        self.weight_dominant = float(max(0.0, weight_dominant))
        self.weight_velocity = float(max(0.0, weight_velocity))
        self.weight_tool = float(max(0.0, weight_tool))
        self.velocity_norm_scale = float(max(1e-6, velocity_norm_scale))
        self.tool_distance_norm = float(max(1e-6, tool_distance_norm))
        self.switch_score_margin = float(max(0.0, switch_score_margin))
        self.switch_confirm_frames = int(max(1, switch_confirm_frames))

        self._last_label: Optional[str] = None
        self._last_anchor: Optional[Tuple[float, float]] = None
        self._pending_switch_label: Optional[str] = None
        self._pending_switch_count: int = 0

    def reset(self):
        self._last_label = None
        self._last_anchor = None
        self._pending_switch_label = None
        self._pending_switch_count = 0

    def select(self, frame_data: dict) -> Optional[str]:
        hands = frame_data.get("hands") or {}
        left = hands.get("left") or {"visible": False}
        right = hands.get("right") or {"visible": False}

        left_visible = bool(left.get("visible", False))
        right_visible = bool(right.get("visible", False))

        if not left_visible and not right_visible:
            self._last_label = None
            self._last_anchor = None
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return None

        if self.policy == "dominant":
            label = self._pick_dominant(left_visible, right_visible)
        elif self.policy == "closest_to_tool":
            label = self._pick_closest_to_tool(frame_data, left, right, left_visible, right_visible)
        else:
            label = self._pick_best_match(frame_data, left, right, left_visible, right_visible)

        if label is None:
            return None
        chosen = left if label == "LEFT" else right
        frame_data["hand"] = chosen
        frame_data.setdefault("hands", {})["active_label"] = label
        wrist = chosen.get("wrist")
        if wrist is not None:
            self._last_anchor = (float(wrist[0]), float(wrist[1]))
        self._last_label = label
        return label

    def _pick_dominant(self, left_visible: bool, right_visible: bool) -> Optional[str]:
        if self.dominant_hand == "LEFT":
            if left_visible:
                return "LEFT"
            if right_visible:
                return "RIGHT"
        else:
            if right_visible:
                return "RIGHT"
            if left_visible:
                return "LEFT"
        return None

    def _pick_closest_to_tool(self, frame_data, left, right, left_visible, right_visible) -> Optional[str]:
        tool = frame_data.get("tool") or {}
        bbox = tool.get("bbox")
        confidence = float(tool.get("confidence", 0.0))
        if bbox is None or len(bbox) != 4 or confidence < self.tool_min_confidence:
            return self._pick_dominant(left_visible, right_visible)

        cx = float(bbox[0] + bbox[2]) * 0.5
        cy = float(bbox[1] + bbox[3]) * 0.5

        best = None
        best_dist = float("inf")
        if left_visible and left.get("wrist") is not None:
            d = self._dist(left["wrist"], (cx, cy))
            best, best_dist = "LEFT", d
        if right_visible and right.get("wrist") is not None:
            d = self._dist(right["wrist"], (cx, cy))
            if d < best_dist:
                best, best_dist = "RIGHT", d
        return best if best is not None else self._pick_dominant(left_visible, right_visible)

    def _pick_best_match(self, frame_data, left, right, left_visible, right_visible) -> Optional[str]:
        if left_visible and not right_visible:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return "LEFT"
        if right_visible and not left_visible:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return "RIGHT"

        # Weighted hybrid score:
        # dominant prior + normalized velocity + optional tool proximity.
        tool = frame_data.get("tool") or {}
        bbox = tool.get("bbox")
        conf = float(tool.get("confidence", 0.0))
        tool_center = None
        if bbox is not None and len(bbox) == 4 and conf >= self.tool_min_confidence:
            tool_center = (float(bbox[0] + bbox[2]) * 0.5, float(bbox[1] + bbox[3]) * 0.5)

        scores = {}
        for label, hand in (("LEFT", left), ("RIGHT", right)):
            if not bool(hand.get("visible", False)):
                continue
            score = 0.0
            if label == self.dominant_hand:
                score += self.weight_dominant

            vel = float(hand.get("velocity", 0.0) or 0.0)
            vel_score = min(1.0, max(0.0, vel) / self.velocity_norm_scale)
            score += self.weight_velocity * vel_score

            wrist = hand.get("wrist")
            if tool_center is not None and wrist is not None:
                dist = self._dist(wrist, tool_center)
                tool_score = max(0.0, 1.0 - (dist / self.tool_distance_norm))
                score += self.weight_tool * tool_score
            scores[label] = score

        if not scores:
            return self._pick_dominant(left_visible, right_visible)
        if len(scores) == 1:
            return next(iter(scores.keys()))

        left_score = scores.get("LEFT", float("-inf"))
        right_score = scores.get("RIGHT", float("-inf"))
        if left_score == right_score:
            candidate = self._pick_dominant(left_visible, right_visible)
        else:
            candidate = "LEFT" if left_score > right_score else "RIGHT"

        if self._last_label not in {"LEFT", "RIGHT"}:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return candidate

        current = self._last_label
        if current not in scores:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return candidate
        if candidate == current:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return current

        candidate_score = scores.get(candidate, float("-inf"))
        current_score = scores.get(current, float("-inf"))
        if (candidate_score - current_score) < self.switch_score_margin:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return current

        if self._pending_switch_label == candidate:
            self._pending_switch_count += 1
        else:
            self._pending_switch_label = candidate
            self._pending_switch_count = 1

        if self._pending_switch_count >= self.switch_confirm_frames:
            self._pending_switch_label = None
            self._pending_switch_count = 0
            return candidate
        return current

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)
