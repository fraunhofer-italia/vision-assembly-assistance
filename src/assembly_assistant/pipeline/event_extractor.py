from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from assembly_assistant.pipeline.region_registry import RegionRegistry


class RegionEventExtractor:
    """
    Region-event layer independent of MotionFSM.

    Emits semantic events from noisy per-frame region hits:
      - enter_region
      - exit_region
      - dwell_region
      - pick_candidate
      - drop_candidate
    """

    def __init__(
        self,
        region_registry: RegionRegistry,
        *,
        operation_mode: str = "dismantle",
        enter_confirm_frames: int = 2,
        exit_confirm_frames: int = 2,
        dwell_confirm_frames: int = 6,
        event_gap_frames: int = 3,
        occlusion_tolerance_frames: int = 3,
        pick_velocity_min: float = 0.004,
        drop_velocity_max: float = 0.02,
    ):
        self.registry = region_registry
        self.operation_mode = str(operation_mode).lower()
        self.enter_confirm_frames = int(max(1, enter_confirm_frames))
        self.exit_confirm_frames = int(max(1, exit_confirm_frames))
        self.dwell_confirm_frames = int(max(1, dwell_confirm_frames))
        self.event_gap_frames = int(max(0, event_gap_frames))
        self.occlusion_tolerance_frames = int(max(0, occlusion_tolerance_frames))
        self.pick_velocity_min = float(max(0.0, pick_velocity_min))
        self.drop_velocity_max = float(max(0.0, drop_velocity_max))

        self._state = defaultdict(
            lambda: {
                "in_region": False,
                "present_count": 0,
                "absent_count": 0,
                "enter_frame": None,
                "enter_ts": None,
                "dwell_emitted": False,
            }
        )
        self._last_emit_frame = {}

    def reset(self) -> None:
        self._state.clear()
        self._last_emit_frame = {}

    def update(self, frame_data: dict, *, frame_id: int, timestamp: float) -> List[dict]:
        return self.update_with_scope(frame_data, frame_id=frame_id, timestamp=timestamp, hand_scope="any")

    def update_with_scope(
        self,
        frame_data: dict,
        *,
        frame_id: int,
        timestamp: float,
        hand_scope: str = "any",
    ) -> List[dict]:
        region_hits = frame_data.get("region_hits") or {}
        hand_hits = region_hits.get("hands") or {}
        active_label = region_hits.get("active_label")
        hands = frame_data.get("hands") or {}
        scope = str(hand_scope).strip().lower()
        if scope not in {"any", "active"}:
            scope = "any"

        visible_labels = []
        for label in ("LEFT", "RIGHT"):
            hb = hands.get(label.lower()) or {}
            if bool(hb.get("visible", False)):
                visible_labels.append(label)
        non_active = next((l for l in visible_labels if l != active_label), None)
        labels_to_process = ("LEFT", "RIGHT")
        if scope == "active":
            labels_to_process = (active_label,) if active_label in {"LEFT", "RIGHT"} else tuple()

        events: List[dict] = []
        for label in labels_to_process:
            hit_regions = set((hand_hits.get(label) or {}).get("regions", []))
            hand_block = (hands.get(label.lower()) or {})
            hand_visible = bool(hand_block.get("visible", False))
            hand_velocity = float(hand_block.get("velocity", 0.0) or 0.0)

            for region_id in self.registry.regions.keys():
                key = (label, region_id)
                st = self._state[key]
                is_inside = region_id in hit_regions

                if is_inside:
                    st["present_count"] += 1
                    st["absent_count"] = 0
                else:
                    st["present_count"] = 0
                    # tolerate short temporary missing visibility
                    if hand_visible:
                        st["absent_count"] += 1
                    else:
                        st["absent_count"] += max(0, 1 - self.occlusion_tolerance_frames)

                if not st["in_region"] and st["present_count"] >= self.enter_confirm_frames:
                    st["in_region"] = True
                    st["enter_frame"] = int(frame_id)
                    st["enter_ts"] = float(timestamp)
                    st["dwell_emitted"] = False
                    evt = self._make_event(
                        event_type="enter_region",
                        frame_id=frame_id,
                        timestamp=timestamp,
                        hand_label=label,
                        active_label=active_label,
                        non_active_label=non_active,
                        visible_labels=visible_labels,
                        hand_scope=scope,
                        region_id=region_id,
                        hand_velocity=hand_velocity,
                    )
                    if self._allow_emit(evt, frame_id):
                        events.append(evt)
                        events.extend(self._candidate_events(evt, hand_velocity, frame_id))

                if st["in_region"] and st["enter_frame"] is not None:
                    dwell_frames = int(frame_id - st["enter_frame"])
                    if dwell_frames >= self.dwell_confirm_frames and not st["dwell_emitted"]:
                        st["dwell_emitted"] = True
                        evt = self._make_event(
                            event_type="dwell_region",
                            frame_id=frame_id,
                            timestamp=timestamp,
                            hand_label=label,
                            active_label=active_label,
                            non_active_label=non_active,
                            visible_labels=visible_labels,
                            hand_scope=scope,
                            region_id=region_id,
                            hand_velocity=hand_velocity,
                            dwell_frames=dwell_frames,
                            dwell_time_s=float(timestamp - float(st["enter_ts"])) if st["enter_ts"] is not None else None,
                        )
                        if self._allow_emit(evt, frame_id):
                            events.append(evt)
                            events.extend(self._candidate_events(evt, hand_velocity, frame_id))

                if st["in_region"] and st["absent_count"] >= (self.exit_confirm_frames + self.occlusion_tolerance_frames):
                    st["in_region"] = False
                    st["present_count"] = 0
                    st["dwell_emitted"] = False
                    evt = self._make_event(
                        event_type="exit_region",
                        frame_id=frame_id,
                        timestamp=timestamp,
                        hand_label=label,
                        active_label=active_label,
                        non_active_label=non_active,
                        visible_labels=visible_labels,
                        hand_scope=scope,
                        region_id=region_id,
                        hand_velocity=hand_velocity,
                    )
                    if self._allow_emit(evt, frame_id):
                        events.append(evt)

        return events

    def _make_event(
        self,
        *,
        event_type: str,
        frame_id: int,
        timestamp: float,
        hand_label: str,
        active_label: str | None,
        non_active_label: str | None,
        visible_labels: List[str],
        hand_scope: str,
        region_id: str,
        hand_velocity: float,
        dwell_frames: int | None = None,
        dwell_time_s: float | None = None,
    ) -> dict:
        roles = [
            role
            for role in self.registry.regions[region_id].roles
            if self.registry.resolve_role_enabled(region_id, self.operation_mode, role)
        ]
        evt = {
            "type": event_type,
            "t": float(timestamp),
            "frame_id": int(frame_id),
            "hand_label": str(hand_label),
            "active_hand": active_label,
            "non_active_hand": non_active_label,
            "active_is_event_hand": bool(active_label == hand_label) if active_label is not None else False,
            "visible_hands": list(visible_labels),
            "hand_scope": str(hand_scope),
            "region_id": str(region_id),
            "roles": roles,
            "velocity": float(hand_velocity),
        }
        if dwell_frames is not None:
            evt["dwell_frames"] = int(dwell_frames)
        if dwell_time_s is not None:
            evt["dwell_time_s"] = float(dwell_time_s)
        return evt

    def _allow_emit(self, event: dict, frame_id: int) -> bool:
        key = (event["hand_label"], event["region_id"], event["type"])
        last = self._last_emit_frame.get(key)
        if last is not None and (int(frame_id) - int(last)) <= self.event_gap_frames:
            return False
        self._last_emit_frame[key] = int(frame_id)
        return True

    def _candidate_events(self, parent_event: dict, velocity: float, frame_id: int) -> List[dict]:
        roles = set(parent_event.get("roles", []))
        out = []
        base = {
            "t": parent_event["t"],
            "frame_id": parent_event["frame_id"],
            "hand_label": parent_event["hand_label"],
            "active_hand": parent_event.get("active_hand"),
            "non_active_hand": parent_event.get("non_active_hand"),
            "active_is_event_hand": parent_event.get("active_is_event_hand"),
            "visible_hands": parent_event.get("visible_hands", []),
            "region_id": parent_event["region_id"],
            "roles": parent_event.get("roles", []),
            "velocity": float(velocity),
            "parent_event": parent_event["type"],
        }

        is_pick_role = bool({"part_pick", "tool_pick"} & roles)
        is_drop_role = bool({"part_drop", "tool_drop"} & roles)

        if is_pick_role and velocity >= self.pick_velocity_min:
            ev = dict(base)
            ev["type"] = "pick_candidate"
            if self._allow_emit(ev, frame_id):
                out.append(ev)

        # Drop candidates are more robust when emitted on dwell confirmation.
        if is_drop_role and parent_event["type"] == "dwell_region":
            if velocity <= self.drop_velocity_max or velocity <= self.pick_velocity_min:
                ev = dict(base)
                ev["type"] = "drop_candidate"
                if self._allow_emit(ev, frame_id):
                    out.append(ev)
        return out

    def build_stable_region_hits(self, frame_data: dict, *, hand_scope: str = "any") -> dict:
        """
        Build debounced/hysteresis-stabilized region hits from extractor internal state.
        These hits are suitable for FSM gating and reduce one-frame toggle noise.
        """
        scope = str(hand_scope).strip().lower()
        if scope not in {"any", "active"}:
            scope = "any"

        region_hits = frame_data.get("region_hits") or {}
        active_label = region_hits.get("active_label")
        hands = frame_data.get("hands") or {}

        labels = ("LEFT", "RIGHT")
        if scope == "active":
            labels = (active_label,) if active_label in {"LEFT", "RIGHT"} else tuple()

        hits_by_hand = {}
        any_regions = set()
        any_roles = set()

        for label in ("LEFT", "RIGHT"):
            hb = hands.get(label.lower()) or {}
            hand_visible = bool(hb.get("visible", False))
            regions = set()
            roles = set()
            if label in labels:
                for region_id, region in self.registry.regions.items():
                    st = self._state.get((label, region_id))
                    if st is None or not bool(st.get("in_region", False)):
                        continue
                    # Keep semantic region state tied to currently visible hands.
                    if not hand_visible:
                        continue
                    regions.add(region_id)
                    for role in region.roles:
                        if self.registry.resolve_role_enabled(region_id, self.operation_mode, role):
                            roles.add(role)
                            any_roles.add(role)
                    any_regions.add(region_id)
            hits_by_hand[label] = {"regions": sorted(regions), "roles": sorted(roles)}

        active = {"regions": [], "roles": []}
        if active_label in {"LEFT", "RIGHT"}:
            active = hits_by_hand.get(active_label, {"regions": [], "roles": []})

        return {
            "mode": self.operation_mode,
            "active_label": active_label,
            "hands": hits_by_hand,
            "active": active,
            "any": {"regions": sorted(any_regions), "roles": sorted(any_roles)},
            "source": "event_stable",
            "hand_scope": scope,
        }
