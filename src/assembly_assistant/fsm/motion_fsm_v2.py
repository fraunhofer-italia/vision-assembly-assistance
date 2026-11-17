from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set

from assembly_assistant.pipeline.stage_region_validator import StageRegionContractValidator
from assembly_assistant.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FrameContext:
    in_work: bool
    in_drop: bool
    visible: bool
    activity: float
    active_regions: Set[str]
    any_regions: Set[str]
    active_roles: Set[str]
    any_roles: Set[str]


class BaseSegmentationPolicy:
    @staticmethod
    def _resolve_scope(stage_descriptor: dict | None, phase: str) -> str:
        desc = stage_descriptor or {}
        scoped_key = f"{phase}_hand_scope"
        scope = str(desc.get(scoped_key, desc.get("hand_scope", "any"))).strip().lower()
        if scope not in {"active", "any"}:
            scope = "any"
        return scope

    @staticmethod
    def _ctx_roles_regions(ctx: FrameContext, *, scope: str):
        if scope == "active":
            return set(ctx.active_roles), set(ctx.active_regions)
        return set(ctx.any_roles), set(ctx.any_regions)

    @classmethod
    def _in_dead_band(cls, ctx: FrameContext, stage_descriptor: dict | None, phase: str) -> bool:
        desc = stage_descriptor or {}
        scope = cls._resolve_scope(desc, phase)
        _, regions = cls._ctx_roles_regions(ctx, scope=scope)
        return "dead_band_zone" in regions

    @classmethod
    def _support_hit(cls, ctx: FrameContext, stage_descriptor: dict | None, phase: str) -> bool:
        desc = stage_descriptor or {}
        support_regions = set(desc.get(f"{phase}_support_regions", []) or desc.get("support_regions", []) or [])
        if not support_regions:
            return False
        scope = cls._resolve_scope(desc, phase)
        roles, regions = cls._ctx_roles_regions(ctx, scope=scope)
        support_roles = set(desc.get(f"{phase}_support_roles", []) or [])
        if support_roles:
            return bool((support_regions & regions) and (support_roles & roles))
        return bool(support_regions & regions)

    def should_start(self, ctx: FrameContext, stage_descriptor: dict | None = None) -> bool:
        raise NotImplementedError

    def completion_signal(self, ctx: FrameContext, stage_descriptor: dict | None = None) -> bool:
        raise NotImplementedError

    def requires_clear_rearm(self, stage_descriptor: dict | None = None) -> bool:
        return True


class DismantlePolicy(BaseSegmentationPolicy):
    """
    Dismantle completion anchor: drop-like region events.
    """

    def __init__(self, completion_roles: List[str] | None = None):
        self.completion_roles = set(completion_roles or ["part_drop", "tool_drop"])

    def should_start(self, ctx: FrameContext, stage_descriptor: dict | None = None) -> bool:
        desc = stage_descriptor or {}
        if self._in_dead_band(ctx, desc, "start"):
            return False
        scope = self._resolve_scope(desc, "start")
        roles, regions = self._ctx_roles_regions(ctx, scope=scope)
        manual_start_allowed = bool(desc.get("manual_start_allowed", True))
        start_tool_regions = set(desc.get("start_tool_regions", []) or [])
        if start_tool_regions:
            if "tool_pick" in roles and bool(start_tool_regions & regions):
                return True
            return manual_start_allowed and bool(ctx.visible)
        return bool(ctx.visible) if manual_start_allowed else False

    def completion_signal(self, ctx: FrameContext, stage_descriptor: dict | None = None) -> bool:
        desc = stage_descriptor or {}
        if self._in_dead_band(ctx, desc, "completion"):
            return False
        scope = self._resolve_scope(desc, "completion")
        roles, regions = self._ctx_roles_regions(ctx, scope=scope)
        completion_roles = set(desc.get("completion_roles", [])) or set(self.completion_roles)
        completion_regions = set(desc.get("completion_regions", []) or [])

        if completion_regions:
            return bool((completion_roles & roles) and (completion_regions & regions))
        return bool(ctx.in_drop or bool(completion_roles & roles))


class AssemblyPolicy(BaseSegmentationPolicy):
    """
    Assembly completion anchor: next pick/tool-drop style events.
    """

    def __init__(
        self,
        trigger_roles: List[str] | None = None,
        completion_roles: List[str] | None = None,
    ):
        self.trigger_roles = set(trigger_roles or ["part_pick", "tool_pick"])
        self.completion_roles = set(completion_roles or ["part_pick", "tool_drop"])

    def should_start(self, ctx: FrameContext, stage_descriptor: dict | None = None) -> bool:
        desc = stage_descriptor or {}
        if self._in_dead_band(ctx, desc, "start"):
            return False
        scope = self._resolve_scope(desc, "start")
        roles, regions = self._ctx_roles_regions(ctx, scope=scope)
        start_roles = set(desc.get("start_roles", [])) or set(self.trigger_roles)
        start_source_regions = set(desc.get("start_source_regions", []) or [])
        manual_start_allowed = bool(desc.get("manual_start_allowed", False))

        if start_source_regions:
            if (start_roles & roles) and (start_source_regions & regions):
                return True
            return manual_start_allowed and bool(ctx.visible and ctx.in_work)

        if start_roles & roles:
            return True
        return bool(manual_start_allowed and ctx.in_work and ctx.visible)

    def completion_signal(self, ctx: FrameContext, stage_descriptor: dict | None = None) -> bool:
        desc = stage_descriptor or {}
        if self._in_dead_band(ctx, desc, "completion"):
            return False
        scope = self._resolve_scope(desc, "completion")
        roles, regions = self._ctx_roles_regions(ctx, scope=scope)
        mode = str(desc.get("completion_mode", "next_stage_pick")).lower()

        if mode == "terminal_end":
            return False

        if mode == "next_stage_pick":
            next_roles = set(desc.get("next_stage_trigger_roles", [])) or {"part_pick"}
            next_regions = set(desc.get("next_stage_source_regions", []) or [])
            if next_regions:
                return bool((next_roles & roles) and (next_regions & regions))
            return bool(next_roles & roles)

        # completion_proxy mode
        completion_roles = set(desc.get("completion_roles", [])) or set(self.completion_roles)
        completion_regions = set(desc.get("completion_regions", []) or [])
        if completion_regions:
            return bool((completion_roles & roles) and (completion_regions & regions))
        return bool((completion_roles & roles) or ctx.in_drop)

    def requires_clear_rearm(self, stage_descriptor: dict | None = None) -> bool:
        # Assembly should allow immediate next-stage picks after cooldown.
        return False


class MotionFSMV2:
    """
    Shared state engine with mode-specific policy hooks.

    States:
      WAIT_TRIGGER -> RECORDING -> WAIT_COMPLETION_CONFIRM -> COOLDOWN -> WAIT_TRIGGER
    """

    WAIT_TRIGGER = "WAIT_TRIGGER"
    RECORDING = "RECORDING"
    WAIT_COMPLETION_CONFIRM = "WAIT_COMPLETION_CONFIRM"
    COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        *,
        policy: BaseSegmentationPolicy,
        stage_descriptors: List[dict] | None = None,
        work_region=None,
        drop_region=None,
        trigger_point_mode="wrist",
        warmup_frames=20,
        min_consecutive_visible=3,
        start_no_activity_frames=10,
        start_recent_hit_window=12,
        start_recent_hit_count=3,
        start_handoff_frames=2,
        require_anchor_before_completion=True,
        anchor_startup_grace_frames=20,
        strict_motion_hits_only=True,
        min_consecutive_completion_hits=2,
        completion_confirm_frames=2,
        min_recording_frames=15,
        max_hand_jump=0.35,
        activity_v_min=0.003,
        occlusion_tolerance_frames=15,
        cooldown_frames=8,
        clear_rearm_frames=1,
        reject_log_min_gap_frames=12,
        contract_validator: StageRegionContractValidator | None = None,
        use_depth_triggers: bool = False,
        depth_fallback_rgb: bool = True,
        use_motion_triggers: bool = False,
    ):
        self.policy = policy
        self.stage_descriptors = list(stage_descriptors or [])
        self.current_stage_index = 0
        self.state = self.WAIT_TRIGGER
        self.contract_validator = contract_validator or StageRegionContractValidator()
        self.trigger_point_mode = trigger_point_mode
        self.use_depth_triggers = bool(use_depth_triggers)
        self.depth_fallback_rgb = bool(depth_fallback_rgb)
        self.use_motion_triggers = bool(use_motion_triggers)
        self._stage_name_to_index = self._build_stage_index()
        self._unordered_groups = self._build_unordered_groups()
        self._unordered_completed: Dict[str, Set[str]] = {gid: set() for gid in self._unordered_groups}

        self.warmup_frames = int(max(0, warmup_frames))
        self.min_consecutive_visible = int(max(1, min_consecutive_visible))
        self.start_no_activity_frames = int(max(1, start_no_activity_frames))
        self.start_recent_hit_window = int(max(0, start_recent_hit_window))
        self.start_recent_hit_count = int(max(0, start_recent_hit_count))
        self.start_handoff_frames = int(max(0, start_handoff_frames))
        self.require_anchor_before_completion = bool(require_anchor_before_completion)
        self.anchor_startup_grace_frames = int(max(0, anchor_startup_grace_frames))
        self.strict_motion_hits_only = bool(strict_motion_hits_only)
        self.min_consecutive_completion_hits = int(max(1, min_consecutive_completion_hits))
        self.completion_confirm_frames = int(max(1, completion_confirm_frames))
        self.min_recording_frames = int(max(0, min_recording_frames))
        self.max_hand_jump = float(max(0.0, max_hand_jump))
        self.activity_v_min = float(max(0.0, activity_v_min))
        self.occlusion_tolerance_frames = int(max(0, occlusion_tolerance_frames))
        self.cooldown_frames = int(max(0, cooldown_frames))
        self.clear_rearm_frames = int(max(0, clear_rearm_frames))
        self.reject_log_min_gap_frames = int(max(0, reject_log_min_gap_frames))

        self.WORK_REGION = self._parse_region(work_region or (0.2, 0.2, 0.6, 0.6))
        self.DROP_REGION = self._parse_region(drop_region or (0.7, 0.7, 0.95, 0.95))

        self._frames_seen = 0
        self._work_hits = 0
        self._work_visible_hits = 0
        self._recent_start_hit_frames = deque()
        self._handoff_start_ctx: FrameContext | None = None
        self._handoff_start_frames_left = 0
        self._handoff_force_start = False
        self._recording_anchor_points = 0
        self._anchor_startup_grace_left = 0
        self._completion_hits = 0
        self._recording_frames = 0
        self._recording_start_ts = None
        self._occlusion_count = 0
        self._completion_confirm_count = 0
        self._cooldown_left = 0
        self._clear_rearm_left = 0
        self._segment_closed = False
        self._last_wrist = None
        self._contract_rejections: List[dict] = []
        self._last_reject_log_frame: Dict[tuple, int] = {}
        self._last_reject_log_key_no_frame: tuple | None = None
        self._transition_reasons: List[dict] = []
        self._pending_completion_reason: dict | None = None
        self._pending_completion_stage_desc: dict | None = None
        self._depth_fallback_count = 0

    def reset(self):
        self.state = self.WAIT_TRIGGER
        self.current_stage_index = 0
        self._work_hits = 0
        self._work_visible_hits = 0
        self._recent_start_hit_frames = deque()
        self._handoff_start_ctx = None
        self._handoff_start_frames_left = 0
        self._handoff_force_start = False
        self._recording_anchor_points = 0
        self._anchor_startup_grace_left = 0
        self._completion_hits = 0
        self._recording_frames = 0
        self._recording_start_ts = None
        self._occlusion_count = 0
        self._completion_confirm_count = 0
        self._cooldown_left = 0
        self._clear_rearm_left = 0
        self._segment_closed = False
        self._last_wrist = None
        self._contract_rejections = []
        self._last_reject_log_frame = {}
        self._last_reject_log_key_no_frame = None
        self._transition_reasons = []
        self._pending_completion_reason = None
        self._pending_completion_stage_desc = None
        self._unordered_completed = {gid: set() for gid in self._unordered_groups}
        self._depth_fallback_count = 0

    def consume_segment_closed(self) -> bool:
        out = bool(self._segment_closed)
        self._segment_closed = False
        return out

    def consume_contract_rejections(self) -> List[dict]:
        out = list(self._contract_rejections)
        self._contract_rejections = []
        return out

    def consume_transition_reasons(self) -> List[dict]:
        out = list(self._transition_reasons)
        self._transition_reasons = []
        return out

    def consume_depth_fallback_count(self) -> int:
        out = int(self._depth_fallback_count)
        self._depth_fallback_count = 0
        return out

    def is_waiting_for_first_anchor(self) -> bool:
        return bool(
            self.state == self.RECORDING
            and self.require_anchor_before_completion
            and self._recording_anchor_points <= 0
        )

    def abort_recording_keep_stage(self) -> None:
        self.state = self.WAIT_TRIGGER
        self._reset_start_gating()
        self._clear_handoff_start()
        self._recording_anchor_points = 0
        self._anchor_startup_grace_left = 0
        self._completion_hits = 0
        self._recording_frames = 0
        self._recording_start_ts = None
        self._occlusion_count = 0
        self._completion_confirm_count = 0
        self._cooldown_left = 0
        self._clear_rearm_left = 0
        self._segment_closed = False
        self._pending_completion_reason = None
        self._pending_completion_stage_desc = None

    def _reset_start_gating(self) -> None:
        self._work_hits = 0
        self._work_visible_hits = 0
        self._recent_start_hit_frames = deque()

    def _clear_handoff_start(self) -> None:
        self._handoff_start_ctx = None
        self._handoff_start_frames_left = 0
        self._handoff_force_start = False

    def _stage_uses_implicit_dismantle_start(self, desc: dict | None) -> bool:
        if not isinstance(self.policy, DismantlePolicy):
            return False
        desc = desc or {}
        if not bool(desc.get("manual_start_allowed", True)):
            return False
        if desc.get("start_tool_regions", []) or desc.get("start_source_regions", []):
            return False
        return True

    def _record_recent_start_hit(self, frame_id: int | None, window: int) -> int:
        if window <= 0:
            return 0
        if frame_id is None:
            frame_id = int(self._frames_seen)
        self._recent_start_hit_frames.append(int(frame_id))
        cutoff = int(frame_id) - int(window) + 1
        while self._recent_start_hit_frames and int(self._recent_start_hit_frames[0]) < cutoff:
            self._recent_start_hit_frames.popleft()
        return len(self._recent_start_hit_frames)

    def _prune_recent_start_hits(self, frame_id: int | None, window: int) -> int:
        if window <= 0:
            self._recent_start_hit_frames = deque()
            return 0
        if frame_id is None:
            frame_id = int(self._frames_seen)
        cutoff = int(frame_id) - int(window) + 1
        while self._recent_start_hit_frames and int(self._recent_start_hit_frames[0]) < cutoff:
            self._recent_start_hit_frames.popleft()
        return len(self._recent_start_hit_frames)

    def _log_contract_reject(self, *, phase: str, frame_data: dict, stage_desc: dict | None, reason_code: str, reason: str):
        key = (
            str(phase),
            str(reason_code),
            (stage_desc or {}).get("stage_name"),
            int(self.current_stage_index),
            str(self.state),
        )
        frame_id = frame_data.get("frame_id")
        try:
            frame_id_i = int(frame_id)
        except Exception:
            frame_id_i = None

        if frame_id_i is not None:
            last = self._last_reject_log_frame.get(key)
            if last is not None and (frame_id_i - int(last)) < self.reject_log_min_gap_frames:
                return
            self._last_reject_log_frame[key] = frame_id_i
        else:
            if self._last_reject_log_key_no_frame == key:
                return
            self._last_reject_log_key_no_frame = key

        self._contract_rejections.append(
            {
                "phase": str(phase),
                "reason_code": str(reason_code),
                "reason": str(reason),
                "frame_id": frame_data.get("frame_id"),
                "timestamp": frame_data.get("timestamp"),
                "stage_name": (stage_desc or {}).get("stage_name"),
                "stage_index": int(self.current_stage_index),
                "state": str(self.state),
            }
        )

    def set_stage_index(self, stage_index: int) -> None:
        self.current_stage_index = int(max(0, stage_index))
        self._unordered_completed = {gid: set() for gid in self._unordered_groups}

    def _current_stage_descriptor(self) -> dict | None:
        if not self.stage_descriptors:
            return None
        idx = int(max(0, min(self.current_stage_index, len(self.stage_descriptors) - 1)))
        return self.stage_descriptors[idx]

    def _build_stage_index(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for idx, desc in enumerate(self.stage_descriptors):
            name = str(desc.get("stage_name", "")).strip()
            if name:
                out[name] = idx
        return out

    def _build_unordered_groups(self) -> Dict[str, dict]:
        groups: Dict[str, dict] = {}
        for idx, desc in enumerate(self.stage_descriptors):
            group_id = str(desc.get("unordered_group_id", "")).strip()
            if not group_id:
                continue
            members = [str(x).strip() for x in (desc.get("unordered_group_members") or []) if str(x).strip()]
            if not members:
                members = [str(desc.get("stage_name", "")).strip()]
            grp = groups.setdefault(group_id, {"members": [], "indices": set()})
            for m in members:
                if m and m not in grp["members"]:
                    grp["members"].append(m)
            grp["indices"].add(idx)
        for gid, grp in groups.items():
            indices = sorted(int(x) for x in grp.get("indices", []) if x is not None)
            grp["indices"] = indices
            grp["after_index"] = (max(indices) + 1) if indices else None
        return groups

    def _unordered_remaining_descs(self, group_id: str) -> List[dict]:
        group = self._unordered_groups.get(group_id)
        if not group:
            return []
        completed = self._unordered_completed.get(group_id, set())
        out: List[dict] = []
        for name in group.get("members", []):
            if name in completed:
                continue
            idx = self._stage_name_to_index.get(name)
            if idx is None:
                continue
            out.append(self.stage_descriptors[idx])
        return out

    def _match_unordered_completion(self, ctx: FrameContext, stage_desc: dict | None):
        group_id = str((stage_desc or {}).get("unordered_group_id", "")).strip()
        if not group_id or group_id not in self._unordered_groups:
            return None, None
        for candidate_desc in self._unordered_remaining_descs(group_id):
            if not self.policy.completion_signal(ctx, candidate_desc):
                continue
            v = self.contract_validator.validate_completion(
                mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                ctx=ctx,
                stage_descriptor=candidate_desc,
            )
            if v.allowed:
                return candidate_desc, v
        return None, None

    def _match_unordered_start(self, ctx: FrameContext, stage_desc: dict | None):
        group_id = str((stage_desc or {}).get("unordered_group_id", "")).strip()
        if not group_id or group_id not in self._unordered_groups:
            return None, None
        for candidate_desc in self._unordered_remaining_descs(group_id):
            if not self.policy.should_start(ctx, candidate_desc):
                continue
            v = self.contract_validator.validate_start(
                mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                ctx=ctx,
                stage_descriptor=candidate_desc,
            )
            if v.allowed:
                return candidate_desc, v
        return None, None

    def _parse_region(self, region):
        if isinstance(region, tuple):
            return {"x_min": region[0], "y_min": region[1], "x_max": region[2], "y_max": region[3]}
        return region

    def _in_region(self, x, y, region):
        return region["x_min"] <= x <= region["x_max"] and region["y_min"] <= y <= region["y_max"]

    def _resolve_trigger_point(self, hand_block):
        if self.trigger_point_mode == "wrist":
            return hand_block["wrist"]
        keypoints = hand_block.get("keypoints")
        if keypoints is not None and len(keypoints) > 5:
            wrist = keypoints[0]
            index_mcp = keypoints[5]
            return (float((wrist[0] + index_mcp[0]) * 0.5), float((wrist[1] + index_mcp[1]) * 0.5))
        return hand_block["wrist"]

    def _iter_visible_hands(self, frame_data):
        hands = frame_data.get("hands")
        if isinstance(hands, dict) and ("left" in hands or "right" in hands):
            for key in ("left", "right"):
                hb = hands.get(key)
                if hb and hb.get("visible", False):
                    yield hb
            return
        hand = frame_data.get("hand")
        if hand and hand.get("visible", False):
            yield hand

    def _build_context(self, frame_data) -> FrameContext:
        in_work = False
        in_drop = False
        visible = False
        activity = 0.0
        velocity_observed = False
        for hb in self._iter_visible_hands(frame_data):
            visible = True
            x, y = self._resolve_trigger_point(hb)
            in_work = in_work or self._in_region(x, y, self.WORK_REGION)
            in_drop = in_drop or self._in_region(x, y, self.DROP_REGION)
            if "velocity" in hb:
                velocity_observed = True
            activity = max(activity, float(hb.get("velocity", 0.0) or 0.0))
        if visible and not velocity_observed:
            activity = max(activity, self.activity_v_min + 1e-6)

        if self.use_motion_triggers and self.strict_motion_hits_only:
            region_hits = frame_data.get("motion_region_hits") or {}
        else:
            region_hits = (
                frame_data.get("motion_region_hits")
                or frame_data.get("fsm_region_hits")
                or frame_data.get("region_hits")
                or {}
            )
        depth_hits = frame_data.get("depth_region_hits") or {}

        if self.use_motion_triggers:
            active_roles = set((region_hits.get("active") or {}).get("roles", []))
            any_roles = set((region_hits.get("any") or {}).get("roles", []))
            active_regions = set((region_hits.get("active") or {}).get("regions", []))
            any_regions = set((region_hits.get("any") or {}).get("regions", []))
            # ROI-only segmentation: ignore hand visibility/position for FSM gating.
            if any_regions or any_roles:
                visible = True
                in_work = True
                in_drop = False
                activity = max(activity, self.activity_v_min + 1e-6)
            else:
                visible = False
                in_work = False
                in_drop = False
                activity = 0.0
        elif self.use_depth_triggers:
            any_roles = set(depth_hits.get("roles", []))
            any_regions = set(depth_hits.get("regions", []))
            active_roles = set(any_roles)
            active_regions = set(any_regions)
            if self.depth_fallback_rgb and not any_regions:
                active_roles = set((region_hits.get("active") or {}).get("roles", []))
                any_roles = set((region_hits.get("any") or {}).get("roles", []))
                active_regions = set((region_hits.get("active") or {}).get("regions", []))
                any_regions = set((region_hits.get("any") or {}).get("regions", []))
                if any_regions or any_roles:
                    self._depth_fallback_count += 1
        else:
            active_roles = set((region_hits.get("active") or {}).get("roles", []))
            any_roles = set((region_hits.get("any") or {}).get("roles", []))
            active_regions = set((region_hits.get("active") or {}).get("regions", []))
            any_regions = set((region_hits.get("any") or {}).get("regions", []))
        return FrameContext(
            in_work=in_work,
            in_drop=in_drop,
            visible=visible,
            activity=activity,
            active_regions={str(r) for r in active_regions},
            any_regions={str(r) for r in any_regions},
            active_roles={str(r) for r in active_roles},
            any_roles={str(r) for r in any_roles},
        )

    def update(self, frame_data):
        self._frames_seen += 1
        if self._frames_seen <= self.warmup_frames:
            return

        self._segment_closed = False
        ctx = self._build_context(frame_data)
        stage_desc = self._current_stage_descriptor()

        # Conservative jump filter only for WAIT_TRIGGER.
        hand = frame_data.get("hand") or {}
        jump_spike = False
        if not self.use_motion_triggers and hand.get("visible", False):
            x, y = self._resolve_trigger_point(hand)
            if self._last_wrist is not None and self.max_hand_jump > 0.0 and self.state == self.WAIT_TRIGGER:
                dx = x - self._last_wrist[0]
                dy = y - self._last_wrist[1]
                if (dx * dx + dy * dy) ** 0.5 > self.max_hand_jump:
                    jump_spike = True
            if not jump_spike:
                self._last_wrist = (x, y)
        elif not self.use_motion_triggers:
            self._last_wrist = None

        if self.state == self.COOLDOWN:
            if self._cooldown_left > 0:
                self._cooldown_left -= 1
                return
            self.state = self.WAIT_TRIGGER
            if self._handoff_force_start:
                self._clear_rearm_left = 0
            elif self.policy.requires_clear_rearm(stage_desc):
                self._clear_rearm_left = self.clear_rearm_frames
            else:
                self._clear_rearm_left = 0
            # Continue processing this frame in WAIT_TRIGGER for near-instant transitions.

        if self.state == self.WAIT_TRIGGER:
            if self._clear_rearm_left > 0:
                # Require completion signal to clear briefly before retrigger.
                completion_candidate = self.policy.completion_signal(ctx, stage_desc)
                if completion_candidate:
                    v = self.contract_validator.validate_completion(
                        mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                        ctx=ctx,
                        stage_descriptor=stage_desc,
                    )
                    if v.allowed:
                        self._clear_rearm_left = self.clear_rearm_frames
                    else:
                        self._clear_rearm_left -= 1
                else:
                    self._clear_rearm_left -= 1
                return

            if jump_spike:
                self._reset_start_gating()
                return

            start_desc = stage_desc
            implicit_dismantle_start = self._stage_uses_implicit_dismantle_start(stage_desc)
            reference_stage_autostart = bool(frame_data.get("reference_stage_autostart", False))
            auto_start_force = bool(
                implicit_dismantle_start
                and (
                    int(self.current_stage_index) == 0
                    or reference_stage_autostart
                )
            )
            handoff_force_start = False
            start_candidate = bool(auto_start_force) if implicit_dismantle_start else self.policy.should_start(ctx, stage_desc)
            start_validation = None
            handoff_used = False
            if not start_candidate:
                unordered_desc, unordered_v = self._match_unordered_start(ctx, stage_desc)
                if unordered_desc is not None and unordered_v is not None:
                    start_candidate = True
                    start_desc = unordered_desc
                    start_validation = unordered_v
            if (
                not start_candidate
                and self._handoff_start_ctx is not None
                and self._handoff_start_frames_left > 0
                and stage_desc is not None
            ):
                if implicit_dismantle_start and self._handoff_force_start:
                    handoff_candidate = True
                    handoff_force_start = True
                else:
                    handoff_candidate = self.policy.should_start(self._handoff_start_ctx, stage_desc)
                if handoff_candidate:
                    handoff_validation = self.contract_validator.validate_start(
                        mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                        ctx=self._handoff_start_ctx,
                        stage_descriptor=stage_desc,
                    )
                    if handoff_validation.allowed:
                        start_candidate = True
                        start_desc = stage_desc
                        start_validation = handoff_validation
                        handoff_used = True
            if start_candidate:
                if start_validation is None:
                    validation_ctx = self._handoff_start_ctx if handoff_force_start and self._handoff_start_ctx is not None else ctx
                    v = self.contract_validator.validate_start(
                        mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                        ctx=validation_ctx,
                        stage_descriptor=start_desc,
                    )
                    start_validation = v
                if not start_validation.allowed:
                    self._log_contract_reject(
                        phase="start",
                        frame_data=frame_data,
                        stage_desc=start_desc,
                        reason_code=start_validation.reason_code,
                        reason=start_validation.reason,
                    )
                    start_candidate = False
            if start_candidate:
                if ctx.activity >= self.activity_v_min:
                    self._work_hits += 1
                else:
                    self._work_hits = 0

                if ctx.visible:
                    self._work_visible_hits += 1
                else:
                    self._work_visible_hits = 0
            else:
                self._work_hits = 0
                self._work_visible_hits = 0

            stage_min_consecutive_start_hits = int(
                (start_desc or {}).get("min_consecutive_start_hits", self.min_consecutive_visible)
            )
            stage_start_no_activity_frames = int(
                (start_desc or {}).get("start_no_activity_frames", self.start_no_activity_frames)
            )
            stage_start_recent_hit_window = int(
                (start_desc or {}).get("start_recent_hit_window", self.start_recent_hit_window)
            )
            stage_start_recent_hit_count = int(
                (start_desc or {}).get("start_recent_hit_count", self.start_recent_hit_count)
            )
            frame_id = frame_data.get("frame_id")
            if start_candidate:
                recent_hit_count = self._record_recent_start_hit(frame_id, stage_start_recent_hit_window)
            else:
                recent_hit_count = self._prune_recent_start_hits(frame_id, stage_start_recent_hit_window)
                if self._handoff_start_frames_left > 0:
                    self._handoff_start_frames_left -= 1
                    if self._handoff_start_frames_left <= 0:
                        self._clear_handoff_start()
            trigger_by_recent_hits = (
                stage_start_recent_hit_window > 0
                and stage_start_recent_hit_count > 0
                and recent_hit_count >= stage_start_recent_hit_count
            )
            if (
                self._work_hits >= stage_min_consecutive_start_hits
                or self._work_visible_hits >= stage_start_no_activity_frames
                or trigger_by_recent_hits
                or auto_start_force
                or handoff_force_start
            ):
                if start_desc is not None:
                    start_name = str(start_desc.get("stage_name", "")).strip()
                    if start_name in self._stage_name_to_index:
                        self.current_stage_index = int(self._stage_name_to_index[start_name])
                self.state = self.RECORDING
                self._recording_frames = 0
                self._recording_anchor_points = 0
                self._anchor_startup_grace_left = int(self.anchor_startup_grace_frames)
                self._recording_start_ts = frame_data.get("timestamp")
                self._occlusion_count = 0
                self._completion_hits = 0
                self._pending_completion_reason = None
                if auto_start_force:
                    trigger_reason_code = "OK_START_AUTO"
                    trigger_reason = (
                        "Reference dismantle stage auto-started from clip context."
                        if reference_stage_autostart and int(self.current_stage_index) != 0
                        else "First manual dismantle stage auto-started after warmup."
                    )
                elif trigger_by_recent_hits:
                    trigger_reason_code = "OK_START_RECENT_WINDOW"
                    trigger_reason = (
                        f"Start condition satisfied from {recent_hit_count} validated hits "
                        f"within the last {stage_start_recent_hit_window} frames."
                    )
                elif handoff_used:
                    trigger_reason_code = "OK_START_HANDOFF"
                    trigger_reason = (
                        "Start condition satisfied from previous stage completion boundary."
                        if handoff_force_start
                        else "Start condition satisfied from previous stage next-pick handoff."
                    )
                else:
                    trigger_reason_code = start_validation.reason_code if start_validation is not None else "OK_START"
                    trigger_reason = start_validation.reason if start_validation is not None else "Start condition satisfied."
                self._transition_reasons.append(
                    {
                        "kind": "trigger",
                        "reason_code": trigger_reason_code,
                        "reason": trigger_reason,
                        "frame_id": frame_data.get("frame_id"),
                        "timestamp": frame_data.get("timestamp"),
                        "stage_name": (start_desc or {}).get("stage_name"),
                        "stage_index": int(self.current_stage_index),
                    }
                )
                self._reset_start_gating()
                self._clear_handoff_start()
                logger.info("[FSMv2] WAIT_TRIGGER -> RECORDING")
            return

        if self.state == self.RECORDING:
            self._recording_frames += 1
            anchor_valid = frame_data.get("anchor_valid")
            if anchor_valid is None:
                anchor_valid = bool((frame_data.get("hand") or {}).get("visible", False))
            if bool(anchor_valid):
                self._recording_anchor_points += 1
                self._anchor_startup_grace_left = 0
            if not self.use_motion_triggers:
                if not ctx.visible:
                    self._occlusion_count += 1
                else:
                    self._occlusion_count = 0
                if self._occlusion_count > self.occlusion_tolerance_frames:
                    self.state = self.WAIT_TRIGGER
                    self._reset_start_gating()
                    self._recording_anchor_points = 0
                    self._completion_hits = 0
                    return

            if self.require_anchor_before_completion and self._recording_anchor_points <= 0:
                if self._anchor_startup_grace_left > 0:
                    self._anchor_startup_grace_left -= 1
                self._completion_hits = 0
                self._pending_completion_stage_desc = None
                self._pending_completion_reason = None
                return

            stage_min_valid_seconds = float((stage_desc or {}).get("min_stage_valid_time", 0.0) or 0.0)
            if stage_min_valid_seconds > 0.0:
                ts = frame_data.get("timestamp")
                if self._recording_start_ts is None and ts is not None:
                    self._recording_start_ts = ts
                if ts is not None and self._recording_start_ts is not None:
                    if float(ts) - float(self._recording_start_ts) < stage_min_valid_seconds:
                        self._completion_hits = 0
                        self._pending_completion_stage_desc = None
                        self._pending_completion_reason = None
                        return

            completion_desc = stage_desc
            completion_validation = None
            completion_candidate = self.policy.completion_signal(ctx, stage_desc)
            # Log explicit wrong-region interactions deterministically even when
            # they don't satisfy policy completion candidate.
            pre_v = self.contract_validator.validate_completion(
                mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                ctx=ctx,
                stage_descriptor=stage_desc,
            )
            if (not pre_v.allowed) and ("WRONG_REGION" in str(pre_v.reason_code)):
                self._log_contract_reject(
                    phase="completion",
                    frame_data=frame_data,
                    stage_desc=stage_desc,
                    reason_code=pre_v.reason_code,
                    reason=pre_v.reason,
                )
            if not completion_candidate:
                unordered_desc, unordered_v = self._match_unordered_completion(ctx, stage_desc)
                if unordered_desc is not None and unordered_v is not None:
                    completion_candidate = True
                    completion_desc = unordered_desc
                    completion_validation = unordered_v
            if completion_candidate:
                v = completion_validation or self.contract_validator.validate_completion(
                    mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                    ctx=ctx,
                    stage_descriptor=completion_desc,
                )
                if not v.allowed:
                    self._log_contract_reject(
                        phase="completion",
                        frame_data=frame_data,
                        stage_desc=completion_desc,
                        reason_code=v.reason_code,
                        reason=v.reason,
                    )
                    completion_candidate = False
                    self._pending_completion_stage_desc = None
                else:
                    self._pending_completion_stage_desc = completion_desc
                    self._pending_completion_reason = {
                        "kind": "completion",
                        "reason_code": v.reason_code,
                        "reason": v.reason,
                        "frame_id": frame_data.get("frame_id"),
                        "timestamp": frame_data.get("timestamp"),
                        "stage_name": (completion_desc or {}).get("stage_name"),
                        "stage_index": int(
                            self._stage_name_to_index.get(
                                str((completion_desc or {}).get("stage_name", "")).strip(),
                                self.current_stage_index,
                            )
                        ),
                    }
            else:
                self._pending_completion_stage_desc = None
            if completion_candidate:
                boost = 1
                if self.policy._support_hit(ctx, stage_desc, "completion"):
                    boost += 1
                self._completion_hits += boost
            else:
                self._completion_hits = 0

            stage_min_recording = int((stage_desc or {}).get("min_recording_frames", self.min_recording_frames))
            stage_min_completion_hits = int(
                (stage_desc or {}).get("min_consecutive_completion_hits", self.min_consecutive_completion_hits)
            )
            if self._recording_frames >= stage_min_recording and self._completion_hits >= stage_min_completion_hits:
                self.state = self.WAIT_COMPLETION_CONFIRM
                self._completion_confirm_count = 0
                logger.info("[FSMv2] RECORDING -> WAIT_COMPLETION_CONFIRM")
            return

        if self.state == self.WAIT_COMPLETION_CONFIRM:
            completion_desc = stage_desc
            completion_validation = None
            completion_candidate = self.policy.completion_signal(ctx, stage_desc)
            pre_v = self.contract_validator.validate_completion(
                mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                ctx=ctx,
                stage_descriptor=stage_desc,
            )
            if (not pre_v.allowed) and ("WRONG_REGION" in str(pre_v.reason_code)):
                self._log_contract_reject(
                    phase="completion_confirm",
                    frame_data=frame_data,
                    stage_desc=stage_desc,
                    reason_code=pre_v.reason_code,
                    reason=pre_v.reason,
                )
            if not completion_candidate:
                unordered_desc, unordered_v = self._match_unordered_completion(ctx, stage_desc)
                if unordered_desc is not None and unordered_v is not None:
                    completion_candidate = True
                    completion_desc = unordered_desc
                    completion_validation = unordered_v
            if completion_candidate:
                v = completion_validation or self.contract_validator.validate_completion(
                    mode=self.policy.__class__.__name__.replace("Policy", "").lower(),
                    ctx=ctx,
                    stage_descriptor=completion_desc,
                )
                if not v.allowed:
                    self._log_contract_reject(
                        phase="completion_confirm",
                        frame_data=frame_data,
                        stage_desc=completion_desc,
                        reason_code=v.reason_code,
                        reason=v.reason,
                    )
                    completion_candidate = False
                    self._pending_completion_stage_desc = None
                else:
                    self._pending_completion_stage_desc = completion_desc
                    self._pending_completion_reason = {
                        "kind": "completion",
                        "reason_code": v.reason_code,
                        "reason": v.reason,
                        "frame_id": frame_data.get("frame_id"),
                        "timestamp": frame_data.get("timestamp"),
                        "stage_name": (completion_desc or {}).get("stage_name"),
                        "stage_index": int(
                            self._stage_name_to_index.get(
                                str((completion_desc or {}).get("stage_name", "")).strip(),
                                self.current_stage_index,
                            )
                        ),
                    }
            if completion_candidate:
                self._completion_confirm_count += 1
            else:
                # transient interruption -> back to recording
                self.state = self.RECORDING
                self._completion_confirm_count = 0
                self._pending_completion_stage_desc = None
                return

            stage_confirm_frames = int((stage_desc or {}).get("completion_confirm_frames", self.completion_confirm_frames))
            if self._completion_confirm_count >= stage_confirm_frames:
                self.state = self.COOLDOWN
                stage_cooldown = int((stage_desc or {}).get("cooldown_frames", self.cooldown_frames))
                self._cooldown_left = int(max(0, stage_cooldown))
                self._segment_closed = True
                completion_stage_desc = self._pending_completion_stage_desc or stage_desc
                self._transition_reasons.append(
                    self._pending_completion_reason
                    if self._pending_completion_reason is not None
                    else {
                        "kind": "completion",
                        "reason_code": "OK_COMPLETION",
                        "reason": "Completion condition satisfied.",
                        "frame_id": frame_data.get("frame_id"),
                        "timestamp": frame_data.get("timestamp"),
                        "stage_name": (completion_stage_desc or {}).get("stage_name"),
                        "stage_index": int(self.current_stage_index),
                    }
                )
                if self.stage_descriptors:
                    completed_stage_index = int(self.current_stage_index)
                    next_index = min(self.current_stage_index + 1, len(self.stage_descriptors) - 1)
                    group_id = str((completion_stage_desc or {}).get("unordered_group_id", "")).strip()
                    if group_id and group_id in self._unordered_groups:
                        group = self._unordered_groups[group_id]
                        completed = self._unordered_completed.setdefault(group_id, set())
                        stage_name = str((completion_stage_desc or {}).get("stage_name", "")).strip()
                        if stage_name:
                            completed.add(stage_name)
                        remaining = [m for m in group.get("members", []) if m not in completed]
                        if remaining:
                            next_name = remaining[0]
                            next_index = self._stage_name_to_index.get(next_name, next_index)
                        else:
                            after_index = group.get("after_index")
                            if after_index is not None:
                                next_index = min(int(after_index), len(self.stage_descriptors) - 1)
                    self.current_stage_index = next_index
                    completion_mode = str((completion_stage_desc or {}).get("completion_mode", "next_stage_pick")).lower()
                    if (
                        isinstance(self.policy, AssemblyPolicy)
                        and completion_mode == "next_stage_pick"
                        and next_index != completed_stage_index
                        and self.start_handoff_frames > 0
                    ):
                        self._handoff_start_ctx = ctx
                        self._handoff_start_frames_left = int(self.start_handoff_frames)
                        self._handoff_force_start = False
                    elif (
                        isinstance(self.policy, DismantlePolicy)
                        and next_index != completed_stage_index
                        and self.start_handoff_frames > 0
                        and self._stage_uses_implicit_dismantle_start(self.stage_descriptors[next_index])
                    ):
                        self._handoff_start_ctx = ctx
                        self._handoff_start_frames_left = int(max(1, self.start_handoff_frames))
                        self._handoff_force_start = True
                    else:
                        self._clear_handoff_start()
                logger.info("[FSMv2] WAIT_COMPLETION_CONFIRM -> COOLDOWN")
