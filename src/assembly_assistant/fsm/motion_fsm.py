from assembly_assistant.utils.logger import get_logger


logger = get_logger(__name__)


class MotionFSM:
 
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    DONE = "DONE"
 
    def __init__(
        self,
        work_region=None,
        drop_region=None,
        operation_mode="dismantle",
        done_region_roles=None,
        use_active_hand_for_done_region=True,
        trigger_point_mode="wrist",
        warmup_frames=20,
        min_consecutive_visible=5,
        start_no_activity_frames=10,
        min_consecutive_drop_hits=3,
        min_recording_frames=15,
        max_hand_jump=0.25,
        activity_v_min=0.003,
        pause_tolerance_frames=20,
        occlusion_tolerance_frames=15,
        done_cooldown_frames=8,
        drop_window_frames=10,
        rearm_clear_frames=8,
    ):
        self.state = self.IDLE
        allowed_modes = {"wrist", "wrist_index_midpoint"}
        if trigger_point_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported trigger_point_mode={trigger_point_mode}. "
                f"Expected one of {sorted(list(allowed_modes))}."
            )
        self.trigger_point_mode = trigger_point_mode
        self.operation_mode = str(operation_mode).strip().lower()
        if self.operation_mode not in {"assembly", "dismantle"}:
            raise ValueError("operation_mode must be one of: assembly, dismantle")
        if done_region_roles is None:
            if self.operation_mode == "assembly":
                done_region_roles = ["part_pick"]
            else:
                done_region_roles = ["part_drop", "tool_drop"]
        self.done_region_roles = [str(x) for x in done_region_roles]
        self.use_active_hand_for_done_region = bool(use_active_hand_for_done_region)
        self.warmup_frames = int(max(0, warmup_frames))
        self.min_consecutive_visible = int(max(1, min_consecutive_visible))
        self.start_no_activity_frames = int(max(1, start_no_activity_frames))
        self.min_consecutive_drop_hits = int(max(1, min_consecutive_drop_hits))
        self.min_recording_frames = int(max(0, min_recording_frames))
        self.max_hand_jump = float(max_hand_jump)
        self.activity_v_min = float(max(0.0, activity_v_min))
        self.pause_tolerance_frames = int(max(0, pause_tolerance_frames))
        self.occlusion_tolerance_frames = int(max(0, occlusion_tolerance_frames))
        self.done_cooldown_frames = int(max(0, done_cooldown_frames))
        self.drop_window_frames = int(max(1, drop_window_frames))
        self.rearm_clear_frames = int(max(0, rearm_clear_frames))
 
        self.WORK_REGION = self._parse_region(
            work_region or (0.2, 0.2, 0.6, 0.6)
        )
 
        self.DROP_REGION = self._parse_region(
            drop_region or (0.7, 0.7, 0.95, 0.95)
        )

        self._frames_seen = 0
        self._consecutive_work_hits = 0
        self._consecutive_work_visible_hits = 0
        self._consecutive_drop_hits = 0
        self._recording_frames = 0
        self._pause_count = 0
        self._occlusion_count = 0
        self._done_cooldown_left = 0
        self._drop_hit_window = []
        self._drop_exit_window = []
        self._require_drop_exit_clear = False
        self._last_wrist = None
 
    def _parse_region(self, region):
        """
        Accepts either:
        - tuple: (x_min, y_min, x_max, y_max)
        - dict : {"x_min":..., "y_min":..., ...}
        """
        if isinstance(region, tuple):
            return {
                "x_min": region[0],
                "y_min": region[1],
                "x_max": region[2],
                "y_max": region[3],
            }
        return region
 
    def reset(self):
        self.state = self.IDLE
        # Keep _frames_seen so warmup applies only at startup, not every stage retry.
        self._consecutive_work_hits = 0
        self._consecutive_work_visible_hits = 0
        self._consecutive_drop_hits = 0
        self._recording_frames = 0
        self._pause_count = 0
        self._occlusion_count = 0
        self._done_cooldown_left = 0
        self._drop_hit_window = []
        self._drop_exit_window = []
        self._require_drop_exit_clear = False
        self._last_wrist = None

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

    def _aggregate_intent_metrics(self, frame_data):
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

        # Backward-compatible fallback for callers that don't populate velocity.
        if visible and not velocity_observed:
            activity = max(activity, self.activity_v_min + 1e-6)

        # Optional role-based DONE trigger from region registry.
        region_hits = frame_data.get("region_hits") or {}
        if self.use_active_hand_for_done_region:
            role_scope = (region_hits.get("active") or {}).get("roles", [])
        else:
            role_scope = (region_hits.get("any") or {}).get("roles", [])
        role_set = set(str(r) for r in role_scope)
        if any(role in role_set for role in self.done_region_roles):
            in_drop = True

        return in_work, in_drop, activity, visible

    def _active_hand_in_drop(self, frame_data):
        hand = frame_data.get("hand") or {}
        if not hand.get("visible", False):
            return False
        x, y = self._resolve_trigger_point(hand)
        return self._in_region(x, y, self.DROP_REGION)
 
    def update(self, frame_data):
        """
        Update FSM state based on current frame_data
        """
        self._frames_seen += 1

        # Ignore startup blur/noise frames.
        if self._frames_seen <= self.warmup_frames:
            return

        if self.state == self.DONE:
            if self._done_cooldown_left > 0:
                self._done_cooldown_left -= 1
            if self._done_cooldown_left <= 0:
                self.state = self.IDLE
                # Require a short drop-exit period before allowing a new start.
                self._require_drop_exit_clear = self.rearm_clear_frames > 0
                self._drop_exit_window = []
            return

        in_work, in_drop, activity, visible = self._aggregate_intent_metrics(frame_data)

        hand_block = frame_data.get("hand") or {}
        jump_spike = False
        if hand_block.get("visible", False):
            x, y = self._resolve_trigger_point(hand_block)
            if self._last_wrist is not None and self.max_hand_jump > 0 and self.state == self.IDLE:
                dx = x - self._last_wrist[0]
                dy = y - self._last_wrist[1]
                if (dx * dx + dy * dy) ** 0.5 > self.max_hand_jump:
                    jump_spike = True
            if not jump_spike:
                self._last_wrist = (x, y)
        else:
            self._last_wrist = None

        if self.state == self.IDLE:
            if self._require_drop_exit_clear:
                # Windowed drop-exit gating avoids runaway delays from boundary jitter.
                active_in_drop = self._active_hand_in_drop(frame_data)
                self._drop_exit_window.append(0 if active_in_drop else 1)
                if len(self._drop_exit_window) > self.rearm_clear_frames:
                    self._drop_exit_window.pop(0)
                required_clear = max(1, (3 * self.rearm_clear_frames + 4) // 5)  # ceil(0.6 * N)
                if sum(self._drop_exit_window) < required_clear:
                    self._consecutive_work_hits = 0
                    self._consecutive_work_visible_hits = 0
                    return
                self._require_drop_exit_clear = False
                self._drop_exit_window = []

            if in_work and visible and not jump_spike:
                self._consecutive_work_visible_hits += 1
            else:
                self._consecutive_work_visible_hits = 0

            if in_work and activity >= self.activity_v_min:
                self._consecutive_work_hits += 1
            else:
                self._consecutive_work_hits = 0

            if (
                self._consecutive_work_hits >= self.min_consecutive_visible
                or self._consecutive_work_visible_hits >= self.start_no_activity_frames
            ):
                    self.state = self.RECORDING
                    self._recording_frames = 0
                    self._pause_count = 0
                    self._occlusion_count = 0
                    self._drop_hit_window = []
                    logger.info("[FSM] IDLE -> RECORDING")
 
        elif self.state == self.RECORDING:
            self._recording_frames += 1

            if visible:
                self._occlusion_count = 0
                if activity < self.activity_v_min:
                    self._pause_count += 1
                else:
                    self._pause_count = 0
            else:
                self._occlusion_count += 1
                self._pause_count = 0
            if self._occlusion_count > self.occlusion_tolerance_frames:
                self.state = self.IDLE
                self._consecutive_work_hits = 0
                self._recording_frames = 0
                self._drop_hit_window = []
                self._consecutive_drop_hits = 0
                return

            self._drop_hit_window.append(1 if in_drop else 0)
            if len(self._drop_hit_window) > self.drop_window_frames:
                self._drop_hit_window.pop(0)
            self._consecutive_drop_hits = int(sum(self._drop_hit_window))

            if (
                self._recording_frames >= self.min_recording_frames and
                self._consecutive_drop_hits >= self.min_consecutive_drop_hits
            ):
                self.state = self.DONE
                self._done_cooldown_left = self.done_cooldown_frames
                logger.info("[FSM] RECORDING -> DONE")
 
    def _in_region(self, x, y, region):
        return (
            region["x_min"] <= x <= region["x_max"] and
            region["y_min"] <= y <= region["y_max"]
        )

    def is_in_work_region(self, hand_block):
        if not hand_block or not hand_block.get("visible", False):
            return False
        x, y = self._resolve_trigger_point(hand_block)
        return self._in_region(x, y, self.WORK_REGION)

    def is_in_drop_region(self, hand_block):
        if not hand_block or not hand_block.get("visible", False):
            return False
        x, y = self._resolve_trigger_point(hand_block)
        return self._in_region(x, y, self.DROP_REGION)

    def resolve_trigger_point(self, hand_block):
        return self._resolve_trigger_point(hand_block)

    def _resolve_trigger_point(self, hand_block):
        if self.trigger_point_mode == "wrist":
            return hand_block["wrist"]

        keypoints = hand_block.get("keypoints")
        if keypoints is not None and len(keypoints) > 5:
            wrist = keypoints[0]
            index_mcp = keypoints[5]
            return (
                float((wrist[0] + index_mcp[0]) * 0.5),
                float((wrist[1] + index_mcp[1]) * 0.5),
            )

        # Fallback to wrist if keypoints are unavailable/incomplete.
        return hand_block["wrist"]

 
