import numpy as np

from assembly_assistant.pipeline import video_extractors as ve


class _FakeSync:
    def sync(self, *, hand_data, tool_data, timestamp, frame_id, **context):
        visible = bool(hand_data.get("visible", False))
        wrist = hand_data.get("wrist")
        return {
            "hand": {"visible": visible, "wrist": wrist, "velocity": 0.0},
            "hands": {
                "left": {"visible": visible, "wrist": wrist, "velocity": 0.0},
                "right": {"visible": False, "wrist": None, "velocity": 0.0},
            },
            "timestamp": timestamp,
            "frame_id": frame_id,
        }


class _FakeSelector:
    def select(self, frame_data):
        return None


class _FakeBuilder:
    def __init__(self):
        self._traj = []

    def build_trajectory_point(self, frame_data):
        if not bool((frame_data.get("hand") or {}).get("visible", False)):
            return None
        return {"x": 0.25, "y": 0.25, "t": frame_data.get("timestamp", 0.0)}

    def append(self, point):
        self._traj.append(point)

    def get_trajectory(self):
        return list(self._traj)


class _FakeMotion:
    WAIT_TRIGGER = "WAIT_TRIGGER"
    RECORDING = "RECORDING"

    def __init__(self):
        self.state = self.WAIT_TRIGGER
        self.started = False
        self._pending_trigger_reason = False
        self.anchor_seen = False
        self.abort_calls = 0

    def update(self, frame_data):
        if not self.started:
            self.state = self.RECORDING
            self.started = True
            self._pending_trigger_reason = True
        elif bool(frame_data.get("anchor_valid", False)):
            self.anchor_seen = True

    def consume_contract_rejections(self):
        return []

    def consume_transition_reasons(self):
        if self._pending_trigger_reason:
            self._pending_trigger_reason = False
            return [{"kind": "trigger", "stage_name": "A01"}]
        return []

    def consume_segment_closed(self):
        return False

    def is_waiting_for_first_anchor(self):
        return self.state == self.RECORDING and not self.anchor_seen

    def abort_recording_keep_stage(self):
        self.abort_calls += 1
        self.state = self.WAIT_TRIGGER


def _packets(visible_flags):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for idx, visible in enumerate(visible_flags):
        yield {
            "frame": frame.copy(),
            "frame_id": idx,
            "timestamp": float(idx) * 0.05,
            "hand_data": {"visible": bool(visible), "wrist": (0.25, 0.25) if visible else None},
            "keypoints_px": None,
            "keypoints_px_by_label": {},
        }


def test_no_hand_split_does_not_fire_before_first_anchor(monkeypatch):
    motion = _FakeMotion()
    monkeypatch.setattr(ve, "_make_runtime", lambda *args, **kwargs: (_FakeSync(), _FakeSelector(), motion, _FakeBuilder()))
    monkeypatch.setattr(ve, "_new_builder", lambda *args, **kwargs: _FakeBuilder())
    monkeypatch.setattr(ve, "_render_frame", lambda *args, **kwargs: False)

    segments = ve._extract_segments_from_packets(
        packets=_packets([False, False, False, False]),
        pre=None,
        anchor_mode="wrist",
        dominant_hand="RIGHT",
        active_hand_policy="best_match",
        show=False,
        delay_ms=1,
        no_hand_frames_to_split=2,
        rearm_no_hand_frames=1,
        min_segment_points=1,
        work_region=None,
        drop_region=None,
        operation_mode="assembly",
        region_registry=None,
        region_detector=None,
    )

    assert segments == []
    assert motion.abort_calls == 0


def test_no_hand_split_after_anchor_uses_abort_without_full_reset(monkeypatch):
    motion = _FakeMotion()
    monkeypatch.setattr(ve, "_make_runtime", lambda *args, **kwargs: (_FakeSync(), _FakeSelector(), motion, _FakeBuilder()))
    monkeypatch.setattr(ve, "_new_builder", lambda *args, **kwargs: _FakeBuilder())
    monkeypatch.setattr(ve, "_render_frame", lambda *args, **kwargs: False)

    segments = ve._extract_segments_from_packets(
        packets=_packets([True, True, False, False, False]),
        pre=None,
        anchor_mode="wrist",
        dominant_hand="RIGHT",
        active_hand_policy="best_match",
        show=False,
        delay_ms=1,
        no_hand_frames_to_split=2,
        rearm_no_hand_frames=1,
        min_segment_points=1,
        work_region=None,
        drop_region=None,
        operation_mode="assembly",
        region_registry=None,
        region_detector=None,
    )

    assert len(segments) == 1
    assert segments[0]["closure_type"] == "no_hand_split"
    assert motion.abort_calls == 1
