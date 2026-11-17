import numpy as np

from assembly_assistant.pipeline.video_extractors import _resolve_offline_hand_label
from configs.experiment_configuration import HANDEDNESS_FALLBACK_IMAGE_MIRRORED


def _kp_with_wrist_x(x):
    kp = np.zeros((21, 2), dtype=float)
    kp[0, 0] = x
    return kp


def test_offline_label_fallback_uses_temporal_continuity():
    prev = {0: "LEFT"}
    label = _resolve_offline_hand_label(
        raw_label=None,
        hand_index=0,
        kp_norm=_kp_with_wrist_x(0.9),
        prev_labels_by_index=prev,
        left_assigned=False,
        right_assigned=False,
    )
    assert label == "LEFT"


def test_offline_label_fallback_uses_geometry_without_right_bias():
    prev = {}
    label_left = _resolve_offline_hand_label(
        raw_label=None,
        hand_index=0,
        kp_norm=_kp_with_wrist_x(0.1),
        prev_labels_by_index=prev,
        left_assigned=False,
        right_assigned=False,
    )
    label_right = _resolve_offline_hand_label(
        raw_label=None,
        hand_index=1,
        kp_norm=_kp_with_wrist_x(0.9),
        prev_labels_by_index=prev,
        left_assigned=True,
        right_assigned=False,
    )
    if HANDEDNESS_FALLBACK_IMAGE_MIRRORED:
        assert label_left == "RIGHT"
    else:
        assert label_left == "LEFT"
    # With left already assigned for this frame, fallback should avoid duplicate side.
    assert label_right == "RIGHT"
