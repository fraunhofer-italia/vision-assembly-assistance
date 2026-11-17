import numpy as np

from assembly_assistant.pipeline.roi_motion_trigger import ROIMotionTrigger


def _frame_with_changes(changed_pixels: set[tuple[int, int]], shape: tuple[int, int] = (20, 20)) -> np.ndarray:
    frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for y, x in changed_pixels:
        frame[y, x] = 255
    return frame


def _roi_and_outside_points():
    roi_points = {(y, x) for y in range(10) for x in range(10)}
    outside_points = [(y, x) for y in range(20) for x in range(20) if (y, x) not in roi_points]
    return roi_points, outside_points


def test_roi_motion_trigger_strong_roi_override_survives_global_suppression():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:10, :10] = 1
    trigger = ROIMotionTrigger(
        roi_masks={"pick_box": mask},
        baseline_frames=1,
        diff_threshold=1,
        score_threshold=0.02,
        dwell_frames=2,
        history_frames=2,
        blur_kernel=(1, 1),
        global_suppress_threshold=0.02,
        global_relative_factor=2.0,
        strong_score_threshold=0.20,
        strong_relative_factor=1.5,
    )
    trigger.update(np.zeros((20, 20, 3), dtype=np.uint8))

    roi_points, outside_points = _roi_and_outside_points()
    changed = set(list(sorted(roi_points))[:55] + outside_points[:67])
    trigger.update(_frame_with_changes(changed))
    result = trigger.update(_frame_with_changes(changed))
    assert result is not None
    roi = result["pick_box"]
    assert 0.30 <= trigger.last_global_motion_score <= 0.35
    assert roi.motion_score >= 0.50
    assert roi.active is True


def test_roi_motion_trigger_keeps_weak_roi_suppressed_under_high_global_motion():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:10, :10] = 1
    trigger = ROIMotionTrigger(
        roi_masks={"pick_box": mask},
        baseline_frames=1,
        diff_threshold=1,
        score_threshold=0.02,
        dwell_frames=2,
        history_frames=2,
        blur_kernel=(1, 1),
        global_suppress_threshold=0.02,
        global_relative_factor=2.0,
        strong_score_threshold=0.20,
        strong_relative_factor=1.5,
    )
    trigger.update(np.zeros((20, 20, 3), dtype=np.uint8))

    roi_points, outside_points = _roi_and_outside_points()
    changed = set(list(sorted(roi_points))[:5] + outside_points[:120])
    trigger.update(_frame_with_changes(changed))
    result = trigger.update(_frame_with_changes(changed))
    assert result is not None
    roi = result["pick_box"]
    assert trigger.last_global_motion_score >= 0.30
    assert 0.04 <= roi.motion_score <= 0.08
    assert roi.active is False


def test_roi_motion_trigger_allows_clear_local_roi_under_busy_global_motion():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:10, :10] = 1
    trigger = ROIMotionTrigger(
        roi_masks={"pick_box": mask},
        baseline_frames=1,
        diff_threshold=1,
        score_threshold=0.02,
        dwell_frames=2,
        history_frames=2,
        blur_kernel=(1, 1),
        global_suppress_threshold=0.02,
        global_relative_factor=2.0,
        strong_score_threshold=0.20,
        strong_relative_factor=1.5,
    )
    trigger.update(np.zeros((20, 20, 3), dtype=np.uint8))

    roi_points, outside_points = _roi_and_outside_points()
    changed = set(list(sorted(roi_points))[:32] + outside_points[:60])
    trigger.update(_frame_with_changes(changed))
    result = trigger.update(_frame_with_changes(changed))
    assert result is not None
    roi = result["pick_box"]
    assert 0.30 <= roi.motion_score <= 0.35
    assert 0.20 <= trigger.last_global_motion_score <= 0.26
    assert roi.active is True
