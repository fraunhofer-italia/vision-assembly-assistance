"""
Real-time guidance UI for pneumatic piston assembly/dismantle.

This script keeps the reference side video-based, but replaces the observed
video with a live camera feed. MotionFSM handles live stage segmentation and
StageFSM evaluates the closed segment against the recorded references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CALIB_DATA_ROOT = SRC / "assembly_assistant" / "calibration" / "calibration_data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from configs.experiment_configuration import (  # noqa: E402
    ACTIVE_HAND_POLICY_DEFAULT,
    ANCHOR_MODE_DEFAULT,
    COLOR_TRIGGER_ENABLED,
    COLOR_BASELINE_FRAMES,
    COLOR_DELTA_THRESHOLD,
    COLOR_DELTA_RANGE,
    DELAY_MS,
    DOMINANT_HAND,
    DROP_REGION,
    FUSED_TRIGGER_THRESHOLD,
    FUSED_WEIGHT_COLOR,
    FUSED_WEIGHT_DEPTH,
    FUSED_WEIGHT_RGB,
    MOTION_BASELINE_FRAMES,
    MOTION_BLUR_KERNEL,
    MOTION_DIFF_THRESHOLD,
    MOTION_DWELL_FRAMES,
    MOTION_FOCUS_IGNORE_FRAMES,
    MOTION_FSM_VERSION,
    MOTION_GLOBAL_RELATIVE_FACTOR,
    MOTION_GLOBAL_SUPPRESS_THRESHOLD,
    MOTION_HISTORY_FRAMES,
    MOTION_LOG_INTERVAL_FRAMES,
    MOTION_SCORE_THRESHOLD,
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    NO_HAND_FRAMES_TO_SPLIT,
    PROTOCOL_VERSION,
    REARM_NO_HAND_FRAMES,
    STAGE_SCORE_THRESHOLD,
    STAGE_THRESHOLDS_BY_STAGE,
    STAGE_WEIGHTS,
    WORK_REGION,
)
from assembly_assistant.fsm.stage_fsm import StageFSM  # noqa: E402
from assembly_assistant.pipeline.color_change_trigger import ColorChangeTrigger  # noqa: E402
from assembly_assistant.pipeline.config_resolver import csv_list, stage_ref_list  # noqa: E402
from assembly_assistant.pipeline.event_extractor import RegionEventExtractor  # noqa: E402
from assembly_assistant.pipeline.protocol import build_protocol_metadata  # noqa: E402
from assembly_assistant.pipeline.overlays import draw_anchor_overlay, draw_detection_overlay  # noqa: E402
from assembly_assistant.pipeline.reference_build import build_stage_references_from_videos  # noqa: E402
from assembly_assistant.pipeline.region_detector import RegionDetector  # noqa: E402
from assembly_assistant.pipeline.region_registry import RegionRegistry  # noqa: E402
from assembly_assistant.pipeline.roi_motion_trigger import ROIMotionTrigger  # noqa: E402
from assembly_assistant.pipeline.stage_descriptor_loader import load_stage_descriptors  # noqa: E402
from assembly_assistant.pipeline.virtual_projection import VirtualProjectionGuide  # noqa: E402
from assembly_assistant.pose.pose_classifier import PoseClassifierManager, parse_pose_classifier_model_map  # noqa: E402
from assembly_assistant.remote.vision_inference import RemoteVisionInferenceClient  # noqa: E402
from assembly_assistant.trajectory.frame_processor import FramePreprocessor  # noqa: E402
from assembly_assistant.utils.logger import get_logger  # noqa: E402
from assembly_assistant.utils.metadata_manager import MetadataManager  # noqa: E402
from assembly_assistant.utils.metadata_validator import validate_metadata_file  # noqa: E402
from scripts.main_full_pipeline import (  # noqa: E402
    _build_descriptor_unordered_groups,
    _build_stage_descriptors,
    _descriptors_have_expected_poses,
    _merge_stage_descriptor_expectations,
    _parse_kernel,
    _postprocess_unordered_segment_progress,
)
from assembly_assistant.pipeline.video_extractors import (  # noqa: E402
    _annotate_region_hits,
    _apply_depth_triggers,
    _attach_pose_classifier_predictions,
    _build_color_roi_masks,
    _build_fused_hits,
    _build_motion_region_hits,
    _get_mediapipe_hands,
    _make_object_detector,
    _make_runtime,
    _make_workspace_filter,
    _new_builder,
    _none_tool,
    _resolve_current_stage_descriptor,
    _resolve_event_hand_scope,
    _resolve_motion_allowed_regions,
    _resolve_offline_hand_label,
    ROIInteractionGate,
)
from assembly_assistant.ui.user_messages import (  # noqa: E402
    action_region_lines,
    expected_pose_lines,
    object_name,
    pose_message,
    region_message,
    stage_instruction,
    stage_title,
    wrap_text,
)


logger = get_logger(__name__)


STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_PASSED = "PASSED"
STATUS_FAILED = "FAILED"
GUIDANCE_FULL = "full_guidance_mode"
GUIDANCE_MID = "mid_guidance_mode"
GUIDANCE_NONE = "no_guidance_mode"


class OakRgbCapture:
    """Small cv2.VideoCapture-like wrapper around the project OAK-D RGB stream."""

    @staticmethod
    def _open_device_with_retries(dai, pipeline, retries=5, delay_sec=1.0):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return dai.Device(pipeline)
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    "Failed to open OAK-D stream (attempt %d/%d): %s",
                    attempt,
                    retries,
                    exc,
                )
                try:
                    return dai.Device(pipeline, usb2Mode=True)
                except RuntimeError:
                    pass
                time.sleep(delay_sec)
        raise RuntimeError(f"Unable to open OAK-D after {retries} retries: {last_error}")

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 20.0,
        warmup_frames: int = 10,
        output: str = "isp",
        color_correction: str = "none",
        queue_size: int = 1,
    ):
        try:
            import depthai as dai
        except ImportError as exc:
            raise RuntimeError(
                "depthai is required for live OAK-D guidance. Install depthai or use --input-video for debug playback."
            ) from exc

        self._dai = dai
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.output = str(output or "isp").strip().lower()
        self.color_correction = str(color_correction or "none").strip().lower()
        self._start = time.monotonic()
        self._device = None
        self._queue = None
        self.last_dropped_frames = 0

        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(float(fps))
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        if self.output == "preview":
            cam_rgb.setPreviewSize(self.width, self.height)
            cam_rgb.preview.link(xout_rgb.input)
        elif self.output == "video":
            cam_rgb.setVideoSize(self.width, self.height)
            cam_rgb.video.link(xout_rgb.input)
        else:
            # 1080p ISP scaled by 2/3 gives the same 1280x720 view used by
            # calibration recordings, but avoids live NV12/video-path tinting.
            if self.width == 1280 and self.height == 720:
                cam_rgb.setIspScale(2, 3)
            else:
                logger.warning(
                    "OAK ISP output is calibrated for 1280x720. Requested %dx%d may not match calibration.",
                    self.width,
                    self.height,
                )
            cam_rgb.isp.link(xout_rgb.input)

        self._device = self._open_device_with_retries(dai, pipeline)
        # Real-time guidance should prefer the newest frame over replaying a
        # backlog. A non-blocking size-1 queue prevents seconds of visual lag
        # when downstream processing is temporarily slower than camera FPS.
        self._queue = self._device.getOutputQueue(
            name="rgb",
            maxSize=max(1, int(queue_size)),
            blocking=False,
        )
        for _ in range(int(max(0, warmup_frames))):
            self._queue.get()
        self._start = time.monotonic()

    def isOpened(self) -> bool:
        return self._device is not None and self._queue is not None

    def read(self):
        if self._queue is None:
            return False, None
        try:
            msg = self._queue.tryGet()
            if msg is None:
                msg = self._queue.get()
            dropped = 0
            while True:
                latest = self._queue.tryGet()
                if latest is None:
                    break
                msg = latest
                dropped += 1
            self.last_dropped_frames = dropped
            frame = msg.getCvFrame()
        except RuntimeError as exc:
            logger.warning("OAK-D RGB stream read failed: %s", exc)
            return False, None
        frame = self._correct_color(frame)
        return frame is not None, frame

    def _correct_color(self, frame):
        if frame is None:
            return None
        mode = self.color_correction
        if mode in {"rgb_to_bgr", "swap_rb"}:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if mode == "bgr_to_rgb":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return float(time.monotonic() - self._start) * 1000.0
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self) -> None:
        device = self._device
        self._queue = None
        self._device = None
        if device is None:
            return
        close = getattr(device, "close", None)
        if callable(close):
            close()


def _stage_demo_clip_path(stage_name: str, root: str | Path = "data/gifs") -> Path | None:
    token = str(stage_name or "").split("_", 1)[0].strip()
    if not token:
        return None
    base = Path(root)
    candidates = [
        base / f"ref_Stage{token}_01_rgb.mp4",
        base / f"ref_Stage{token}_01_rgb.gif",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(base.glob(f"*Stage{token}*"))
    return matches[0] if matches else None


class ReferenceClipLooper:
    def __init__(self, *, root: str | Path = "data/gifs", speed: int = 3):
        self.root = Path(root)
        self.speed = int(max(1, speed))
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._last_frames: dict[str, np.ndarray] = {}

    def read(self, stage_name: str) -> np.ndarray | None:
        path = _stage_demo_clip_path(stage_name, self.root)
        if path is None:
            return None
        key = str(path)
        cap = self._caps.get(key)
        if cap is None:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                return None
            self._caps[key] = cap
        frame = None
        for _ in range(self.speed):
            ok, candidate = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, candidate = cap.read()
            if ok:
                frame = candidate
        if frame is not None:
            self._last_frames[key] = frame
        return frame if frame is not None else self._last_frames.get(key)

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps = {}
        self._last_frames = {}


class AsyncLatestRunner:
    """Runs slow real-time inference in the background and reuses latest result."""

    def __init__(self, name: str, *, min_interval_sec: float = 0.20, enabled: bool = True):
        self.name = str(name)
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.enabled = bool(enabled)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.name}_rt")
        self._future: Future | None = None
        self._latest = None
        self._last_submit_t = float("-inf")
        self._last_error_log_t = float("-inf")

    def poll(self):
        if self._future is not None and self._future.done():
            try:
                self._latest = self._future.result()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_error_log_t > 2.0:
                    logger.warning("%s async inference failed: %s", self.name, exc)
                    self._last_error_log_t = now
            finally:
                self._future = None
        return self._latest

    def maybe_submit(self, fn, *, now: float, force: bool = False):
        latest = self.poll()
        if not self.enabled:
            try:
                self._latest = fn()
            except Exception as exc:
                logger.warning("%s synchronous inference failed: %s", self.name, exc)
            return self._latest
        if self._future is not None:
            return latest
        if not force and (now - self._last_submit_t) < self.min_interval_sec:
            return latest
        self._future = self._executor.submit(fn)
        self._last_submit_t = now
        return latest

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _latest_calibration_dir(calib_root: Path) -> Path:
    candidates = [p for p in calib_root.iterdir() if p.is_dir() and p.name.startswith("calibration_")]
    if not candidates:
        raise FileNotFoundError(f"No calibration folders found in {calib_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_references(stage_names: list[str], spec: str) -> tuple[dict[str, list[Path]], list[Path]]:
    ref_specs = csv_list(spec)
    if len(ref_specs) != len(stage_names):
        raise ValueError("--stage-names and --reference-videos must have the same length.")
    by_stage: dict[str, list[Path]] = {}
    all_refs: list[Path] = []
    for stage_name, ref_spec in zip(stage_names, ref_specs):
        refs = stage_ref_list(ref_spec)
        if not refs:
            raise ValueError(f"No references provided for stage '{stage_name}'.")
        missing = [str(p) for p in refs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing reference video(s): {missing}")
        by_stage[stage_name] = refs
        all_refs.extend(refs)
    return by_stage, all_refs


def _path_signature(path: str | Path | None) -> dict:
    if not path:
        return {"path": None, "exists": False}
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return {"path": str(p), "exists": False}
    return {"path": str(p), "exists": True, "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _dir_signature(path: str | Path | None) -> dict:
    if not path:
        return {"path": None, "exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    items = []
    for child in sorted(x for x in p.rglob("*") if x.is_file()):
        try:
            st = child.stat()
        except OSError:
            continue
        items.append((str(child.relative_to(p)), int(st.st_mtime_ns), int(st.st_size)))
    return {"path": str(p), "exists": True, "files": items}


def _reference_cache_key(
    *,
    args,
    stage_names: list[str],
    stage_ref_videos: dict[str, list[Path]],
    calib_path: Path,
    effective_color_trigger_enabled: bool,
    effective_motion_trigger_enabled: bool,
) -> str:
    payload = {
        "version": 2,
        "stage_names": list(stage_names),
        "stage_ref_videos": {
            str(k): [_path_signature(p) for p in v]
            for k, v in sorted((stage_ref_videos or {}).items())
        },
        "calibration": _dir_signature(calib_path),
        "regions_config": _path_signature(args.regions_config),
        "stage_descriptors_config": _path_signature(args.stage_descriptors_config),
        "operation_mode": args.operation_mode,
        "anchor_mode": args.anchor_mode,
        "dominant_hand": args.dominant_hand,
        "active_hand_policy": args.active_hand_policy,
        "motion_fsm_version": args.motion_fsm_version,
        "use_region_detector": bool(args.use_region_detector),
        "color_trigger_enabled": bool(effective_color_trigger_enabled),
        "color_baseline_frames": int(args.color_baseline_frames),
        "color_delta_threshold": float(args.color_delta_threshold),
        "color_delta_range": float(args.color_delta_range),
        "motion_trigger_enabled": bool(effective_motion_trigger_enabled),
        "motion_hand_confirmation": not bool(args.no_motion_hand_confirmation),
        "motion_hand_fallback_score_threshold": float(args.motion_hand_fallback_score_threshold),
        "motion_baseline_frames": int(args.motion_baseline_frames),
        "motion_ignore_frames": int(args.motion_ignore_frames),
        "motion_diff_threshold": int(args.motion_diff_threshold),
        "motion_score_threshold": float(args.motion_score_threshold),
        "motion_dwell_frames": int(args.motion_dwell_frames),
        "motion_history_frames": int(args.motion_history_frames),
        "motion_blur_kernel": tuple(args.motion_blur_kernel),
        "motion_global_suppress_threshold": float(args.motion_global_suppress_threshold),
        "motion_global_relative_factor": float(args.motion_global_relative_factor),
        "fused_trigger_threshold": float(args.fused_trigger_threshold),
        "fused_weight_color": float(args.fused_weight_color),
        "fused_weight_depth": float(args.fused_weight_depth),
        "fused_weight_rgb": float(args.fused_weight_rgb),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_reference_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        logger.warning("Ignoring unreadable reference cache %s: %s", cache_path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    refs = payload.get("stage_references")
    names = payload.get("stage_reference_names")
    if not isinstance(refs, dict) or not isinstance(names, dict):
        return None
    return refs, names


def _save_reference_cache(cache_path: Path, stage_references: dict, stage_reference_names: dict) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(
                {
                    "created_at": time.time(),
                    "stage_references": stage_references,
                    "stage_reference_names": stage_reference_names,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
    except Exception as exc:
        logger.warning("Failed to write reference cache %s: %s", cache_path, exc)


def _expand_csv(value: str | None, n: int, default: str) -> list[str]:
    values = csv_list(value) if value else [default]
    if len(values) == 1 and n > 1:
        values *= n
    if len(values) != n:
        raise ValueError(f"Expected list length 1 or {n}, got {len(values)}.")
    return values


def _expand_bool_csv(value: str | None, n: int, default: bool) -> list[bool]:
    values = _expand_csv(value, n, "true" if default else "false")
    out = []
    for item in values:
        v = str(item).strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            out.append(True)
        elif v in {"false", "0", "no", "n", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {item}")
    return out


def _build_stage_config(
    *,
    stage_names: list[str],
    expected_tools: list[str],
    thresholds: list[float],
    stage_types: list[str],
    required_counts: list[int],
    lockout_enabled: list[bool],
    margin_mins: list[float],
    stage_ref_videos: dict[str, list[Path]],
) -> list[dict]:
    out = []
    for name, tool, threshold, stage_type, required_count, lockout, margin_min in zip(
        stage_names,
        expected_tools,
        thresholds,
        stage_types,
        required_counts,
        lockout_enabled,
        margin_mins,
    ):
        refs = stage_ref_videos[name]
        out.append(
            {
                "name": name,
                "expected_tools": [tool],
                "score_threshold": float(threshold),
                "margin_min": float(max(0.0, margin_min)),
                "stage_type": str(stage_type).strip().lower(),
                "required_count": int(required_count),
                "lockout_enabled": bool(lockout),
                "subtask_groups": {f"S{i + 1}": [i] for i in range(len(refs))},
                "subtask_refs": [str(p) for p in refs],
            }
        )
    return out


def _descriptor_by_name(stage_descriptors: list[dict]) -> dict[str, dict]:
    return {
        str((desc or {}).get("stage_name", "")).strip(): dict(desc or {})
        for desc in (stage_descriptors or [])
        if str((desc or {}).get("stage_name", "")).strip()
    }


def _apply_descriptor_stage_overrides(stage_config: list[dict], stage_descriptors: list[dict]) -> list[dict]:
    by_name = _descriptor_by_name(stage_descriptors)
    out = []
    for stage in stage_config:
        item = dict(stage)
        desc = by_name.get(str(stage.get("name", "")).strip(), {})
        for key in (
            "score_threshold",
            "min_segment_points",
            "unordered_group_id",
            "unordered_group_members",
        ):
            if key in desc:
                item[key] = desc[key]
        out.append(item)
    return out


def _stage_min_segment_points(stage_config: list[dict], completed_stage_name: str, default: int) -> int:
    for stage in stage_config or []:
        if str(stage.get("name")) == str(completed_stage_name):
            return int(max(1, stage.get("min_segment_points", default)))
    return int(max(1, default))


def _is_unordered_descriptor(stage_desc: dict | None) -> bool:
    return bool(str((stage_desc or {}).get("unordered_group_id", "")).strip())


def _remaining_unordered_display_descs(motion, stage_desc: dict | None) -> list[dict]:
    group_id = str((stage_desc or {}).get("unordered_group_id", "")).strip()
    if not group_id:
        return [stage_desc] if stage_desc else []
    remaining_fn = getattr(motion, "_unordered_remaining_descs", None)
    if callable(remaining_fn):
        remaining = [dict(d) for d in (remaining_fn(group_id) or []) if d]
        if remaining:
            return remaining
    return [stage_desc] if stage_desc else []


def _set_motion_stage_index_preserving_unordered(
    *,
    motion,
    stage_index: int,
    descriptor_stage_info: dict,
    completed_unordered_members: set,
) -> None:
    if not hasattr(motion, "set_stage_index"):
        return
    motion.set_stage_index(int(stage_index))
    completed_by_group: dict[str, set[str]] = {}
    for name in completed_unordered_members or set():
        group_id = (descriptor_stage_info.get(str(name)) or {}).get("group_id")
        if group_id:
            completed_by_group.setdefault(str(group_id), set()).add(str(name))
    if completed_by_group and hasattr(motion, "_unordered_completed"):
        current = getattr(motion, "_unordered_completed", {}) or {}
        for group_id, members in completed_by_group.items():
            current.setdefault(group_id, set()).update(members)
        setattr(motion, "_unordered_completed", current)


def _composite_stage_descriptor(stage_descs: list[dict]) -> dict:
    descs = [dict(d or {}) for d in (stage_descs or []) if d]
    if not descs:
        return {}
    if len(descs) == 1:
        return descs[0]
    base = dict(descs[0])
    base["stage_name"] = " / ".join(str(d.get("stage_name", "")) for d in descs if d.get("stage_name"))
    for key in (
        "start_source_regions",
        "start_tool_regions",
        "completion_regions",
        "next_stage_source_regions",
        "completion_support_regions",
        "start_support_regions",
        "expected_parts",
        "expected_tools",
        "active_part_candidates",
        "active_tool_candidates",
    ):
        values = []
        for desc in descs:
            for value in desc.get(key, []) or []:
                if value not in values:
                    values.append(value)
        if values:
            base[key] = values
    return base


def _resolve_region_points(region, pre: FramePreprocessor):
    if region.coordinate_frame == "image_px":
        transform = getattr(pre, "image_px_transform", None)
        pts = []
        for x, y in region.polygon:
            if transform is not None:
                p = transform @ np.array([float(x), float(y), 1.0], dtype=float)
                if abs(float(p[2])) < 1e-9:
                    return []
                pts.append((int(round(float(p[0] / p[2]))), int(round(float(p[1] / p[2])))))
            else:
                pts.append((int(round(x)), int(round(y))))
        return pts
    try:
        h_inv = np.linalg.inv(pre.H)
    except np.linalg.LinAlgError:
        return []
    pts = []
    for x, y in region.polygon:
        p = h_inv @ np.array([float(x), float(y), 1.0], dtype=float)
        if abs(float(p[2])) < 1e-9:
            return []
        pts.append((int(round(float(p[0] / p[2]))), int(round(float(p[1] / p[2])))))
    return pts


def _flipped_display_preprocessor(pre: FramePreprocessor, frame_shape) -> object:
    h, w = frame_shape[:2]
    flip_to_original = np.array(
        [
            [-1.0, 0.0, float(w - 1)],
            [0.0, -1.0, float(h - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    class _DisplayPreprocessor:
        pass

    out = _DisplayPreprocessor()
    out.H = pre.H @ flip_to_original
    out.image_px_transform = flip_to_original
    return out


def _flip_keypoints_px(keypoints_px, frame_shape):
    if keypoints_px is None:
        return None
    h, w = frame_shape[:2]
    arr = np.array(keypoints_px, dtype=float).copy()
    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        arr[..., 0] = float(w - 1) - arr[..., 0]
        arr[..., 1] = float(h - 1) - arr[..., 1]
    return arr


def _flip_bbox_xyxy(bbox, frame_shape):
    if bbox is None or len(bbox) != 4:
        return bbox
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [float(w - 1) - x2, float(h - 1) - y2, float(w - 1) - x1, float(h - 1) - y1]


def _flip_frame_data_for_display(frame_data: dict | None, frame_shape) -> dict | None:
    if not frame_data:
        return frame_data
    out = dict(frame_data)
    for key in ("parts", "tools"):
        items = []
        for det in frame_data.get(key) or []:
            item = dict(det)
            if "bbox" in item:
                item["bbox"] = _flip_bbox_xyxy(item.get("bbox"), frame_shape)
            items.append(item)
        out[key] = items
    pose_candidate = dict(frame_data.get("pose_pilot_candidate") or {})
    if pose_candidate.get("bbox_xyxy"):
        pose_candidate["bbox_xyxy"] = _flip_bbox_xyxy(pose_candidate.get("bbox_xyxy"), frame_shape)
        out["pose_pilot_candidate"] = pose_candidate
    return out


def _stage_guidance_region_ids(stage_desc: dict, operation_mode: str) -> set[str]:
    return _stage_guidance_region_ids_for_state(stage_desc, operation_mode, motion_state=None)


def _stage_guidance_region_ids_for_state(stage_desc: dict, operation_mode: str, motion_state: str | None) -> set[str]:
    state = str(motion_state or "").upper()
    if str(operation_mode).lower() == "assembly":
        if state in {"", "WAIT_TRIGGER", "COOLDOWN"}:
            keys = ["start_source_regions", "start_tool_regions", "start_support_regions"]
        else:
            keys = [
                "completion_regions",
                "next_stage_source_regions",
                "completion_support_regions",
            ]
        ids = {str(r) for key in keys for r in (stage_desc.get(key) or []) if str(r)}
        if ids:
            return ids
    keys = [
        "start_source_regions",
        "start_tool_regions",
        "completion_regions",
        "next_stage_source_regions",
        "completion_support_regions",
        "start_support_regions",
    ]
    ids = {str(r) for key in keys for r in (stage_desc.get(key) or []) if str(r)}
    if str(operation_mode).lower() == "assembly":
        ids.update(str(r) for r in (stage_desc.get("next_stage_source_regions") or []) if str(r))
    return ids


def _hand_confirmed_motion_results(
    motion_results: dict | None,
    frame_data: dict,
    *,
    require_confirmation: bool = True,
    fallback_score_threshold: float = 0.55,
) -> dict | None:
    if not motion_results or not require_confirmation:
        return motion_results
    raw_hits = frame_data.get("region_hits") or {}
    hand_regions = set((raw_hits.get("any") or {}).get("regions", []))
    hands = frame_data.get("hands") or {}
    any_hand_visible = any(bool((hands.get(label) or {}).get("visible", False)) for label in ("left", "right"))
    for rid, res in motion_results.items():
        if not bool(getattr(res, "active", False)):
            continue
        if str(rid) in hand_regions:
            continue
        score = float(getattr(res, "mean_history", getattr(res, "motion_score", 0.0)) or 0.0)
        # If MediaPipe sees a hand but misses the exact ROI polygon, allow only
        # very strong, dwell-confirmed ROI motion. This keeps valid fast picks
        # working without letting head/shadow/global changes advance stages.
        if any_hand_visible and score >= float(fallback_score_threshold):
            continue
        else:
            res.active = False
    return motion_results


def _draw_guidance_regions(
    frame,
    *,
    pre: FramePreprocessor,
    region_registry: RegionRegistry | None,
    stage_desc: dict,
    frame_data: dict,
    operation_mode: str,
    motion_state: str | None = None,
) -> None:
    if region_registry is None:
        return
    focus = _stage_guidance_region_ids_for_state(stage_desc, operation_mode, motion_state)
    active_regions = set(((frame_data.get("fsm_region_hits") or frame_data.get("region_hits") or {}).get("any") or {}).get("regions", []))
    role_filter = {"part_pick", "part_drop", "tool_pick", "tool_drop"}
    for rid, region in region_registry.regions.items():
        roles = [r for r in region.roles if region_registry.resolve_role_enabled(rid, operation_mode, r)]
        if not any(r in role_filter for r in roles):
            continue
        if focus and rid not in focus:
            continue
        pts = _resolve_region_points(region, pre)
        if not pts:
            continue
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        # is_active = rid in active_regions
        # color = (80, 220, 80) if is_active else (150, 150, 150)
        # fill_color = (70, 190, 70) if is_active else (90, 90, 90)
        # overlay = frame.copy()
        # cv2.fillPoly(overlay, [arr], fill_color)
        # cv2.addWeighted(overlay, 0.18 if is_active else 0.10, frame, 0.82 if is_active else 0.90, 0, frame)
        # cv2.polylines(frame, [arr], True, color, 3 if is_active else 2, cv2.LINE_AA)
        is_active = rid in active_regions

        # ---------------------------------------------------------
        # Better region colour scheme
        # ---------------------------------------------------------

        label_lower = rid.lower()

        if "pick" in label_lower or "start" in label_lower:
            # Blue = Start / Pick region
            inactive_border = (255, 170, 40)
            inactive_fill   = (255, 120, 30)

        elif "drop" in label_lower or "end" in label_lower:
            # Orange = Destination / Drop region
            inactive_border = (40, 180, 255)
            inactive_fill   = (30, 130, 255)

        else:
            # Cyan = Generic region
            inactive_border = (220, 220, 70)
            inactive_fill   = (170, 170, 40)

        active_border = (60, 255, 60)
        active_fill   = (40, 220, 40)

        color = active_border if is_active else inactive_border
        fill_color = active_fill if is_active else inactive_fill

        overlay = frame.copy()

        cv2.fillPoly(overlay, [arr], fill_color)

        cv2.addWeighted(
            overlay,
            0.28 if is_active else 0.18,
            frame,
            0.72 if is_active else 0.82,
            0,
            frame,
        )

        cv2.polylines(
            frame,
            [arr],
            True,
            color,
            4 if is_active else 3,
            cv2.LINE_AA,
        )
        label = rid.replace("_", " ")
        lx, ly = pts[0]
        #cv2.putText(frame, label, (lx + 6, ly + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        # Draw black outline first
        cv2.putText(
            frame,
            label,
            (lx + 6, ly + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        # Draw coloured text
        cv2.putText(
            frame,
            label,
            (lx + 6, ly + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_text_block(img, lines: Iterable[str], x: int, y: int, *, color=(230, 230, 230), scale=0.55, line_h=24):
    yy = y
    for raw in lines:
        text = str(raw)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        yy += line_h
    return yy


def _guidance_ui_scale(ui_width: int, ui_height: int) -> float:
    scale = min(float(ui_width) / 1600.0, float(ui_height) / 900.0)
    return max(0.90, min(scale, 2.20))


def _draw_text_block_2(
    img,
    lines,
    x,
    y,
    *,
    color=(230,230,230),
    scale=0.55,
    line_h=24,
    bold=False,
):
    yy = y
    scale = float(scale)
    thickness = int(round(scale * 1.55))
    thickness = max(1, min(4, thickness + (1 if bold else 0)))
    for line in lines:
        cv2.putText(
            img,
            str(line),
            (int(x), int(yy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            color,
            thickness,
            cv2.LINE_AA,
        )
        yy += line_h

    return yy

def _recording_output_path(args, metadata) -> Path:
    if not args.record_video_output:
        raise RuntimeError("--record-video-output must be provided to record guidance video.")
    return Path(args.record_video_output)


def _open_record_writer(path: Path, frame, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps or 20.0),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _feedback_result_lines(feedback: dict | None, status: str, *, width: int = 42) -> list[str]:
    if not feedback:
        if status == STATUS_IN_PROGRESS:
            return ["In progress"]
        return [status.replace("_", " ").title()]
    if bool(feedback.get("stage_passed", False)):
        return [f"Passed. Score {float(feedback.get('final_score', 0.0)):.3f}"]
    recovery = feedback.get("recovery") or {}
    reasons = recovery.get("reason") or []
    if not reasons:
        taxonomy = feedback.get("failure_taxonomy") or {}
        primary = taxonomy.get("primary")
        if primary:
            reasons = [str(primary).replace("_", " ")]
    if not reasons:
        reasons = [f"Failed. Score {float(feedback.get('final_score', 0.0)):.3f}"]
    return wrap_text("; ".join(str(r) for r in reasons[:3]), width=width)


def _bbox_center(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _filter_detections_by_confidence(detections, min_confidence: float) -> list[dict]:
    out = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        try:
            conf = float(det.get("confidence", det.get("score", 0.0)))
        except Exception:
            conf = 0.0
        if conf >= float(min_confidence):
            out.append(dict(det))
    return out


def _point_in_region_px(point, region, pre: FramePreprocessor) -> bool:
    if point is None or region is None:
        return False
    if region.coordinate_frame == "image_px":
        pts = np.array(region.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0
    wx, wy = pre.warp_point((float(point[0]), float(point[1])))
    pts = np.array(region.polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(wx), float(wy)), False) >= 0


def _filter_detections_by_region(detections, *, region_registry, pre: FramePreprocessor, region_ids: list[str]) -> list[dict]:
    if region_registry is None or not region_ids:
        return [dict(d) for d in (detections or []) if isinstance(d, dict)]
    regions = [region_registry.regions.get(str(rid)) for rid in region_ids]
    regions = [r for r in regions if r is not None]
    if not regions:
        return []
    out = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        center = _bbox_center(det.get("bbox") or det.get("bbox_xyxy"))
        if any(_point_in_region_px(center, region, pre) for region in regions):
            out.append(dict(det))
    return out


def _attach_filtered_pose_predictions(
    *,
    frame,
    frame_data: dict,
    current_stage_desc: dict,
    frame_id: int,
    pose_classifier,
    min_confidence: float,
    region_registry,
    pre: FramePreprocessor,
    assembly_target_only: bool,
    region_ids: list[str] | None = None,
):
    original_detections = list(frame_data.get("detections") or [])
    pose_detections = _filter_detections_by_confidence(original_detections, min_confidence)
    if assembly_target_only:
        pose_detections = _filter_detections_by_region(
            pose_detections,
            region_registry=region_registry,
            pre=pre,
            region_ids=region_ids or ["assembly_target"],
        )
    frame_data["detections"] = pose_detections
    _attach_pose_classifier_predictions(
        frame=frame,
        frame_data=frame_data,
        current_stage_desc=current_stage_desc,
        frame_id=frame_id,
        pose_classifier=pose_classifier,
    )
    frame_data["pose_predictions"] = [
        dict(pred)
        for pred in (frame_data.get("pose_predictions") or [])
        if float(pred.get("confidence", 0.0)) >= float(min_confidence)
    ]
    frame_data["detections"] = original_detections


def _predict_filtered_pose_predictions(
    *,
    frame,
    detections: list[dict],
    current_stage_desc: dict,
    frame_id: int,
    pose_classifier,
    min_confidence: float,
    region_registry,
    pre: FramePreprocessor,
    assembly_target_only: bool,
    region_ids: list[str] | None = None,
) -> list[dict]:
    temp_frame_data = {"detections": [dict(d) for d in (detections or []) if isinstance(d, dict)]}
    _attach_filtered_pose_predictions(
        frame=frame,
        frame_data=temp_frame_data,
        current_stage_desc=dict(current_stage_desc or {}),
        frame_id=frame_id,
        pose_classifier=pose_classifier,
        min_confidence=min_confidence,
        region_registry=region_registry,
        pre=pre,
        assembly_target_only=assembly_target_only,
        region_ids=region_ids,
    )
    return [dict(p) for p in (temp_frame_data.get("pose_predictions") or []) if isinstance(p, dict)]


def _soften_pose_gates(stage_config: list[dict], stage_descriptors: list[dict]) -> None:
    for stage in stage_config:
        if stage.get("expected_poses"):
            stage["enforce_expected_pose_match"] = False
    for desc in stage_descriptors:
        if desc.get("expected_poses"):
            desc["enforce_expected_pose_match"] = False


def _scaled_stage_weights(object_weight_scale: float) -> dict:
    weights = dict(STAGE_WEIGHTS)
    scale = max(0.0, float(object_weight_scale))
    for key in ("active_part_match", "expected_part_match", "inventory_match"):
        if key in weights:
            weights[key] = float(weights[key]) * scale
    return weights


def _guidance_run_options(args, *, effective_color_trigger_enabled: bool, effective_motion_trigger_enabled: bool) -> dict:
    return {
        "mode": "guidance_ui",
        "guidance_mode": args.guidance_mode,
        "experiment_name": args.experiment_name,
        "user_id": args.user_id,
        "operator_id": args.operator_id,
        "session_id": args.session_id,
        "trial_id": args.trial_id,
        "operation_mode": args.operation_mode,
        "anchor_mode": args.anchor_mode,
        "dominant_hand": args.dominant_hand,
        "active_hand_policy": args.active_hand_policy,
        "regions_config": args.regions_config,
        "use_region_detector": bool(args.use_region_detector),
        "motion_fsm_version": args.motion_fsm_version,
        "stage_descriptors_config": args.stage_descriptors_config,
        "no_hand_frames_to_split": int(args.no_hand_frames_to_split),
        "rearm_no_hand_frames": int(args.rearm_no_hand_frames),
        "min_segment_points": int(args.min_segment_points),
        "color_trigger_enabled": bool(effective_color_trigger_enabled),
        "motion_trigger_enabled": bool(effective_motion_trigger_enabled),
        "object_detection_min_confidence": float(args.object_detection_min_confidence),
        "disable_object_detector": bool(args.disable_object_detector),
        "object_score_weight_scale": float(args.object_score_weight_scale),
        "pose_classifier_min_confidence": float(args.pose_classifier_min_confidence),
        "pose_detection_regions": csv_list(args.pose_detection_regions),
        "pose_all_workspace": bool(args.pose_all_workspace),
        "enforce_pose_gate": bool(args.enforce_pose_gate),
        "async_inference": not bool(args.synchronous_inference),
        "detector_min_interval_sec": float(args.detector_min_interval_sec),
        "pose_min_interval_sec": float(args.pose_min_interval_sec),
        "remote_inference_min_interval_sec": float(args.remote_inference_min_interval_sec),
        "hand_frame_scale": float(args.hand_frame_scale),
        "remote_inference_url": args.remote_inference_url,
        "input_video": args.input_video,
        "reference_cache_enabled": not bool(args.no_reference_cache),
        "reference_cache_dir": args.reference_cache_dir,
        "record_video_output": args.record_video_output,
        "reference_demo_dir": args.reference_demo_dir,
        "reference_demo_speed": int(args.reference_demo_speed),
    }


def _start_guidance_metadata(
    *,
    args,
    run_options: dict,
    stage_names: list[str],
    expected_tools: list[str],
    stage_config: list[dict],
    stage_ref_videos: dict[str, list[Path]],
    calib_path: Path,
    guidance_weights: dict,
):
    if args.disable_experiment_metadata:
        return None
    MetadataManager.reset_instance()
    metadata = MetadataManager.get_instance()
    references = [str(p) for refs in stage_ref_videos.values() for p in refs]
    all_refs = [p for refs in stage_ref_videos.values() for p in refs]
    observed_full_video = Path(args.input_video) if args.input_video else None
    system_config = {
        "mode": "guidance_ui",
        "guidance_mode": args.guidance_mode,
        "operation_mode": args.operation_mode,
        "anchor_mode": args.anchor_mode,
        "dominant_hand": args.dominant_hand,
        "active_hand_policy": args.active_hand_policy,
        "regions_config": args.regions_config,
        "stage_descriptors_config": args.stage_descriptors_config,
        "stage_weights": guidance_weights,
        "remote_inference": {
            "enabled": bool(args.remote_inference_url),
            "url": args.remote_inference_url,
            "timeout_sec": float(args.remote_inference_timeout_sec),
            "jpeg_quality": int(args.remote_inference_jpeg_quality),
        },
        "guidance_filters": {
            "object_detection_min_confidence": float(args.object_detection_min_confidence),
            "disable_object_detector": bool(args.disable_object_detector),
            "object_score_weight_scale": float(args.object_score_weight_scale),
            "pose_classifier_min_confidence": float(args.pose_classifier_min_confidence),
            "pose_detection_regions": csv_list(args.pose_detection_regions),
            "pose_all_workspace": bool(args.pose_all_workspace),
            "enforce_pose_gate": bool(args.enforce_pose_gate),
            "async_inference": not bool(args.synchronous_inference),
            "detector_min_interval_sec": float(args.detector_min_interval_sec),
            "pose_min_interval_sec": float(args.pose_min_interval_sec),
            "remote_inference_min_interval_sec": float(args.remote_inference_min_interval_sec),
            "hand_frame_scale": float(args.hand_frame_scale),
        },
        "segment_split_policy": {
            "no_hand_frames_to_split": int(args.no_hand_frames_to_split),
            "rearm_no_hand_frames": int(args.rearm_no_hand_frames),
            "min_segment_points": int(args.min_segment_points),
            "terminal_no_hand_frames_to_close": int(args.terminal_no_hand_frames_to_close),
            "terminal_min_segment_points": int(args.terminal_min_segment_points),
        },
        "reference_cache": {
            "enabled": not bool(args.no_reference_cache),
            "dir": args.reference_cache_dir,
        },
        "record_video": {
            "enabled": bool(args.record_video_output),
            "output": args.record_video_output,
        },
        "reference_demo": {
            "dir": args.reference_demo_dir,
            "speed": int(args.reference_demo_speed),
        },
    }
    protocol_meta = build_protocol_metadata(
        protocol_version=PROTOCOL_VERSION,
        run_options=run_options,
        stage_names=stage_names,
        expected_tools=expected_tools,
        stage_config=stage_config,
        references=all_refs,
        observed_videos=[],
        observed_full_video=observed_full_video,
        calibration_path=calib_path,
    )
    metadata.create_experiment(
        experiment_name=args.experiment_name,
        user_id=args.user_id,
        session_id=args.session_id,
        references=references,
        config=system_config,
        session_meta={"operator_id": args.operator_id, "trial_id": args.trial_id},
        protocol_meta=protocol_meta,
        root=args.experiments_root,
    )
    return metadata


def _log_guidance_stage_decision(
    *,
    metadata,
    feedback: dict,
    stage_name: str,
    trajectory: list[dict],
    raw_oob_count: int,
    frame_id: int,
    active_start_frame: int | None,
    active_start_time: float | None,
    timestamp: float,
    fsm_transitions: list[dict],
    source_label: str,
):
    if metadata is None:
        return
    metadata.add_fsm_decision(
        {
            "stage": stage_name,
            "operation_mode": metadata.metadata.get("system_config", {}).get("operation_mode"),
            "feedback": dict(feedback),
            "raw_oob_count": int(raw_oob_count),
            "observed_video": source_label,
            "input_segment_index": len(metadata.metadata.get("fsm_decisions", [])),
            "segment": {
                "start_frame": active_start_frame,
                "end_frame": int(frame_id),
                "start_time": active_start_time,
                "end_time": float(timestamp),
                "num_points": len(trajectory),
            },
            "fsm_transitions": list(fsm_transitions),
        }
    )


def _format_feedback_lines(
    frame_data: dict | None,
    motion_state: str,
    last_feedback: dict | None,
    last_rejects: list[dict],
    stage_desc: dict | None,
) -> list[str]:
    frame_data = frame_data or {}
    pose_predictions = frame_data.get("pose_predictions") or []
    lines = [f"Tracking: {motion_state.replace('_', ' ').title()}"]
    motion_results = frame_data.get("motion_trigger_results") or {}
    if motion_results:
        active = [rid for rid, res in motion_results.items() if bool(getattr(res, "active", False))]
        best_rid = None
        best_score = 0.0
        for rid, res in motion_results.items():
            score = float(getattr(res, "motion_score", 0.0) or 0.0)
            if score >= best_score:
                best_rid = rid
                best_score = score
        if active:
            lines.extend(wrap_text(f"ROI motion active: {', '.join(active[:2])}", width=44))
        elif best_rid:
            lines.append(f"ROI motion: {best_rid} m={best_score:.3f}")
    expected_pose = expected_pose_lines(stage_desc)
    if expected_pose:
        lines.append("")
        lines.append("Pose check:")
        lines.extend(f"- {line}" for line in expected_pose[:3])
    if pose_predictions:
        best = max(pose_predictions, key=lambda p: float(p.get("confidence", 0.0)))
        lines.append("")
        lines.append(
            f"Pose: {object_name(best.get('part_label'))}: {pose_message(best.get('class_name'))} ({float(best.get('confidence', 0.0)):.2f})"
        )
    if last_rejects:
        reason = last_rejects[-1].get("reason") or last_rejects[-1].get("reason_code")
        lines.extend(["", "Region note:"])
        lines.extend(wrap_text(str(reason), width=44)[:2])
    if last_feedback:
        lines.append("")
        lines.extend(
            [
                f"Final score: {float(last_feedback.get('final_score', 0.0)):.3f}",
                f"Motion match: {float(last_feedback.get('dtw_similarity', 0.0)):.3f}",
            ]
        )
        if "expected_pose_match_score" in last_feedback:
            lines.append(f"Pose match: {float(last_feedback.get('expected_pose_match_score', 0.0)):.3f}")
        if not bool(last_feedback.get("stage_passed", False)):
            lines.extend(_feedback_result_lines(last_feedback, STATUS_FAILED, width=44)[:4])
    return lines


def _draw_reference_video_panel(
    canvas: np.ndarray,
    reference_frame: np.ndarray | None,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    title: str = "Reference",
) -> None:
    ui_scale = _guidance_ui_scale(canvas.shape[1], canvas.shape[0])
    title_scale = 0.62 * ui_scale
    body_scale = 0.56 * ui_scale
    line_h = int(30 * ui_scale)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (18, 20, 22), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 85, 90), 1)
    _draw_text_block_2(canvas, [title], x1 + int(18 * ui_scale), y1 + int(32 * ui_scale), scale=title_scale, color=(235, 235, 235), bold=True)
    content_x1, content_y1 = x1 + 16, y1 + int(48 * ui_scale)
    content_x2, content_y2 = x2 - 16, y2 - 16
    if reference_frame is None:
        msg = "No reference clip found in data/gifs"
        for i, line in enumerate(wrap_text(msg, width=34)[:3]):
            _draw_text_block_2(canvas, [line], content_x1 + 10, content_y1 + 42 + i * line_h, scale=body_scale, color=(185, 185, 185))
        return
    h, w = reference_frame.shape[:2]
    box_w = max(1, content_x2 - content_x1)
    box_h = max(1, content_y2 - content_y1)
    scale = min(box_w / max(1, w), box_h / max(1, h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(reference_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    px = content_x1 + (box_w - new_w) // 2
    py = content_y1 + (box_h - new_h) // 2
    canvas[py : py + new_h, px : px + new_w] = resized


def _status_color(status: str) -> tuple[int, int, int]:
    return {
        STATUS_IN_PROGRESS: (150, 150, 150),
        STATUS_PASSED: (60, 190, 80),
        STATUS_FAILED: (70, 70, 220),
    }.get(status, (150, 150, 150))


def _stage_display_name(stage_name: str) -> str:
    return stage_title(str(stage_name or "UNKNOWN"))


def _frame_live_summary(frame_data: dict | None) -> list[str]:
    return ["Follow the current stage instruction."]


def _stage_row_statuses(
    *,
    stage_names: list[str],
    current_stage_index: int,
    stage_statuses: dict[str, str],
) -> list[str]:
    rows = []
    for idx, name in enumerate(stage_names):
        stored = stage_statuses.get(str(name))
        if stored:
            rows.append(stored)
        elif idx == int(current_stage_index):
            rows.append(STATUS_IN_PROGRESS)
        else:
            rows.append("PENDING")
    return rows

def draw_text(
    img,
    text,
    position,
    size=28,
    color=(255, 255, 255),
    bold=False,
):
    scale = max(0.35, float(size) / 42.0)
    thickness = int(round(scale * 1.55))
    thickness = max(1, min(4, thickness + (1 if bold else 0)))
    cv2.putText(
        img,
        str(text),
        (int(position[0]), int(position[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _build_full_guidance_ui(
    frame,
    *,
    pre: FramePreprocessor,
    region_registry: RegionRegistry | None,
    virtual_projector: VirtualProjectionGuide | None,
    stage_desc: dict,
    projection_stage_descs: list[dict] | None,
    operation_mode: str,
    frame_data: dict | None,
    keypoints_px,
    anchor_mode: str,
    motion_state: str,
    status: str,
    last_feedback: dict | None,
    last_rejects: list[dict],
    reference_frame: np.ndarray | None,
    stage_index: int,
    total_stages: int,
    ui_width: int,
    ui_height: int,
):
    frame_shape = frame.shape
    vis = cv2.flip(frame, -1)
    display_pre = _flipped_display_preprocessor(pre, frame_shape)
    display_keypoints_px = _flip_keypoints_px(keypoints_px, frame_shape)
    display_frame_data = _flip_frame_data_for_display(frame_data, frame_shape)
    if virtual_projector is not None:
        draw_descs = projection_stage_descs or [stage_desc]
        for draw_desc in draw_descs:
            virtual_projector.draw(
                vis,
                pre=display_pre,
                region_registry=region_registry,
                stage_descriptor=draw_desc,
                operation_mode=operation_mode,
            )
    _draw_guidance_regions(
        vis,
        pre=display_pre,
        region_registry=region_registry,
        stage_desc=stage_desc,
        frame_data=frame_data or {},
        operation_mode=operation_mode,
        motion_state=motion_state,
    )
    if display_keypoints_px is not None:
        draw_anchor_overlay(vis, display_keypoints_px, anchor_mode)
    draw_detection_overlay(vis, display_frame_data)
    left_w = int(ui_width * 0.62)
    right_w = ui_width - left_w
    canvas = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)
    canvas[:] = (24, 26, 28)
    # ------------------------------------------------------------
    # UI scaling (base resolution: 1600x900)
    # ------------------------------------------------------------
    ui_scale = _guidance_ui_scale(ui_width, ui_height)

    # Font sizes
    FONT_SMALL = 0.50 * ui_scale
    FONT_MED = 0.60 * ui_scale
    FONT_LARGE = 0.72 * ui_scale
    FONT_STATUS = 1.00 * ui_scale

    # Thickness
    THIN = max(2, int(round(2 * ui_scale)))
    THICK = max(3, int(round(3 * ui_scale)))

    # Layout constants
    M = int(round(24 * ui_scale))       # margin
    P = int(round(44 * ui_scale))       # text padding
    BOX_H = int(round(130 * ui_scale))  # top status box height

    LINE_H_SMALL = int(round(28 * ui_scale))
    LINE_H_MED = int(round(32 * ui_scale))
    LINE_H_LARGE = int(round(36 * ui_scale))

    h, w = vis.shape[:2]
    scale = min(left_w / max(1, w), ui_height / max(1, h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (left_w - new_w) // 2
    y0 = (ui_height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # =====================================================================
    # RIGHT PANEL (Responsive Layout)
    # =====================================================================

    rx = left_w

    # Background
    cv2.rectangle(canvas, (rx, 0), (ui_width - 1, ui_height - 1), (38, 41, 44), -1)

    # ---------- Scaling ----------
    ui_scale = _guidance_ui_scale(ui_width, ui_height)

    MARGIN = int(24 * ui_scale)
    PAD = int(24 * ui_scale)

    FONT_TITLE  = 0.78 * ui_scale
    FONT_HEADER = 0.58 * ui_scale
    FONT_TEXT   = 0.56 * ui_scale
    FONT_SMALL  = 0.45 * ui_scale

    LINE = int(28 * ui_scale)

    panel_w = ui_width - rx

    # ---------- Heights ----------
    STATUS_H = int(ui_height * 0.19)
    INSTR_H = int(ui_height * 0.20)
    FEEDBACK_H = int(ui_height * 0.22)

    status_y1 = MARGIN
    status_y2 = status_y1 + STATUS_H

    instr_y1 = status_y2 + MARGIN
    instr_y2 = instr_y1 + INSTR_H

    feedback_y1 = instr_y2 + MARGIN
    feedback_y2 = feedback_y1 + FEEDBACK_H

    video_y1 = feedback_y2 + MARGIN
    video_y2 = ui_height - int(70 * ui_scale)

    # =====================================================================
    # STATUS
    # =====================================================================

    status_color = _status_color(status)

    cv2.rectangle(
        canvas,
        (rx + MARGIN, status_y1),
        (ui_width - MARGIN, status_y2),
        status_color,
        -1,
    )

    _draw_text_block_2(
        canvas,
        [status.replace("_", " ")],
        rx + PAD + MARGIN,
        status_y1 + int(24 * ui_scale),
        color=(255,255,255),
        scale=FONT_TITLE + 0.18,
        line_h=LINE,
        bold=True,
    )

    stage_name = str(stage_desc.get("stage_name") or "UNKNOWN")

    _draw_text_block_2(
        canvas,
        [
            f"Stage {stage_index + 1}/{total_stages}",
            stage_title(stage_name),
        ],
        rx + PAD + MARGIN,
        status_y1 + int(82 * ui_scale),
        color=(255,255,255),
        scale=FONT_HEADER,
        line_h=LINE,
    )

    # =====================================================================
    # INSTRUCTIONS
    # =====================================================================

    cv2.rectangle(
        canvas,
        (rx + MARGIN, instr_y1),
        (ui_width - MARGIN, instr_y2),
        (18, 20, 22),
        -1,
    )

    cv2.rectangle(
        canvas,
        (rx + MARGIN, instr_y1),
        (ui_width - MARGIN, instr_y2),
        (95, 95, 95),
        THIN,
    )

    _draw_text_block_2(
        canvas,
        ["Instruction"],
        rx + PAD + MARGIN,
        instr_y1 + int(18 * ui_scale),
        color=(245,245,245),
        scale=FONT_HEADER + 0.08,
        line_h=LINE,
        bold=True,
    )

    instruction_lines = action_region_lines(stage_desc)

    _draw_text_block_2(
        canvas,
        instruction_lines,
        rx + PAD + MARGIN,
        instr_y1 + int(60 * ui_scale),
        color=(235,235,235),
        scale=FONT_TEXT,
        line_h=LINE,
    )

    # =====================================================================
    # FEEDBACK
    # =====================================================================

    cv2.rectangle(
        canvas,
        (rx + MARGIN, feedback_y1),
        (ui_width - MARGIN, feedback_y2),
        (34, 48, 58),
        -1,
    )

    cv2.rectangle(
        canvas,
        (rx + MARGIN, feedback_y1),
        (ui_width - MARGIN, feedback_y2),
        (92, 185, 235),
        THICK,
    )

    _draw_text_block_2(
        canvas,
        ["Live feedback"],
        rx + PAD + MARGIN,
        feedback_y1 + int(18 * ui_scale),
        color=(225,245,255),
        scale=FONT_HEADER + 0.08,
        line_h=LINE,
        bold=True,
    )

    feedback_lines = _format_feedback_lines(
        frame_data,
        motion_state,
        last_feedback,
        last_rejects,
        stage_desc,
    )

    feedback_lines = [
        line.strip()
        for line in feedback_lines
        if str(line).strip()
    ]

    _draw_text_block_2(
        canvas,
        feedback_lines[:10],
        rx + PAD + MARGIN,
        feedback_y1 + int(58 * ui_scale),
        color=(235,245,250),
        scale=FONT_TEXT,
        line_h=LINE,
    )

    # =====================================================================
    # VIDEO
    # =====================================================================

    _draw_reference_video_panel(
        canvas,
        cv2.flip(reference_frame, -1) if reference_frame is not None else None,
        x1=rx + MARGIN,
        y1=video_y1,
        x2=ui_width - MARGIN,
        y2=video_y2,
        title="Looping reference",
    )

    # =====================================================================
    # FOOTER
    # =====================================================================

    _draw_text_block_2(
        canvas,
        ["q : quit     r : retry stage     n : force next"],
        rx + PAD + MARGIN,
        ui_height - int(36 * ui_scale),
        color=(180,180,180),
        scale=FONT_SMALL,
        line_h=LINE,
    )

    return canvas

def _build_mid_guidance_ui(
    *,
    stage_names: list[str],
    stage_descriptors: list[dict],
    current_stage_index: int,
    frame_data: dict | None,
    motion_state: str,
    status: str,
    last_feedback: dict |None,
    last_rejects: list[dict],
    stage_statuses: dict[str, str],
    stage_feedbacks: dict[str, dict],
    ui_width: int,
    ui_height: int,
):
    canvas = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)
    canvas[:] = (22, 24, 26)

    # ============================================================
    # SCALE
    # ============================================================
    ui_scale = _guidance_ui_scale(ui_width, ui_height)

    FONT_TITLE = 0.92 * ui_scale
    FONT_HEADER = 0.68 * ui_scale
    FONT_TEXT = 0.58 * ui_scale
    FONT_SMALL = 0.50 * ui_scale
    FONT_STATUS = 0.62 * ui_scale

    M = int(24 * ui_scale)
    PAD = int(20 * ui_scale)

    header_h = int(122 * ui_scale)

    cv2.rectangle(
        canvas,
        (0, 0),
        (ui_width, header_h),
        (34, 37, 40),
        -1,
    )

    # ============================================================
    # TITLE
    # ============================================================

    _draw_text_block_2(
        canvas,
        ["Assembly Guidance - Medium Help"],
        M,
        int(18 * ui_scale),
        scale=FONT_TITLE,
        line_h=int(44 * ui_scale),
        color=(245,245,245),
        bold=True,
    )

    _draw_text_block_2(
        canvas,
        ["Previous / Current / Next stage overview"],
        M,
        int(68 * ui_scale),
        scale=FONT_SMALL,
        line_h=int(28 * ui_scale),
        color=(180,180,180),
    )

    # ============================================================
    # TABLE
    # ============================================================

    total = len(stage_names)
    current_idx = max(0, min(current_stage_index, total - 1))

    window_size = min(3, total)
    start_idx = max(0, min(current_idx - 1, max(0, total - window_size)))
    visible_indices = list(range(start_idx, start_idx + window_size))

    table_top = header_h + M
    table_left = M
    table_right = ui_width - M

    header_row_h = int(64 * ui_scale)

    row_h = int(
        (ui_height - table_top - header_row_h - M)
        / max(1, window_size)
    )
    row_h = max(int(140 * ui_scale), row_h)

    col_stage = M
    col_instruction = int(ui_width * 0.26)
    col_result = int(ui_width * 0.58)
    col_status = int(ui_width * 0.82)

    y = table_top

    cv2.rectangle(
        canvas,
        (table_left, y),
        (table_right, y + header_row_h),
        (48,52,56),
        -1,
    )

    # ============================================================
    # TABLE HEADERS
    # ============================================================

    _draw_text_block_2(
        canvas,
        ["Stage"],
        col_stage,
        y + int(40 * ui_scale),
        scale=FONT_TEXT,
        bold=True,
    )

    _draw_text_block_2(
        canvas,
        ["Instruction"],
        col_instruction,
        y + int(40 * ui_scale),
        scale=FONT_TEXT,
        bold=True,
    )

    _draw_text_block_2(
        canvas,
        ["Result"],
        col_result,
        y + int(40 * ui_scale),
        scale=FONT_TEXT,
        bold=True,
    )

    _draw_text_block_2(
        canvas,
        ["Status"],
        col_status,
        y + int(40 * ui_scale),
        scale=FONT_TEXT,
        bold=True,
    )

    y += header_row_h

    # ============================================================
    # ROWS
    # ============================================================

    for idx in visible_indices:

        name = stage_names[idx]

        is_current = idx == current_idx

        row_status = stage_statuses.get(name, "PENDING")

        bg = (40,48,54) if is_current else (28,30,33)
        border = (92,185,235) if is_current else (60,65,70)

        cv2.rectangle(
            canvas,
            (table_left, y),
            (table_right, y + row_h - 10),
            bg,
            -1,
        )

        cv2.rectangle(
            canvas,
            (table_left, y),
            (table_right, y + row_h - 10),
            border,
            2 if is_current else 1,
        )

        relation = (
            "CURRENT"
            if is_current
            else ("PREV" if idx < current_idx else "NEXT")
        )

        stage_label = f"{idx+1}. {_stage_display_name(name)}"

        _draw_text_block_2(
            canvas,
            [relation, stage_label],
            col_stage,
            y + int(32 * ui_scale),
            scale=FONT_TEXT,
            line_h=int(34 * ui_scale),
            bold=is_current,
        )

        desc = (
            stage_descriptors[idx]
            if idx < len(stage_descriptors)
            else {"stage_name": name}
        )

        instruction_lines = action_region_lines(desc)[:4]

        _draw_text_block_2(
            canvas,
            instruction_lines,
            col_instruction,
            y + int(32 * ui_scale),
            scale=FONT_SMALL,
            line_h=int(31 * ui_scale),
        )

        feedback = stage_feedbacks.get(str(name))

        if (
            is_current
            and last_feedback
            and str(last_feedback.get("stage_name")) == str(name)
        ):
            feedback = last_feedback

        if feedback:
            result_lines = _feedback_result_lines(
                feedback,
                row_status,
                width=26,
            )
        elif is_current:
            result_lines = [
                f"Live: {motion_state}",
                "Waiting..."
            ]
        else:
            result_lines = ["Upcoming"]

        _draw_text_block_2(
            canvas,
            result_lines[:4],
            col_result,
            y + int(32 * ui_scale),
            scale=FONT_SMALL,
            line_h=int(31 * ui_scale),
        )

        box_color = (
            (80,80,80)
            if row_status == "PENDING"
            else _status_color(row_status)
        )

        cv2.rectangle(
            canvas,
            (col_status, y + int(28 * ui_scale)),
            (table_right - M, y + row_h - int(30 * ui_scale)),
            box_color,
            -1,
        )

        _draw_text_block_2(
            canvas,
            [row_status.replace("_", " ")],
            col_status + PAD,
            y + int(row_h * 0.43),
            scale=FONT_STATUS,
            bold=True,
            color=(255,255,255),
        )

        y += row_h

    # ============================================================
    # FOOTER
    # ============================================================

    _draw_text_block_2(
        canvas,
        ["q : quit      r : retry stage      n : next"],
        M,
        ui_height - int(30 * ui_scale),
        scale=FONT_SMALL,
        color=(175,175,175),
    )

    return canvas


def _build_no_guidance_ui(
    *,
    stage_desc: dict,
    stage_index: int,
    total_stages: int,
    status: str,
    last_feedback: dict | None,
    ui_width: int,
    ui_height: int,
):
    canvas = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)
    canvas[:] = (20, 22, 24)
    stage_name = str(stage_desc.get("stage_name") or "UNKNOWN")
    status_color = _status_color(status)
    ui_scale = _guidance_ui_scale(ui_width, ui_height)
    title_scale = 1.15 * ui_scale
    stage_scale = 0.86 * ui_scale
    status_scale = 1.30 * ui_scale
    body_scale = 0.62 * ui_scale
    small_scale = 0.52 * ui_scale
    cx = ui_width // 2
    stage_text = f"Stage {stage_index + 1}/{total_stages}"
    stage_size = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3)[0]
    _draw_text_block_2(
        canvas,
        [stage_text],
        cx - stage_size[0] // 2,
        int(ui_height * 0.26),
        scale=title_scale,
        color=(230, 230, 230),
        bold=True,
    )
    title_lines = wrap_text(_stage_display_name(stage_name), width=32)
    y = int(ui_height * 0.36)
    for line in title_lines[:3]:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, stage_scale, 2)[0]
        _draw_text_block_2(canvas, [line], cx - text_size[0] // 2, y, scale=stage_scale, color=(245, 245, 245), bold=True)
        y += int(44 * ui_scale)
    cv2.rectangle(canvas, (int(ui_width * 0.22), int(ui_height * 0.56)), (int(ui_width * 0.78), int(ui_height * 0.74)), status_color, -1)
    label = status.replace("_", " ")
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 3)[0]
    _draw_text_block_2(
        canvas,
        [label],
        cx - text_size[0] // 2,
        int(ui_height * 0.66),
        scale=status_scale,
        color=(255, 255, 255),
        bold=True,
    )
    if last_feedback:
        result_lines = _feedback_result_lines(last_feedback, status, width=58)
        y = int(ui_height * 0.81)
        for line in result_lines[:4]:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, body_scale, 1)[0]
            _draw_text_block_2(canvas, [line], cx - text_size[0] // 2, y, scale=body_scale, color=(205, 205, 205))
            y += int(32 * ui_scale)
    _draw_text_block_2(
        canvas,
        ["q: quit  r: retry stage  n: force next"],
        int(32 * ui_scale),
        ui_height - int(30 * ui_scale),
        scale=small_scale,
        color=(160, 160, 160),
    )
    return canvas


def _build_ui(
    *,
    guidance_mode: str,
    frame,
    pre: FramePreprocessor,
    region_registry: RegionRegistry | None,
    virtual_projector: VirtualProjectionGuide | None,
    stage_desc: dict,
    projection_stage_descs: list[dict] | None,
    stage_descriptors: list[dict],
    stage_names: list[str],
    operation_mode: str,
    frame_data: dict | None,
    keypoints_px,
    anchor_mode: str,
    motion_state: str,
    status: str,
    last_feedback: dict | None,
    last_rejects: list[dict],
    reference_frame: np.ndarray | None,
    stage_index: int,
    total_stages: int,
    stage_statuses: dict[str, str],
    stage_feedbacks: dict[str, dict],
    ui_width: int,
    ui_height: int,
):
    if guidance_mode == GUIDANCE_MID:
        return _build_mid_guidance_ui(
            stage_names=stage_names,
            stage_descriptors=stage_descriptors,
            current_stage_index=stage_index,
            frame_data=frame_data,
            motion_state=motion_state,
            status=status,
            last_feedback=last_feedback,
            last_rejects=last_rejects,
            stage_statuses=stage_statuses,
            stage_feedbacks=stage_feedbacks,
            ui_width=ui_width,
            ui_height=ui_height,
        )
    if guidance_mode == GUIDANCE_NONE:
        return _build_no_guidance_ui(
            stage_desc=stage_desc,
            stage_index=stage_index,
            total_stages=total_stages,
            status=status,
            last_feedback=last_feedback,
            ui_width=ui_width,
            ui_height=ui_height,
        )
    return _build_full_guidance_ui(
        frame,
        pre=pre,
        region_registry=region_registry,
        virtual_projector=virtual_projector,
        stage_desc=stage_desc,
        projection_stage_descs=projection_stage_descs,
        operation_mode=operation_mode,
        frame_data=frame_data,
        keypoints_px=keypoints_px,
        anchor_mode=anchor_mode,
        motion_state=motion_state,
        status=status,
        last_feedback=last_feedback,
        last_rejects=last_rejects,
        reference_frame=reference_frame,
        stage_index=stage_index,
        total_stages=total_stages,
        ui_width=ui_width,
        ui_height=ui_height,
    )


def _live_packets(
    capture,
    pre: FramePreprocessor,
    dominant_hand: str,
    *,
    source_is_camera: bool,
    hand_frame_scale: float = 1.0,
):
    mp_hands = _get_mediapipe_hands()
    frame_id = 0
    start = time.monotonic()
    prev_labels_by_index = {}
    prev_wrist_by_label = {"LEFT": None, "RIGHT": None}
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MP_MAX_NUM_HANDS,
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    ) as hands_model:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_id += 1
            timestamp = time.monotonic() - start if source_is_camera else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp <= 0:
                timestamp = frame_id / 30.0
            scale = float(hand_frame_scale)
            if 0.1 < scale < 1.0:
                proc_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                scale = 1.0
                proc_frame = frame
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            result = hands_model.process(rgb)
            hand_data = None
            keypoints_px = None
            keypoints_px_by_label = {}
            if result.multi_hand_landmarks:
                h, w = frame.shape[:2]
                ph, pw = proc_frame.shape[:2]
                left_data = right_data = None
                left_kp_px = right_kp_px = None
                handedness_list = getattr(result, "multi_handedness", None) or []
                for i, hand_lm in enumerate(result.multi_hand_landmarks):
                    kp_proc_px = np.array([[p.x * pw, p.y * ph] for p in hand_lm.landmark], dtype=float)
                    kp_px = kp_proc_px / scale
                    kp_norm = pre.warp_keypoints(kp_px)
                    label = None
                    if i < len(handedness_list):
                        cls = handedness_list[i].classification
                        if cls:
                            label = str(cls[0].label).upper()
                    label = _resolve_offline_hand_label(
                        raw_label=label,
                        hand_index=i,
                        kp_norm=kp_norm,
                        prev_labels_by_index=prev_labels_by_index,
                        left_assigned=left_data is not None,
                        right_assigned=right_data is not None,
                    )
                    wrist = (float(kp_norm[0][0]), float(kp_norm[0][1]))
                    prev_wrist = prev_wrist_by_label.get(label)
                    velocity = 0.0
                    if prev_wrist is not None:
                        dx = wrist[0] - float(prev_wrist[0])
                        dy = wrist[1] - float(prev_wrist[1])
                        velocity = float((dx * dx + dy * dy) ** 0.5)
                    prev_wrist_by_label[label] = wrist
                    payload = {
                        "keypoints": kp_norm,
                        "keypoints_px": kp_px,
                        "velocity": velocity,
                        "wrist_px": (float(kp_px[0][0]), float(kp_px[0][1])),
                    }
                    if label == "LEFT":
                        left_data = payload
                        left_kp_px = kp_px
                    else:
                        right_data = payload
                        right_kp_px = kp_px
                hand_data = {"left": left_data, "right": right_data}
                preferred = dominant_hand.upper()
                keypoints_px = (left_kp_px if left_kp_px is not None else right_kp_px) if preferred == "LEFT" else (right_kp_px if right_kp_px is not None else left_kp_px)
                keypoints_px_by_label = {"LEFT": left_kp_px, "RIGHT": right_kp_px}
            yield {
                "frame": frame,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "hand_data": hand_data,
                "keypoints_px": keypoints_px,
                "keypoints_px_by_label": keypoints_px_by_label,
            }


def run_guidance(args) -> None:
    stage_names = csv_list(args.stage_names)
    stage_ref_videos, _ = _parse_references(stage_names, args.reference_videos)
    expected_tools = _expand_csv(args.expected_tools, len(stage_names), "NONE")
    thresholds = [float(v) for v in _expand_csv(args.stage_thresholds, len(stage_names), str(STAGE_SCORE_THRESHOLD))]
    stage_types = _expand_csv(args.stage_types, len(stage_names), "ordered")
    required_counts = [int(v) for v in _expand_csv(args.required_counts, len(stage_names), "1")]
    lockout_enabled = _expand_bool_csv(args.lockout_enabled, len(stage_names), True)
    margin_mins = [float(v) for v in _expand_csv(args.margin_mins, len(stage_names), "0.0")]

    calib_path = Path(args.calibration) if args.calibration else _latest_calibration_dir(CALIB_DATA_ROOT)
    pre = FramePreprocessor(str(calib_path))
    region_registry = RegionRegistry.from_file(args.regions_config) if args.regions_config else None
    region_detector = RegionDetector(region_registry, preprocessor=pre) if (region_registry and args.use_region_detector) else None
    stage_config = _build_stage_config(
        stage_names=stage_names,
        expected_tools=expected_tools,
        thresholds=thresholds,
        stage_types=stage_types,
        required_counts=required_counts,
        lockout_enabled=lockout_enabled,
        margin_mins=margin_mins,
        stage_ref_videos=stage_ref_videos,
    )
    if STAGE_THRESHOLDS_BY_STAGE:
        for s in stage_config:
            if s["name"] in STAGE_THRESHOLDS_BY_STAGE:
                s["score_threshold"] = float(STAGE_THRESHOLDS_BY_STAGE[s["name"]])
    if args.stage_descriptors_config:
        stage_descriptors = load_stage_descriptors(
            config_path=args.stage_descriptors_config,
            operation_mode=args.operation_mode,
            stage_names=stage_names,
        )
    else:
        stage_descriptors = _build_stage_descriptors(stage_names, stage_config, region_registry, args.operation_mode)
    stage_config = _merge_stage_descriptor_expectations(stage_config, stage_descriptors)
    stage_config = _apply_descriptor_stage_overrides(stage_config, stage_descriptors)
    if not args.enforce_pose_gate:
        _soften_pose_gates(stage_config, stage_descriptors)

    pose_classifier = None
    if _descriptors_have_expected_poses(stage_descriptors) and not args.disable_pose_classifier and not args.remote_inference_url:
        pose_classifier = PoseClassifierManager.auto(
            model_map=parse_pose_classifier_model_map(args.pose_classifier_models),
            device=args.pose_classifier_device,
            bbox_pad=float(args.pose_classifier_bbox_pad),
        )

    effective_color_trigger_enabled = bool((args.color_trigger_enabled or COLOR_TRIGGER_ENABLED) and not args.no_color_trigger)
    effective_motion_trigger_enabled = bool(not args.no_motion_trigger)
    pose_detection_region_ids = csv_list(args.pose_detection_regions)
    guidance_weights = _scaled_stage_weights(args.object_score_weight_scale)
    run_options = _guidance_run_options(
        args,
        effective_color_trigger_enabled=effective_color_trigger_enabled,
        effective_motion_trigger_enabled=effective_motion_trigger_enabled,
    )
    metadata = _start_guidance_metadata(
        args=args,
        run_options=run_options,
        stage_names=stage_names,
        expected_tools=expected_tools,
        stage_config=stage_config,
        stage_ref_videos=stage_ref_videos,
        calib_path=calib_path,
        guidance_weights=guidance_weights,
    )
    if metadata is not None:
        logger.info("Guidance experiment metadata: %s", metadata.experiment_path / "metadata.json")

    virtual_projector = (
        VirtualProjectionGuide.from_file(args.virtual_projection_config)
        if args.virtual_projection_enabled
        else None
    )

    cache_key = _reference_cache_key(
        args=args,
        stage_names=stage_names,
        stage_ref_videos=stage_ref_videos,
        calib_path=calib_path,
        effective_color_trigger_enabled=effective_color_trigger_enabled,
        effective_motion_trigger_enabled=effective_motion_trigger_enabled,
    )
    cache_path = Path(args.reference_cache_dir) / f"guidance_refs_{cache_key[:16]}.pkl"
    cached_refs = None if args.no_reference_cache else _load_reference_cache(cache_path)
    if cached_refs is not None:
        stage_references, _stage_reference_names = cached_refs
        logger.info("Loaded cached reference trajectories: %s", cache_path)
    else:
        logger.info("Building reference trajectories before starting live guidance...")
        stage_references, _stage_reference_names = build_stage_references_from_videos(
            stage_names=stage_names,
            stage_ref_videos=stage_ref_videos,
            stage_ref_depths=None,
            mode="offline",
            pre=pre,
            anchor_mode=args.anchor_mode,
            dominant_hand=args.dominant_hand,
            active_hand_policy=args.active_hand_policy,
            show=False,
            delay_ms=1,
            logger=logger,
            operation_mode=args.operation_mode,
            region_registry=region_registry,
            region_detector=region_detector,
            motion_fsm_version=args.motion_fsm_version,
            stage_descriptors=stage_descriptors,
            color_trigger_enabled=effective_color_trigger_enabled,
            color_baseline_frames=args.color_baseline_frames,
            color_delta_threshold=args.color_delta_threshold,
            color_delta_range=args.color_delta_range,
            motion_trigger_enabled=effective_motion_trigger_enabled,
            motion_baseline_frames=args.motion_baseline_frames,
            motion_ignore_frames=args.motion_ignore_frames,
            motion_diff_threshold=args.motion_diff_threshold,
            motion_score_threshold=args.motion_score_threshold,
            motion_dwell_frames=args.motion_dwell_frames,
            motion_history_frames=args.motion_history_frames,
            motion_blur_kernel=args.motion_blur_kernel,
            motion_log_interval_frames=args.motion_log_interval_frames,
            motion_global_suppress_threshold=args.motion_global_suppress_threshold,
            motion_global_relative_factor=args.motion_global_relative_factor,
            fused_trigger_threshold=args.fused_trigger_threshold,
            fused_weight_color=args.fused_weight_color,
            fused_weight_depth=args.fused_weight_depth,
            fused_weight_rgb=args.fused_weight_rgb,
        )
        if not args.no_reference_cache:
            _save_reference_cache(cache_path, stage_references, _stage_reference_names)
            logger.info("Saved reference trajectory cache: %s", cache_path)

    stage_fsm = StageFSM(
        stage_config=stage_config,
        weights=guidance_weights,
        stage_references=stage_references,
        load_references=False,
    )
    descriptor_stage_info, descriptor_unordered_groups = _build_descriptor_unordered_groups(stage_descriptors)
    completed_unordered_members: set[str] = set()

    if args.input_video:
        src = args.input_video
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        source_is_camera = False
        logger.info("Using debug video input: %s", src)
    else:
        cap = OakRgbCapture(
            width=args.oak_width,
            height=args.oak_height,
            fps=args.oak_fps,
            warmup_frames=args.oak_warmup_frames,
            output=args.oak_output,
            color_correction=args.oak_color_correction,
            queue_size=args.oak_queue_size,
        )
        source_is_camera = True
        logger.info(
            "Using live OAK-D RGB stream: %dx%d @ %.1f FPS output=%s color_correction=%s",
            int(args.oak_width),
            int(args.oak_height),
            float(args.oak_fps),
            str(args.oak_output),
            str(args.oak_color_correction),
        )
    if not cap.isOpened():
        raise RuntimeError("Cannot open OAK-D stream." if source_is_camera else f"Cannot open input video: {args.input_video}")

    remote_inference = (
        RemoteVisionInferenceClient(
            args.remote_inference_url,
            timeout_sec=float(args.remote_inference_timeout_sec),
            jpeg_quality=int(args.remote_inference_jpeg_quality),
        )
        if args.remote_inference_url
        else None
    )
    object_detector = None if (remote_inference is not None or args.disable_object_detector) else _make_object_detector()
    async_enabled = not bool(args.synchronous_inference)
    detector_runner = AsyncLatestRunner(
        "object_detector",
        min_interval_sec=float(args.detector_min_interval_sec),
        enabled=async_enabled,
    )
    pose_runner = AsyncLatestRunner(
        "pose_classifier",
        min_interval_sec=float(args.pose_min_interval_sec),
        enabled=async_enabled,
    )
    remote_runner = AsyncLatestRunner(
        "remote_inference",
        min_interval_sec=float(args.remote_inference_min_interval_sec),
        enabled=async_enabled,
    )
    last_inference_stage_name = None
    workspace_filter = _make_workspace_filter(pre)
    sync, selector, motion, builder = _make_runtime(
        args.anchor_mode,
        args.dominant_hand,
        args.active_hand_policy,
        WORK_REGION,
        DROP_REGION,
        rearm_clear_frames=args.rearm_no_hand_frames,
        operation_mode=args.operation_mode,
        motion_fsm_version=args.motion_fsm_version,
        stage_descriptors=stage_descriptors,
        use_motion_triggers=effective_motion_trigger_enabled,
    )
    event_extractor = RegionEventExtractor(region_registry, operation_mode=args.operation_mode) if region_registry else None

    color_trigger = None
    motion_trigger = None
    first_shape = None
    last_feedback = None
    last_rejects: list[dict] = []
    status = STATUS_IN_PROGRESS
    stage_statuses: dict[str, str] = {}
    stage_feedbacks: dict[str, dict] = {}
    active = False
    active_start_frame = None
    active_start_time = None
    active_stage_name = None
    fsm_transitions: list[dict] = []
    raw_oob_count = 0
    no_hand_streak = 0
    record_writer = None
    record_path = None
    reference_looper = ReferenceClipLooper(root=args.reference_demo_dir, speed=args.reference_demo_speed)

    source_label = str(args.input_video) if args.input_video else "oakd_live"
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    try:
        for pkt in _live_packets(
            cap,
            pre,
            args.dominant_hand,
            source_is_camera=source_is_camera,
            hand_frame_scale=args.hand_frame_scale,
        ):
            frame = pkt["frame"]
            frame_id = int(pkt["frame_id"])
            timestamp = float(pkt["timestamp"])
            hand_data = pkt["hand_data"]
            keypoints_px = pkt["keypoints_px"]
            keypoints_px_by_label = pkt.get("keypoints_px_by_label", {})
            if args.record_video_output:
                if record_writer is None:
                    record_path = _recording_output_path(args, metadata)
                    fps = cap.get(cv2.CAP_PROP_FPS) if hasattr(cap, "get") else args.oak_fps
                    record_writer = _open_record_writer(record_path, frame, fps or args.oak_fps)
                    source_label = str(record_path)
                    logger.info("Recording raw guidance feed to: %s", record_path)
                record_writer.write(frame)
            if first_shape is None:
                first_shape = frame.shape[:2]
                if effective_color_trigger_enabled and region_registry is not None:
                    color_trigger = ColorChangeTrigger(
                        roi_masks=_build_color_roi_masks(pre, region_registry, first_shape, args.operation_mode),
                        baseline_frames=args.color_baseline_frames,
                        delta_threshold=args.color_delta_threshold,
                        delta_range=args.color_delta_range,
                    )
                if effective_motion_trigger_enabled and region_registry is not None:
                    motion_trigger = ROIMotionTrigger(
                        roi_masks=_build_color_roi_masks(pre, region_registry, first_shape, args.operation_mode),
                        baseline_frames=args.motion_baseline_frames,
                        ignore_frames=args.motion_ignore_frames,
                        diff_threshold=args.motion_diff_threshold,
                        score_threshold=args.motion_score_threshold,
                        dwell_frames=args.motion_dwell_frames,
                        history_frames=args.motion_history_frames,
                        blur_kernel=args.motion_blur_kernel,
                        global_suppress_threshold=args.motion_global_suppress_threshold,
                        global_relative_factor=args.motion_global_relative_factor,
                    )

            current_stage_desc = _resolve_current_stage_descriptor(motion, stage_descriptors, None)
            current_stage_name = str(current_stage_desc.get("stage_name") or "")
            stage_changed_for_inference = current_stage_name != last_inference_stage_name
            last_inference_stage_name = current_stage_name
            now_perf = time.monotonic()
            remote_pose_predictions = []
            remote_debug = None
            if remote_inference is not None:
                frame_for_inference = frame.copy()
                stage_desc_for_inference = dict(current_stage_desc or {})
                remote_payload = remote_runner.maybe_submit(
                    lambda frame_bgr=frame_for_inference, stage_desc=stage_desc_for_inference, fid=frame_id: {
                        "result": remote_inference.infer(
                            frame_bgr,
                            expected_poses=stage_desc.get("expected_poses"),
                            detection_min_confidence=float(args.object_detection_min_confidence),
                            pose_min_confidence=float(args.pose_classifier_min_confidence),
                            frame_id=fid,
                            stage_name=stage_desc.get("stage_name"),
                        )
                    },
                    now=now_perf,
                    force=stage_changed_for_inference,
                ) or {}
                remote_result = remote_payload.get("result")
                if remote_result is not None:
                    raw_detections = remote_result.detections
                    remote_pose_predictions = remote_result.pose_predictions
                    remote_debug = {
                        "latency_ms": remote_result.latency_ms,
                        "remote_latency_ms": remote_result.remote_latency_ms,
                    }
                else:
                    raw_detections = []
                    remote_debug = {"pending": True}
            else:
                if object_detector is not None:
                    frame_for_detection = frame.copy()
                    raw_detections = detector_runner.maybe_submit(
                        lambda frame_bgr=frame_for_detection: object_detector.detect(frame_bgr),
                        now=now_perf,
                        force=stage_changed_for_inference,
                    )
                else:
                    raw_detections = None
            detections = _filter_detections_by_confidence(
                raw_detections,
                min_confidence=float(args.object_detection_min_confidence),
            )
            frame_data = sync.sync(
                hand_data=hand_data,
                tool_data=_none_tool(timestamp),
                timestamp=timestamp,
                frame_id=frame_id,
                detections=detections,
                expected_parts=current_stage_desc.get("expected_parts"),
                expected_tools=current_stage_desc.get("expected_tools"),
                expected_inventory=current_stage_desc.get("expected_inventory"),
                active_part_candidates=current_stage_desc.get("active_part_candidates"),
                active_tool_candidates=current_stage_desc.get("active_tool_candidates"),
                stage_name=current_stage_desc.get("stage_name"),
                workspace_filter=workspace_filter,
            )
            if remote_inference is not None:
                pose_predictions = _filter_detections_by_confidence(
                    remote_pose_predictions,
                    min_confidence=float(args.pose_classifier_min_confidence),
                )
                if not args.pose_all_workspace:
                    pose_predictions = _filter_detections_by_region(
                        pose_predictions,
                        region_registry=region_registry,
                        pre=pre,
                        region_ids=pose_detection_region_ids,
                    )
                frame_data["pose_predictions"] = pose_predictions
                frame_data["remote_inference"] = remote_debug or {}
            else:
                pose_predictions = []
                if pose_classifier is not None and (current_stage_desc.get("expected_poses") or []):
                    frame_for_pose = frame.copy()
                    detections_for_pose = [dict(d) for d in detections]
                    stage_desc_for_pose = dict(current_stage_desc or {})
                    pose_payload = pose_runner.maybe_submit(
                        lambda frame_bgr=frame_for_pose, dets=detections_for_pose, desc=stage_desc_for_pose, fid=frame_id: {
                            "stage_name": str(desc.get("stage_name") or ""),
                            "predictions": _predict_filtered_pose_predictions(
                                frame=frame_bgr,
                                detections=dets,
                                current_stage_desc=desc,
                                frame_id=fid,
                                pose_classifier=pose_classifier,
                                min_confidence=float(args.pose_classifier_min_confidence),
                                region_registry=region_registry,
                                pre=pre,
                                assembly_target_only=not args.pose_all_workspace,
                                region_ids=pose_detection_region_ids,
                            ),
                        },
                        now=now_perf,
                        force=stage_changed_for_inference,
                    ) or {}
                    if str(pose_payload.get("stage_name") or "") == current_stage_name:
                        pose_predictions = pose_payload.get("predictions") or []
                frame_data["pose_predictions"] = pose_predictions
            cached_point = builder.build_trajectory_point(frame_data)
            frame_data["anchor_valid"] = cached_point is not None
            selected_label = selector.select(frame_data)
            if selected_label is not None:
                keypoints_px = keypoints_px_by_label.get(selected_label)
            detected_regions = region_detector.detect_regions(frame) if region_detector is not None else None
            _annotate_region_hits(
                frame_data=frame_data,
                keypoints_px_by_label=keypoints_px_by_label,
                selected_label=selected_label,
                region_registry=region_registry,
                detected_regions=detected_regions,
                operation_mode=args.operation_mode,
                trigger_point_mode=args.anchor_mode,
            )
            if event_extractor is not None:
                scope = _resolve_event_hand_scope(motion, stage_descriptors)
                event_extractor.update_with_scope(frame_data, frame_id=frame_id, timestamp=timestamp, hand_scope=scope)
                frame_data["fsm_region_hits"] = event_extractor.build_stable_region_hits(frame_data, hand_scope=scope)
            if color_trigger is not None:
                frame_data["color_trigger_results"] = color_trigger.update(frame) or {}
            if motion_trigger is not None:
                motion_results = motion_trigger.update(frame)
                phase = "start" if str(motion.state) == "WAIT_TRIGGER" else "completion"
                allowed = _resolve_motion_allowed_regions(motion, stage_descriptors, args.operation_mode, phase)
                if allowed:
                    motion_results = {rid: res for rid, res in (motion_results or {}).items() if rid in allowed}
                motion_results = _hand_confirmed_motion_results(
                    motion_results,
                    frame_data,
                    require_confirmation=not bool(args.no_motion_hand_confirmation),
                    fallback_score_threshold=float(args.motion_hand_fallback_score_threshold),
                )
                frame_data["motion_trigger_results"] = motion_results or {}
                frame_data["motion_region_hits"] = _build_motion_region_hits(
                    region_registry=region_registry,
                    operation_mode=args.operation_mode,
                    motion_results=motion_results or {},
                )
                if motion_results and args.motion_log_interval_frames > 0:
                    try:
                        if int(frame_id) % int(args.motion_log_interval_frames) == 0:
                            logger.info("[Motion] global=%.4f", float(getattr(motion_trigger, "last_global_motion_score", 0.0)))
                            for rid in sorted(motion_results.keys()):
                                res = motion_results[rid]
                                logger.info(
                                    "[Motion] ROI %s m=%.3f active=%s",
                                    rid,
                                    float(res.motion_score),
                                    bool(res.active),
                                )
                    except Exception:
                        pass
            if region_registry is not None:
                fused_hits, _ = _build_fused_hits(
                    region_registry=region_registry,
                    operation_mode=args.operation_mode,
                    rgb_region_hits=frame_data.get("fsm_region_hits") or frame_data.get("region_hits") or {},
                    depth_results=None,
                    color_results=frame_data.get("color_trigger_results"),
                    threshold=args.fused_trigger_threshold,
                    w_color=args.fused_weight_color,
                    w_depth=args.fused_weight_depth,
                    w_rgb=args.fused_weight_rgb,
                )
                frame_data["fused_region_hits"] = {"active": fused_hits, "any": fused_hits}

            no_hand_streak = 0 if frame_data["hand"]["visible"] else no_hand_streak + 1
            prev_state = motion.state
            motion.update(frame_data)
            last_rejects.extend(getattr(motion, "consume_contract_rejections", lambda: [])())
            last_rejects = last_rejects[-5:]
            transition_reasons = getattr(motion, "consume_transition_reasons", lambda: [])()
            trigger_reason = next((r for r in transition_reasons if r.get("kind") == "trigger"), None)
            completion_reason = next((r for r in transition_reasons if r.get("kind") == "completion"), None)
            if motion.state != prev_state:
                fsm_transitions.append({"from": prev_state, "to": motion.state, "t": timestamp})
                if motion.state == "RECORDING":
                    active = True
                    status = STATUS_IN_PROGRESS
                    last_feedback = None
                    builder = _new_builder(args.dominant_hand, args.anchor_mode)
                    raw_oob_count = 0
                    active_start_frame = frame_id
                    active_start_time = timestamp
                    active_stage_name = str((trigger_reason or {}).get("stage_name") or current_stage_desc.get("stage_name") or "")

            if motion.state == "RECORDING" and active and cached_point is not None:
                builder.append(cached_point)
                if not (0.0 <= cached_point["x"] <= 1.0 and 0.0 <= cached_point["y"] <= 1.0):
                    raw_oob_count += 1

            segment_closed = bool(getattr(motion, "consume_segment_closed", lambda: False)())
            force_close = active and motion.state == "RECORDING" and no_hand_streak >= args.no_hand_frames_to_split
            is_terminal_stage = str(current_stage_desc.get("completion_mode", "")).strip().lower() == "terminal_end"
            terminal_force_close = (
                active
                and is_terminal_stage
                and motion.state == "RECORDING"
                and len(builder.get_trajectory()) >= int(args.terminal_min_segment_points)
                and no_hand_streak >= int(args.terminal_no_hand_frames_to_close)
            )
            if active and (segment_closed or force_close or terminal_force_close):
                completed_stage_name = str((completion_reason or {}).get("stage_name") or active_stage_name or "").strip()
                if completed_stage_name:
                    completed_stage_index = None
                    for idx, stage in enumerate(stage_config):
                        if str(stage.get("name")) == completed_stage_name:
                            stage_fsm.current_stage_index = idx
                            completed_stage_index = idx
                            break
                traj = builder.get_trajectory()
                min_points_for_stage = _stage_min_segment_points(stage_config, completed_stage_name, args.min_segment_points)
                if len(traj) >= min_points_for_stage:
                    last_feedback = stage_fsm.evaluate(traj, raw_oob_count=raw_oob_count)
                    evaluated_stage_name = str(last_feedback.get("stage_name") or completed_stage_name)
                    _postprocess_unordered_segment_progress(
                        stage_fsm=stage_fsm,
                        feedback=last_feedback,
                        evaluated_stage_name=evaluated_stage_name,
                        descriptor_stage_info=descriptor_stage_info,
                        descriptor_unordered_groups=descriptor_unordered_groups,
                        completed_unordered_members=completed_unordered_members,
                    )
                    status = STATUS_PASSED if bool(last_feedback.get("stage_passed", False)) else STATUS_FAILED
                    stage_statuses[evaluated_stage_name] = status
                    stage_feedbacks[evaluated_stage_name] = dict(last_feedback)
                    _log_guidance_stage_decision(
                        metadata=metadata,
                        feedback=last_feedback,
                        stage_name=evaluated_stage_name,
                        trajectory=traj,
                        raw_oob_count=raw_oob_count,
                        frame_id=frame_id,
                        active_start_frame=active_start_frame,
                        active_start_time=active_start_time,
                        timestamp=timestamp,
                        fsm_transitions=fsm_transitions,
                        source_label=source_label,
                    )
                    is_unordered_completed_stage = bool(
                        (descriptor_stage_info.get(evaluated_stage_name) or {}).get("group_id")
                    )
                    if (
                        status == STATUS_PASSED
                        and not bool(last_feedback.get("task_complete", False))
                        and not is_unordered_completed_stage
                    ):
                        next_stage = last_feedback.get("next_stage")
                        if next_stage:
                            for idx, desc in enumerate(stage_descriptors):
                                if str(desc.get("stage_name")) == str(next_stage):
                                    _set_motion_stage_index_preserving_unordered(
                                        motion=motion,
                                        stage_index=idx,
                                        descriptor_stage_info=descriptor_stage_info,
                                        completed_unordered_members=completed_unordered_members,
                                    )
                                    break
                    elif status == STATUS_FAILED and completed_stage_index is not None and hasattr(motion, "set_stage_index"):
                        _set_motion_stage_index_preserving_unordered(
                            motion=motion,
                            stage_index=int(completed_stage_index),
                            descriptor_stage_info=descriptor_stage_info,
                            completed_unordered_members=completed_unordered_members,
                        )
                    logger.info(
                        "Live stage result stage=%s status=%s score=%.3f",
                        last_feedback.get("stage_name"),
                        status,
                        float(last_feedback.get("final_score", 0.0)),
                    )
                else:
                    last_feedback = {
                        "stage_name": completed_stage_name,
                        "stage_passed": False,
                        "final_score": 0.0,
                        "recovery": {"reason": [f"Too few trajectory points: {len(traj)} < {min_points_for_stage}"]},
                    }
                    status = STATUS_FAILED
                    stage_statuses[str(completed_stage_name)] = status
                    stage_feedbacks[str(completed_stage_name)] = dict(last_feedback)
                    _log_guidance_stage_decision(
                        metadata=metadata,
                        feedback=last_feedback,
                        stage_name=completed_stage_name,
                        trajectory=traj,
                        raw_oob_count=raw_oob_count,
                        frame_id=frame_id,
                        active_start_frame=active_start_frame,
                        active_start_time=active_start_time,
                        timestamp=timestamp,
                        fsm_transitions=fsm_transitions,
                        source_label=source_label,
                    )
                active = False
                builder = _new_builder(args.dominant_hand, args.anchor_mode)
                raw_oob_count = 0
                if (force_close or terminal_force_close) and hasattr(motion, "abort_recording_keep_stage"):
                    motion.abort_recording_keep_stage()

            current_stage_index = int(getattr(motion, "current_stage_index", 0))
            current_stage_desc = _resolve_current_stage_descriptor(motion, stage_descriptors, None)
            display_stage_descs = _remaining_unordered_display_descs(motion, current_stage_desc)
            display_stage_desc = _composite_stage_descriptor(display_stage_descs)
            reference_frame = (
                reference_looper.read(str(display_stage_desc.get("stage_name") or ""))
                if args.guidance_mode == GUIDANCE_FULL
                else None
            )
            ui = _build_ui(
                guidance_mode=args.guidance_mode,
                frame=frame,
                pre=pre,
                region_registry=region_registry,
                virtual_projector=virtual_projector,
                stage_desc=display_stage_desc,
                projection_stage_descs=display_stage_descs,
                stage_descriptors=stage_descriptors,
                stage_names=stage_names,
                operation_mode=args.operation_mode,
                frame_data=frame_data,
                keypoints_px=keypoints_px,
                anchor_mode=args.anchor_mode,
                motion_state=str(motion.state),
                status=status,
                last_feedback=last_feedback,
                last_rejects=last_rejects,
                reference_frame=reference_frame,
                stage_index=current_stage_index,
                total_stages=len(stage_names),
                stage_statuses=stage_statuses,
                stage_feedbacks=stage_feedbacks,
                ui_width=args.ui_width,
                ui_height=args.ui_height,
            )
            cv2.imshow(args.window_name, ui)
            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                status = STATUS_IN_PROGRESS
                last_feedback = None
                retry_stage_name = str(_resolve_current_stage_descriptor(motion, stage_descriptors, None).get("stage_name"))
                stage_statuses.pop(retry_stage_name, None)
                stage_feedbacks.pop(retry_stage_name, None)
                if hasattr(motion, "abort_recording_keep_stage"):
                    motion.abort_recording_keep_stage()
                builder = _new_builder(args.dominant_hand, args.anchor_mode)
                active = False
            if key == ord("n"):
                next_idx = min(int(getattr(motion, "current_stage_index", 0)) + 1, len(stage_descriptors) - 1)
                _set_motion_stage_index_preserving_unordered(
                    motion=motion,
                    stage_index=next_idx,
                    descriptor_stage_info=descriptor_stage_info,
                    completed_unordered_members=completed_unordered_members,
                )
                stage_fsm.current_stage_index = next_idx
                status = STATUS_IN_PROGRESS
                last_feedback = None
    finally:
        if metadata is not None:
            metadata.finalize()
            errors = validate_metadata_file(metadata.experiment_path / "metadata.json")
            if errors:
                logger.warning("Guidance metadata validation warnings: %s", errors)
            else:
                logger.info("Guidance metadata finalized: %s", metadata.experiment_path / "metadata.json")
        cap.release()
        if record_writer is not None:
            record_writer.release()
            logger.info("Recorded guidance feed saved: %s", record_path)
        detector_runner.close()
        pose_runner.close()
        remote_runner.close()
        reference_looper.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time assembly guidance UI.")
    parser.add_argument("--operation-mode", choices=["assembly", "dismantle"], default="assembly")
    parser.add_argument("--stage-names", required=True)
    parser.add_argument("--reference-videos", required=True)
    parser.add_argument("--expected-tools", default="NONE")
    parser.add_argument("--experiment-name", default="guidance_ui")
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--operator-id", default="OP_GUIDANCE")
    parser.add_argument("--session-id", default="SESS_GUIDANCE")
    parser.add_argument("--trial-id", default="TRIAL_GUIDANCE")
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--disable-experiment-metadata", action="store_true")
    parser.add_argument(
        "--reference-cache-dir",
        default="experiments/reference_cache",
        help="Directory for cached processed reference trajectories.",
    )
    parser.add_argument(
        "--no-reference-cache",
        action="store_true",
        help="Force rebuilding reference trajectories instead of loading/saving cache.",
    )
    parser.add_argument(
        "--record-video-output",
        default=None,
        help="Optional output .mp4 path. If provided, the raw live/debug RGB feed is recorded.",
    )
    parser.add_argument(
        "--guidance-mode",
        choices=[GUIDANCE_FULL, GUIDANCE_MID, GUIDANCE_NONE],
        default=GUIDANCE_FULL,
        help=(
            "UI assistance level. full_guidance_mode shows live feed/projections/instructions; "
            "mid_guidance_mode shows a table with verbal guidance and live signals; "
            "no_guidance_mode only shows current stage correctness."
        ),
    )
    parser.add_argument("--stage-thresholds", default=None)
    parser.add_argument("--margin-mins", default=None)
    parser.add_argument("--stage-types", default=None)
    parser.add_argument("--required-counts", default=None)
    parser.add_argument("--lockout-enabled", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--regions-config", default="configs/regions_v2.json")
    parser.add_argument("--stage-descriptors-config", default="configs/stage_descriptors_v1.json")
    parser.add_argument("--use-region-detector", action="store_true")
    parser.add_argument("--anchor-mode", choices=["wrist", "wrist_index_midpoint"], default=ANCHOR_MODE_DEFAULT)
    parser.add_argument("--dominant-hand", choices=["RIGHT", "LEFT"], default=DOMINANT_HAND)
    parser.add_argument("--active-hand-policy", choices=["dominant", "closest_to_tool", "best_match"], default=ACTIVE_HAND_POLICY_DEFAULT)
    parser.add_argument("--motion-fsm-version", choices=["v1", "v2"], default=MOTION_FSM_VERSION)
    parser.add_argument("--input-video", default=None, help="Optional debug source. If omitted, uses the live OAK-D RGB stream.")
    parser.add_argument("--oak-width", type=int, default=1280)
    parser.add_argument("--oak-height", type=int, default=720)
    parser.add_argument("--oak-fps", type=float, default=20.0)
    parser.add_argument("--oak-warmup-frames", type=int, default=10)
    parser.add_argument(
        "--oak-queue-size",
        type=int,
        default=1,
        help="OAK-D output queue size. Keep at 1 for lowest-latency live guidance.",
    )
    parser.add_argument(
        "--oak-output",
        choices=["isp", "video", "preview"],
        default="video",
        help="OAK-D ColorCamera output. 'isp' is the default because it avoids live video-path color tinting.",
    )
    parser.add_argument(
        "--oak-color-correction",
        choices=["none", "rgb_to_bgr", "swap_rb", "bgr_to_rgb"],
        default="none",
        help="Optional channel swap if the camera stream appears color-swapped.",
    )
    parser.add_argument("--window-name", default="Assembly Guidance")
    parser.add_argument("--ui-width", type=int, default=3200)
    parser.add_argument("--ui-height", type=int, default=1800)
    parser.add_argument(
        "--reference-demo-dir",
        default="data/gifs",
        help="Folder containing per-stage looping reference clips, e.g. ref_StageA01_01_rgb.mp4.",
    )
    parser.add_argument(
        "--reference-demo-speed",
        type=int,
        default=3,
        help="Number of reference clip frames advanced per UI frame.",
    )
    parser.add_argument("--delay-ms", type=int, default=DELAY_MS)
    parser.add_argument("--no-hand-frames-to-split", type=int, default=NO_HAND_FRAMES_TO_SPLIT)
    parser.add_argument("--rearm-no-hand-frames", type=int, default=REARM_NO_HAND_FRAMES)
    parser.add_argument("--min-segment-points", type=int, default=30)
    parser.add_argument("--virtual-projection-enabled", action="store_true")
    parser.add_argument("--virtual-projection-config", default="configs/virtual_projections_v2.json")
    parser.add_argument("--disable-pose-classifier", action="store_true")
    parser.add_argument("--pose-classifier-models", default=None)
    parser.add_argument("--pose-classifier-device", default=None)
    parser.add_argument("--pose-classifier-bbox-pad", type=float, default=0.20)
    parser.add_argument("--disable-object-detector", action="store_true")
    parser.add_argument("--object-detection-min-confidence", type=float, default=0.50)
    parser.add_argument("--object-score-weight-scale", type=float, default=0.50)
    parser.add_argument("--pose-classifier-min-confidence", type=float, default=0.50)
    parser.add_argument("--pose-detection-regions", default="assembly_target")
    parser.add_argument("--pose-all-workspace", action="store_true")
    parser.add_argument("--enforce-pose-gate", action="store_true")
    parser.add_argument(
        "--hand-frame-scale",
        type=float,
        default=0.75,
        help="Scale frames before MediaPipe hand tracking, then map landmarks back. Use 1.0 for maximum accuracy.",
    )
    parser.add_argument(
        "--synchronous-inference",
        action="store_true",
        help="Disable async object/pose inference and block every frame. Useful only for debugging parity with old behavior.",
    )
    parser.add_argument(
        "--detector-min-interval-sec",
        type=float,
        default=0.20,
        help="Minimum time between async object detector submissions.",
    )
    parser.add_argument(
        "--pose-min-interval-sec",
        type=float,
        default=0.25,
        help="Minimum time between async pose classifier submissions.",
    )
    parser.add_argument(
        "--remote-inference-min-interval-sec",
        type=float,
        default=0.20,
        help="Minimum time between async remote inference submissions.",
    )
    parser.add_argument("--remote-inference-url", default=None)
    parser.add_argument("--remote-inference-timeout-sec", type=float, default=2.0)
    parser.add_argument("--remote-inference-jpeg-quality", type=int, default=80)
    parser.add_argument("--color-trigger-enabled", action="store_true")
    parser.add_argument("--no-color-trigger", action="store_true")
    parser.add_argument("--color-baseline-frames", type=int, default=COLOR_BASELINE_FRAMES)
    parser.add_argument("--color-delta-threshold", type=float, default=COLOR_DELTA_THRESHOLD)
    parser.add_argument("--color-delta-range", type=float, default=COLOR_DELTA_RANGE)
    parser.add_argument("--motion-trigger-enabled", action="store_true")
    parser.add_argument("--no-motion-trigger", action="store_true")
    parser.add_argument(
        "--no-motion-hand-confirmation",
        action="store_true",
        help="Disable the live guidance safety gate that requires hand keypoints inside an active ROI motion region.",
    )
    parser.add_argument(
        "--motion-hand-fallback-score-threshold",
        type=float,
        default=0.55,
        help=(
            "If a hand is visible but MediaPipe misses the exact ROI polygon, allow ROI motion "
            "only when its dwell-averaged score is at least this value."
        ),
    )
    parser.add_argument("--motion-baseline-frames", type=int, default=MOTION_BASELINE_FRAMES)
    parser.add_argument("--motion-ignore-frames", type=int, default=MOTION_FOCUS_IGNORE_FRAMES)
    parser.add_argument("--motion-diff-threshold", type=int, default=MOTION_DIFF_THRESHOLD)
    parser.add_argument("--motion-score-threshold", type=float, default=MOTION_SCORE_THRESHOLD)
    parser.add_argument("--motion-dwell-frames", type=int, default=MOTION_DWELL_FRAMES)
    parser.add_argument("--motion-history-frames", type=int, default=MOTION_HISTORY_FRAMES)
    parser.add_argument("--motion-blur-kernel", type=_parse_kernel, default=MOTION_BLUR_KERNEL)
    parser.add_argument("--motion-log-interval-frames", type=int, default=MOTION_LOG_INTERVAL_FRAMES)
    parser.add_argument("--motion-global-suppress-threshold", type=float, default=MOTION_GLOBAL_SUPPRESS_THRESHOLD)
    parser.add_argument("--motion-global-relative-factor", type=float, default=MOTION_GLOBAL_RELATIVE_FACTOR)
    parser.add_argument("--terminal-no-hand-frames-to-close", type=int, default=60)
    parser.add_argument("--terminal-min-segment-points", type=int, default=30)
    parser.add_argument("--fused-trigger-threshold", type=float, default=FUSED_TRIGGER_THRESHOLD)
    parser.add_argument("--fused-weight-color", type=float, default=FUSED_WEIGHT_COLOR)
    parser.add_argument("--fused-weight-depth", type=float, default=FUSED_WEIGHT_DEPTH)
    parser.add_argument("--fused-weight-rgb", type=float, default=FUSED_WEIGHT_RGB)
    args = parser.parse_args()
    run_guidance(args)


if __name__ == "__main__":
    main()
