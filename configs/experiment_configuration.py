"""
Centralized numeric configuration for experiment pipeline defaults.
"""

# Protocol/metadata defaults
# PROTOCOL_VERSION:
#   Increment when experiment metadata schema/checklist is updated.
PROTOCOL_VERSION = "1.1"

# Performer/profile defaults
DOMINANT_HAND = "RIGHT"  # "RIGHT" or "LEFT"; LEFT mirrors x as x = 1 - x.
ANCHOR_MODE_DEFAULT = "wrist_index_midpoint"  # "wrist" or "wrist_index_midpoint".
# ACTIVE_HAND_POLICY_DEFAULT:
#   Frame-level hand selection policy when two hands are visible.
#   - dominant: prefer configured dominant hand.
#   - closest_to_tool: prefer wrist closest to detected tool center (falls back to dominant).
#   - best_match: weighted hybrid (dominant + velocity + tool proximity) with switch hysteresis.
ACTIVE_HAND_POLICY_DEFAULT = "dominant"
# ACTIVE_HAND_WEIGHT_*:
#   Weighted best_match policy coefficients.
#   Keep velocity highest for bimanual/tool-like motion, dominant as tie-break prior,
#   and tool weight lower until tool detection is stable.
ACTIVE_HAND_WEIGHT_DOMINANT = 0.10
ACTIVE_HAND_WEIGHT_VELOCITY = 0.80
ACTIVE_HAND_WEIGHT_TOOL = 0.10
# ACTIVE_HAND_VELOCITY_NORM_SCALE:
#   Velocity normalization scale for weighted best_match (higher => less sensitive).
ACTIVE_HAND_VELOCITY_NORM_SCALE = 0.06
# ACTIVE_HAND_TOOL_DISTANCE_NORM:
#   Tool proximity normalization radius in normalized workspace units.
ACTIVE_HAND_TOOL_DISTANCE_NORM = 0.30
# ACTIVE_HAND_SWITCH_SCORE_MARGIN:
#   Minimum score lead required by challenger hand before switch is considered.
ACTIVE_HAND_SWITCH_SCORE_MARGIN = 0.10
# ACTIVE_HAND_SWITCH_CONFIRM_FRAMES:
#   Number of consecutive frames challenger must maintain lead to switch active hand.
ACTIVE_HAND_SWITCH_CONFIRM_FRAMES = 2
# HANDEDNESS_FALLBACK_IMAGE_MIRRORED:
#   Used only when detector handedness label is missing/ambiguous.
#   False: smaller x -> LEFT, larger x -> RIGHT.
#   True:  smaller x -> RIGHT, larger x -> LEFT.
HANDEDNESS_FALLBACK_IMAGE_MIRRORED = True
# HANDEDNESS_SWAP_LABELS:
#   If True, swaps detector-provided handedness labels (LEFT<->RIGHT) before use.
#   Useful for camera/view setups where detector labels are consistently inverted.
HANDEDNESS_SWAP_LABELS = True

# Motion/workspace defaults
# WORK_REGION:
#   Normalized [x_min, y_min, x_max, y_max] gate that starts recording
#   when trigger point enters this region.
WORK_REGION = (0.05, 0.05, 0.95, 0.95)
# DROP_REGION:
#   Normalized [x_min, y_min, x_max, y_max] gate that ends recording
#   (RECORDING -> DONE) after drop-hit conditions are satisfied.
DROP_REGION = (0.60, 0.00, 1.00, 0.38)

# MOTION_FSM_WARMUP_FRAMES:
#   Ignores first N frames to avoid startup blur / detector boot noise.
MOTION_FSM_WARMUP_FRAMES = 10
# MOTION_FSM_MIN_CONSECUTIVE_VISIBLE:
#   Requires N consecutive frames in WORK_REGION before IDLE -> RECORDING.
MOTION_FSM_MIN_CONSECUTIVE_VISIBLE = 3
# MOTION_FSM_START_NO_ACTIVITY_FRAMES:
#   Allows IDLE -> RECORDING on persistent in-work visibility even when
#   activity is low (disability-friendly slow starts).
MOTION_FSM_START_NO_ACTIVITY_FRAMES = 10
# MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS:
#   Requires N consecutive frames in DROP_REGION before RECORDING -> DONE.
MOTION_FSM_MIN_CONSECUTIVE_DROP_HITS = 2
# MOTION_FSM_MIN_RECORDING_FRAMES:
#   Minimum frames to spend in RECORDING before DONE is allowed.
MOTION_FSM_MIN_RECORDING_FRAMES = 15
# MOTION_FSM_MAX_HAND_JUMP:
#   Rejects unrealistic one-frame jumps of trigger point (normalized units).
MOTION_FSM_MAX_HAND_JUMP = 0.35
# MOTION_FSM_ACTIVITY_V_MIN:
#   Minimal movement/activity threshold for intent-based transitions.
MOTION_FSM_ACTIVITY_V_MIN = 0.003
# MOTION_FSM_PAUSE_TOLERANCE_FRAMES:
#   Number of low-activity frames tolerated while recording.
MOTION_FSM_PAUSE_TOLERANCE_FRAMES = 20
# MOTION_FSM_OCCLUSION_TOLERANCE_FRAMES:
#   Number of fully invisible frames tolerated while recording.
MOTION_FSM_OCCLUSION_TOLERANCE_FRAMES = 30
# MOTION_FSM_DONE_COOLDOWN_FRAMES:
#   Cooldown frames after DONE before FSM returns to IDLE.
MOTION_FSM_DONE_COOLDOWN_FRAMES = 8
# MOTION_FSM_DROP_WINDOW_FRAMES:
#   Rolling window size used for drop-hit accumulation.
MOTION_FSM_DROP_WINDOW_FRAMES = 10
# MOTION_FSM_VERSION:
#   v1 = legacy IDLE/RECORDING/DONE FSM.
#   v2 = policy-based shared engine (WAIT_TRIGGER/RECORDING/WAIT_COMPLETION_CONFIRM/COOLDOWN).
MOTION_FSM_VERSION = "v2"
# MOTION_FSM_START_RECENT_HIT_WINDOW:
#   Sliding window size for forgiving start gating in FSM v2. If a stage accumulates
#   enough validated start hits inside this many frames, RECORDING can begin even when
#   strict consecutive hits are missed by brief/noisy picks.
MOTION_FSM_START_RECENT_HIT_WINDOW = 12
# MOTION_FSM_START_RECENT_HIT_COUNT:
#   Number of validated start hits required inside the recent-hit window above.
#   Keep this lower than strict consecutive start hits so weak but real picks survive.
MOTION_FSM_START_RECENT_HIT_COUNT = 3
# MOTION_FSM_REQUIRE_ANCHOR_BEFORE_COMPLETION:
#   Once a stage has started, do not allow completion to close the segment until
#   at least one valid hand-anchor trajectory point has been observed. This avoids
#   zero-point segments when the trigger fires slightly before hand landmarks stabilize.
MOTION_FSM_REQUIRE_ANCHOR_BEFORE_COMPLETION = True
# MOTION_FSM_ANCHOR_STARTUP_GRACE_FRAMES:
#   After a valid stage trigger, allow this many recording frames for hand landmarks
#   to appear before we treat the segment as a trajectory-less startup. Completion
#   remains blocked until an anchor appears, so segmentation can stay ROI-driven.
MOTION_FSM_ANCHOR_STARTUP_GRACE_FRAMES = 20
# MOTION_FSM_STRICT_MOTION_HITS_ONLY:
#   In motion-trigger mode, gate the FSM from ROI motion hits only. This prevents
#   hand-region fallback from silently retriggering the same stage when the hand
#   detector is noisy or late.
MOTION_FSM_STRICT_MOTION_HITS_ONLY = True

# Stage scoring defaults
# STAGE_TYPES_ALLOWED:
#   Supported stage semantic types. "ordered" is the current sequential mode.
#   "unordered_group" is implemented for order-invariant repeated subtasks.
STAGE_TYPES_ALLOWED = ("ordered", "unordered_group")
# STAGE_TYPE_DEFAULT:
#   Default stage type when not explicitly provided from runtime config.
STAGE_TYPE_DEFAULT = "ordered"
# UNORDERED_GROUP_LOCKOUT_DEFAULT:
#   Default guard flag for unordered groups to prevent repeated counting of
#   already completed subtasks.
UNORDERED_GROUP_LOCKOUT_DEFAULT = True
# UNORDERED_GROUP_REQUIRED_COUNT_DEFAULT:
#   Default number of unique subtasks required for unordered_group completion.
UNORDERED_GROUP_REQUIRED_COUNT_DEFAULT = 1

# STAGE_SCORE_THRESHOLD:
#   Final score threshold for stage pass/fail in StageFSM.
STAGE_SCORE_THRESHOLD = 0.50
# Optional per-stage threshold overrides by stage name.
# Example: {"StageA": 0.92, "StageB": 0.88}
STAGE_THRESHOLDS_BY_STAGE = {"StageA": 0.80, "StageB": 0.70, "StageC": 0.70}
# STAGE_WEIGHTS:
#   Weighted combination of DTW similarity and tool score.
STAGE_WEIGHTS = {
    "dtw": 1.0,
    "tool": 0.0,
    # Conservative auxiliary factors. Keep small and tune with data.
    "smoothness": 0.05,
    "efficiency": 0.03,
    "temporal_stability": 0.02,
}

# Synchronization / trajectory defaults
# SYNC_MAX_TIME_DIFF:
#   Max allowed hand-tool timestamp mismatch when building frame_data.
SYNC_MAX_TIME_DIFF = 0.05
# TOOL_HAND_DISTANCE_THRESHOLD:
#   In normalized coordinates; max hand-tool center distance for association.
TOOL_HAND_DISTANCE_THRESHOLD = 0.10

# MediaPipe defaults (offline modes)
# MP_MAX_NUM_HANDS:
#   Number of hands to track in offline extraction.
#   Phase-2 default is 2 so left+right can be captured simultaneously.
MP_MAX_NUM_HANDS = 2
# MP_MODEL_COMPLEXITY:
#   MediaPipe Hands complexity (0=lightweight, 1=full).
MP_MODEL_COMPLEXITY = 0
# MP_MIN_DETECTION_CONFIDENCE / MP_MIN_TRACKING_CONFIDENCE:
#   Standard MediaPipe thresholds for hand detection/tracking confidence.
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# Full-video segmentation defaults
# NO_HAND_FRAMES_TO_SPLIT:
#   In full-video mode, split segment if hand missing for N frames in RECORDING.
NO_HAND_FRAMES_TO_SPLIT = 200
# REARM_NO_HAND_FRAMES:
#   Legacy name retained for CLI compatibility; currently used as drop-exit
#   rearm gate (frames) before next segment can start.
REARM_NO_HAND_FRAMES = 1
# MIN_SEGMENT_POINTS:
#   Discard tiny/transient segments shorter than N trajectory points.
MIN_SEGMENT_POINTS = 50

# Visualization/runtime defaults
# DELAY_MS:
#   Frame delay for OpenCV display loop when --show is enabled.
DELAY_MS = 40

# Depth-trigger defaults (optional segmentation aid)
DEPTH_TRIGGER_ENABLED = False
DEPTH_BASELINE_FRAMES = 30
DEPTH_DELTA_MM = 15.0
DEPTH_DWELL_FRAMES = 6
DEPTH_FALLBACK_TO_RGB = True
DEPTH_ROI_OVERLAY_ENABLED = True

# Color-change trigger defaults (optional segmentation aid)
COLOR_TRIGGER_ENABLED = False
COLOR_BASELINE_FRAMES = 30
COLOR_DELTA_THRESHOLD = 18.0
COLOR_DELTA_RANGE = 40.0

# ROI motion trigger defaults (primary segmentation aid when enabled)
MOTION_TRIGGER_ENABLED = True
MOTION_BASELINE_FRAMES = 40
MOTION_FOCUS_IGNORE_FRAMES = 40
MOTION_DIFF_THRESHOLD = 12
MOTION_SCORE_THRESHOLD = 0.01
MOTION_DWELL_FRAMES = 4
MOTION_HISTORY_FRAMES = 6
MOTION_BLUR_KERNEL = (5, 5)
MOTION_LOG_INTERVAL_FRAMES = 10
MOTION_GLOBAL_SUPPRESS_THRESHOLD = 0.02
MOTION_GLOBAL_RELATIVE_FACTOR = 2.0
# MOTION_STRONG_SCORE_THRESHOLD:
#   Under high global motion, a region can still activate if its recent ROI motion
#   is absolutely strong enough. This prevents real source-box picks from getting
#   suppressed just because the whole frame is busy.
MOTION_STRONG_SCORE_THRESHOLD = 0.20
# MOTION_STRONG_RELATIVE_FACTOR:
#   Softer relative threshold used only for the strong-ROI override above.
#   Keep this below MOTION_GLOBAL_RELATIVE_FACTOR so very strong local motion can
#   survive global suppression without opening the door to weak false positives.
MOTION_STRONG_RELATIVE_FACTOR = 1.5

# Fused trigger defaults (color + depth + RGB)
FUSED_TRIGGER_THRESHOLD = 0.65
FUSED_WEIGHT_COLOR = 0.6
FUSED_WEIGHT_DEPTH = 0.2
FUSED_WEIGHT_RGB = 0.2

# Motion FSM debug logging
# MOTION_FSM_DEBUG_LOGGING_ENABLED:
#   When True, writes detailed FSM/event debug data to motion_fsm_debug.json.
MOTION_FSM_DEBUG_LOGGING_ENABLED = True
