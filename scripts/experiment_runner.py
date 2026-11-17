"""
Batch experiment runner for StageLevelOperatorAssistance.

This runner automates repeated calls to main_full_pipeline.py so you can
compare multiple observed videos against the same reference setup and collect
summary metrics/plots in one place.

Example:
python3 experiment_runner.py --config configs/experiment_runner_example.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.utils.logger import get_logger
from assembly_assistant.utils.paths import get_experiments_root


logger = get_logger(__name__)


@dataclass
class CaseResult:
    case_name: str
    status: str
    error: Optional[str]
    error_type: Optional[str]
    experiment_dir: Optional[Path]
    experiment_name: Optional[str]
    metrics: Dict[str, Any]


def _csv(values: List[str]) -> str:
    return ",".join(values)


def _listify(value: Any, field: str) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"Expected list for '{field}', got: {type(value).__name__}")


def _require(config: Dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required config key: '{key}'")
    return config[key]


def _build_command(
    python_bin: str,
    main_script: Path,
    merged: Dict[str, Any],
    experiment_name: str,
) -> List[str]:
    mode = str(merged.get("mode", "offline"))
    stage_names = _listify(_require(merged, "stage_names"), "stage_names")
    reference_videos_by_stage = merged.get("reference_videos_by_stage")
    if reference_videos_by_stage is not None:
        if not isinstance(reference_videos_by_stage, dict):
            raise ValueError("reference_videos_by_stage must be a dict of stage_name -> video_path")
        reference_stage_specs = []
        for stage in stage_names:
            if stage not in reference_videos_by_stage:
                raise ValueError(f"Missing reference video mapping for stage '{stage}'")
            stage_refs = reference_videos_by_stage[stage]
            if isinstance(stage_refs, list):
                if not stage_refs:
                    raise ValueError(f"Stage '{stage}' has empty reference list.")
                spec = "|".join(str(v) for v in stage_refs)
            else:
                spec = str(stage_refs)
            reference_stage_specs.append(spec)
    else:
        reference_stage_specs = _listify(_require(merged, "reference_videos"), "reference_videos")

    expected_tools_by_stage = merged.get("expected_tools_by_stage")
    if expected_tools_by_stage is not None:
        if not isinstance(expected_tools_by_stage, dict):
            raise ValueError("expected_tools_by_stage must be a dict of stage_name -> tool_label")
        expected_tools = []
        for stage in stage_names:
            expected_tools.append(str(expected_tools_by_stage.get(stage, "NONE")))
    else:
        expected_tools = _listify(merged.get("expected_tools", ["NONE"]), "expected_tools")
    stage_thresholds_by_stage = merged.get("stage_thresholds_by_stage")
    if stage_thresholds_by_stage is not None:
        if not isinstance(stage_thresholds_by_stage, dict):
            raise ValueError("stage_thresholds_by_stage must be a dict of stage_name -> threshold")
        stage_thresholds = []
        for stage in stage_names:
            if stage not in stage_thresholds_by_stage:
                raise ValueError(f"Missing stage threshold mapping for stage '{stage}'")
            stage_thresholds.append(str(stage_thresholds_by_stage[stage]))
    else:
        raw_stage_thresholds = merged.get("stage_thresholds")
        stage_thresholds = _listify(raw_stage_thresholds, "stage_thresholds") if raw_stage_thresholds is not None else None
    margin_mins_by_stage = merged.get("margin_mins_by_stage")
    if margin_mins_by_stage is not None:
        if not isinstance(margin_mins_by_stage, dict):
            raise ValueError("margin_mins_by_stage must be a dict of stage_name -> margin_min")
        margin_mins = []
        for stage in stage_names:
            if stage not in margin_mins_by_stage:
                raise ValueError(f"Missing margin_min mapping for stage '{stage}'")
            margin_mins.append(str(margin_mins_by_stage[stage]))
    else:
        raw_margin_mins = merged.get("margin_mins")
        margin_mins = _listify(raw_margin_mins, "margin_mins") if raw_margin_mins is not None else None
    stage_types_by_stage = merged.get("stage_types_by_stage")
    if stage_types_by_stage is not None:
        if not isinstance(stage_types_by_stage, dict):
            raise ValueError("stage_types_by_stage must be a dict of stage_name -> stage_type")
        stage_types = []
        for stage in stage_names:
            if stage not in stage_types_by_stage:
                raise ValueError(f"Missing stage type mapping for stage '{stage}'")
            stage_types.append(str(stage_types_by_stage[stage]))
    else:
        raw_stage_types = merged.get("stage_types")
        stage_types = _listify(raw_stage_types, "stage_types") if raw_stage_types is not None else None
    required_counts_by_stage = merged.get("required_counts_by_stage")
    if required_counts_by_stage is not None:
        if not isinstance(required_counts_by_stage, dict):
            raise ValueError("required_counts_by_stage must be a dict of stage_name -> required_count")
        required_counts = []
        for stage in stage_names:
            if stage not in required_counts_by_stage:
                raise ValueError(f"Missing required_count mapping for stage '{stage}'")
            required_counts.append(str(required_counts_by_stage[stage]))
    else:
        raw_required_counts = merged.get("required_counts")
        required_counts = _listify(raw_required_counts, "required_counts") if raw_required_counts is not None else None
    lockout_enabled_by_stage = merged.get("lockout_enabled_by_stage")
    if lockout_enabled_by_stage is not None:
        if not isinstance(lockout_enabled_by_stage, dict):
            raise ValueError("lockout_enabled_by_stage must be a dict of stage_name -> bool")
        lockout_enabled = []
        for stage in stage_names:
            if stage not in lockout_enabled_by_stage:
                raise ValueError(f"Missing lockout_enabled mapping for stage '{stage}'")
            lockout_enabled.append(str(lockout_enabled_by_stage[stage]))
    else:
        raw_lockout_enabled = merged.get("lockout_enabled")
        lockout_enabled = _listify(raw_lockout_enabled, "lockout_enabled") if raw_lockout_enabled is not None else None
    subtask_groups_by_stage = merged.get("subtask_groups_by_stage")
    if subtask_groups_by_stage is not None:
        if not isinstance(subtask_groups_by_stage, dict):
            raise ValueError("subtask_groups_by_stage must be a dict of stage_name -> subtask-group spec string")
        subtask_groups = []
        for stage in stage_names:
            if stage not in subtask_groups_by_stage:
                raise ValueError(f"Missing subtask_groups mapping for stage '{stage}'")
            subtask_groups.append(str(subtask_groups_by_stage[stage]))
    else:
        raw_subtask_groups = merged.get("subtask_groups")
        subtask_groups = _listify(raw_subtask_groups, "subtask_groups") if raw_subtask_groups is not None else None
    anchor_mode = str(merged.get("anchor_mode", "wrist"))
    user_id = str(merged.get("user_id", "batch_user"))
    operator_id = str(merged.get("operator_id", user_id))
    session_id = str(merged.get("session_id", "batch_session"))
    trial_id = str(merged.get("trial_id", f"trial_{experiment_name}"))
    dominant_hand = str(merged.get("dominant_hand", "RIGHT"))
    active_hand_policy = str(merged.get("active_hand_policy", "dominant"))
    operation_mode = str(merged.get("operation_mode", "dismantle"))
    motion_fsm_version = str(merged.get("motion_fsm_version", "v2"))
    regions_config = merged.get("regions_config")
    use_region_detector = bool(merged.get("use_region_detector", False))
    stage_descriptors_config = merged.get("stage_descriptors_config")

    cmd = [
        python_bin,
        str(main_script),
        "--mode",
        mode,
        "--operation-mode",
        operation_mode,
        "--stage-names",
        _csv(stage_names),
        "--reference-videos",
        _csv(reference_stage_specs),
        "--expected-tools",
        _csv(expected_tools),
        "--anchor-mode",
        anchor_mode,
        "--dominant-hand",
        dominant_hand,
        "--active-hand-policy",
        active_hand_policy,
        "--motion-fsm-version",
        motion_fsm_version,
        "--experiment-name",
        experiment_name,
        "--user-id",
        user_id,
        "--operator-id",
        operator_id,
        "--session-id",
        session_id,
        "--trial-id",
        trial_id,
    ]
    if stage_thresholds is not None:
        cmd.extend(["--stage-thresholds", _csv(stage_thresholds)])
    if margin_mins is not None:
        cmd.extend(["--margin-mins", _csv(margin_mins)])
    if stage_types is not None:
        cmd.extend(["--stage-types", _csv(stage_types)])
    if required_counts is not None:
        cmd.extend(["--required-counts", _csv(required_counts)])
    if lockout_enabled is not None:
        cmd.extend(["--lockout-enabled", _csv(lockout_enabled)])
    if subtask_groups is not None:
        cmd.extend(["--subtask-groups", _csv(subtask_groups)])

    calibration = merged.get("calibration")
    if calibration:
        cmd.extend(["--calibration", str(calibration)])
    if regions_config:
        cmd.extend(["--regions-config", str(regions_config)])
    if use_region_detector:
        cmd.append("--use-region-detector")
    if stage_descriptors_config:
        cmd.extend(["--stage-descriptors-config", str(stage_descriptors_config)])

    if bool(merged.get("show", False)):
        cmd.append("--show")
    if "delay_ms" in merged:
        cmd.extend(["--delay-ms", str(int(merged["delay_ms"]))])
    if "no_hand_frames_to_split" in merged:
        cmd.extend(["--no-hand-frames-to-split", str(int(merged["no_hand_frames_to_split"]))])
    if "rearm_no_hand_frames" in merged:
        cmd.extend(["--rearm-no-hand-frames", str(int(merged["rearm_no_hand_frames"]))])
    if "min_segment_points" in merged:
        cmd.extend(["--min-segment-points", str(int(merged["min_segment_points"]))])
    if bool(merged.get("motion_trigger_enabled", False)):
        cmd.append("--motion-trigger-enabled")
    if "motion_baseline_frames" in merged:
        cmd.extend(["--motion-baseline-frames", str(int(merged["motion_baseline_frames"]))])
    if "motion_ignore_frames" in merged:
        cmd.extend(["--motion-ignore-frames", str(int(merged["motion_ignore_frames"]))])
    if "motion_diff_threshold" in merged:
        cmd.extend(["--motion-diff-threshold", str(int(merged["motion_diff_threshold"]))])
    if "motion_score_threshold" in merged:
        cmd.extend(["--motion-score-threshold", str(float(merged["motion_score_threshold"]))])
    if "motion_dwell_frames" in merged:
        cmd.extend(["--motion-dwell-frames", str(int(merged["motion_dwell_frames"]))])
    if "motion_history_frames" in merged:
        cmd.extend(["--motion-history-frames", str(int(merged["motion_history_frames"]))])
    if "motion_blur_kernel" in merged:
        kernel = merged["motion_blur_kernel"]
        if isinstance(kernel, (list, tuple)) and len(kernel) == 2:
            kernel_str = f"{int(kernel[0])},{int(kernel[1])}"
        else:
            kernel_str = str(kernel)
        cmd.extend(["--motion-blur-kernel", kernel_str])
    if "motion_log_interval_frames" in merged:
        cmd.extend(["--motion-log-interval-frames", str(int(merged["motion_log_interval_frames"]))])
    if "motion_global_suppress_threshold" in merged:
        cmd.extend(["--motion-global-suppress-threshold", str(float(merged["motion_global_suppress_threshold"]))])
    if "motion_global_relative_factor" in merged:
        cmd.extend(["--motion-global-relative-factor", str(float(merged["motion_global_relative_factor"]))])

    observed_full_video = merged.get("observed_full_video")
    observed_videos = merged.get("observed_videos")
    if observed_full_video and observed_videos:
        raise ValueError("Use either 'observed_full_video' or 'observed_videos', not both.")
    if observed_full_video:
        cmd.extend(["--observed-full-video", str(observed_full_video)])
    elif observed_videos:
        cmd.extend(["--observed-videos", _csv(_listify(observed_videos, "observed_videos"))])
    else:
        raise ValueError("Each case must provide observed_full_video or observed_videos.")

    return cmd


def _snapshot_experiment_dirs(root: Path) -> set:
    if not root.exists():
        return set()
    return {p.resolve() for p in root.iterdir() if p.is_dir()}


def _infer_created_experiment_dir(before: set, after: set) -> Optional[Path]:
    created = sorted(list(after - before), key=lambda p: p.stat().st_mtime)
    if not created:
        return None
    return created[-1]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    decisions = metadata.get("fsm_decisions", [])
    final_scores = [d.get("feedback", {}).get("final_score") for d in decisions]
    dtw_scores = [d.get("feedback", {}).get("dtw_similarity") for d in decisions]
    stage_passes = [bool(d.get("feedback", {}).get("stage_passed", False)) for d in decisions]
    raw_oob = [int(d.get("raw_oob_count", 0)) for d in decisions]

    final_scores = [float(v) for v in final_scores if v is not None]
    dtw_scores = [float(v) for v in dtw_scores if v is not None]

    stage_order: List[str] = []
    stage_latest_score: Dict[str, float] = {}
    for d in decisions:
        stage_name = d.get("feedback", {}).get("stage_name") or d.get("stage")
        if not stage_name:
            continue
        if stage_name not in stage_order:
            stage_order.append(stage_name)
        score = d.get("feedback", {}).get("final_score")
        if score is not None:
            stage_latest_score[stage_name] = float(score)

    task_complete = False
    if decisions:
        task_complete = bool(decisions[-1].get("feedback", {}).get("task_complete", False))

    return {
        "attempts": len(decisions),
        "attempt_pass_rate": (sum(stage_passes) / len(stage_passes)) if stage_passes else 0.0,
        "mean_final_score": mean(final_scores) if final_scores else 0.0,
        "mean_dtw_similarity": mean(dtw_scores) if dtw_scores else 0.0,
        "total_raw_oob": sum(raw_oob),
        "task_complete": task_complete,
        "stage_latest_score": stage_latest_score,
        "stage_order": stage_order,
    }


def _classify_failure(stderr: str) -> str:
    s = (stderr or "").lower()
    env_tokens = [
        "modulenotfounderror",
        "importerror",
        "no module named",
        "qt.qpa.plugin",
        "x_link_error",
        "couldn't open stream",
        "device already closed",
        "permissionerror",
    ]
    if any(t in s for t in env_tokens):
        return "environment"
    return "runtime"


def _save_summary(
    run_dir: Path,
    results: List[CaseResult],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "generated_at_epoch_s": time.time(),
        "cases": [
            {
                "case_name": r.case_name,
                "status": r.status,
                "error": r.error,
                "error_type": r.error_type,
                "experiment_dir": str(r.experiment_dir) if r.experiment_dir else None,
                "experiment_name": r.experiment_name,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }
    json_path = run_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "status",
                "experiment_name",
                "attempts",
                "attempt_pass_rate",
                "mean_final_score",
                "mean_dtw_similarity",
                "total_raw_oob",
                "task_complete",
                "error",
                "error_type",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "case_name": r.case_name,
                    "status": r.status,
                    "experiment_name": r.experiment_name,
                    "attempts": r.metrics.get("attempts"),
                    "attempt_pass_rate": r.metrics.get("attempt_pass_rate"),
                    "mean_final_score": r.metrics.get("mean_final_score"),
                    "mean_dtw_similarity": r.metrics.get("mean_dtw_similarity"),
                    "total_raw_oob": r.metrics.get("total_raw_oob"),
                    "task_complete": r.metrics.get("task_complete"),
                    "error": r.error,
                    "error_type": r.error_type,
                }
            )

    logger.info("Saved summary JSON: %s", json_path)
    logger.info("Saved summary CSV: %s", csv_path)


def _save_plots(run_dir: Path, results: List[CaseResult]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    ok_results = [r for r in results if r.status == "ok"]
    if not ok_results:
        logger.warning("No successful runs; skipping plots.")
        return

    case_names = [r.case_name for r in ok_results]
    mean_scores = [float(r.metrics.get("mean_final_score", 0.0)) for r in ok_results]
    pass_rates = [float(r.metrics.get("attempt_pass_rate", 0.0)) for r in ok_results]

    run_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(case_names, mean_scores)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Final Score")
    ax.set_title("Case Comparison: Mean Final Score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path1 = run_dir / "comparison_mean_final_score.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(case_names, pass_rates)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Attempt Pass Rate")
    ax.set_title("Case Comparison: Attempt Pass Rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path2 = run_dir / "comparison_attempt_pass_rate.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)

    logger.info("Saved plots: %s, %s", path1, path2)


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = _read_json(path)
    if "global" not in cfg or "cases" not in cfg:
        raise ValueError("Config must have top-level keys: 'global' and 'cases'.")
    if not isinstance(cfg["cases"], list) or not cfg["cases"]:
        raise ValueError("'cases' must be a non-empty list.")
    return cfg


def _merge_config(global_cfg: Dict[str, Any], case_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(global_cfg)
    merged.update(case_cfg)
    return merged


def _versioned_dir(base: Path, name: str) -> Path:
    i = 1
    while True:
        p = base / f"{name}_v{i}"
        if not p.exists():
            return p
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment runner JSON config.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to run main_full_pipeline.py")
    parser.add_argument("--main-script", default="scripts/main_full_pipeline.py", help="Path to pipeline entry script.")
    parser.add_argument("--experiments-root", default=str(get_experiments_root()), help="Experiments output root.")
    parser.add_argument("--runner-name", default="batch_run", help="Name prefix for this batch summary folder.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining cases if one fails.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    global_cfg = cfg["global"]
    cases = cfg["cases"]

    experiments_root = Path(args.experiments_root)
    experiments_root.mkdir(parents=True, exist_ok=True)

    summary_dir = _versioned_dir(experiments_root, args.runner_name)
    summary_dir.mkdir(parents=True, exist_ok=True)

    main_script = Path(args.main_script)
    if not main_script.exists():
        raise FileNotFoundError(f"main script not found: {main_script}")

    results: List[CaseResult] = []

    for idx, case_cfg in enumerate(cases):
        case_name = str(case_cfg.get("name", f"case_{idx + 1}"))
        merged = _merge_config(global_cfg, case_cfg)
        experiment_name = f"{case_name}"
        logger.info("Running case '%s'...", case_name)

        before = _snapshot_experiment_dirs(experiments_root)
        cmd = _build_command(
            python_bin=args.python_bin,
            main_script=main_script,
            merged=merged,
            experiment_name=experiment_name,
        )
        logger.info("Command: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("Case '%s' failed (exit=%d).", case_name, proc.returncode)
            logger.error("stderr:\n%s", proc.stderr.strip())
            results.append(
                CaseResult(
                    case_name=case_name,
                    status="failed",
                    error=proc.stderr.strip() or f"exit={proc.returncode}",
                    error_type=_classify_failure(proc.stderr),
                    experiment_dir=None,
                    experiment_name=None,
                    metrics={},
                )
            )
            if not args.continue_on_error:
                break
            continue

        after = _snapshot_experiment_dirs(experiments_root)
        exp_dir = _infer_created_experiment_dir(before, after)
        if exp_dir is None:
            logger.error("Case '%s' succeeded but no new experiment directory was found.", case_name)
            results.append(
                CaseResult(
                    case_name=case_name,
                    status="failed",
                    error="no_new_experiment_dir_found",
                    error_type="runtime",
                    experiment_dir=None,
                    experiment_name=None,
                    metrics={},
                )
            )
            if not args.continue_on_error:
                break
            continue

        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            logger.error("Case '%s': metadata.json missing in %s", case_name, exp_dir)
            results.append(
                CaseResult(
                    case_name=case_name,
                    status="failed",
                    error="metadata_missing",
                    error_type="runtime",
                    experiment_dir=exp_dir,
                    experiment_name=exp_dir.name,
                    metrics={},
                )
            )
            if not args.continue_on_error:
                break
            continue

        metadata = _read_json(metadata_path)
        metrics = _extract_metrics(metadata)
        results.append(
            CaseResult(
                case_name=case_name,
                status="ok",
                error=None,
                error_type=None,
                experiment_dir=exp_dir,
                experiment_name=metadata.get("experiment_name", exp_dir.name),
                metrics=metrics,
            )
        )
        logger.info(
            "Case '%s' done: mean_score=%.4f mean_dtw=%.4f task_complete=%s",
            case_name,
            float(metrics.get("mean_final_score", 0.0)),
            float(metrics.get("mean_dtw_similarity", 0.0)),
            bool(metrics.get("task_complete", False)),
        )

    _save_summary(summary_dir, results)
    _save_plots(summary_dir, results)
    logger.info("Batch complete. Summary folder: %s", summary_dir)


if __name__ == "__main__":
    main()
