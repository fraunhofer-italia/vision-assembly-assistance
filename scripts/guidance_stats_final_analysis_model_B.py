"""
Chapter 11 statistical analysis for guidance-level assembly experiments.

Replacement script (v2) implementing:

    1. Primary mixed-effects model (ML) — condition + log_global + interruptions
    2. Per-metric analysis (4 metrics, appropriate transforms, BH correction)
    3. Learning-trajectory analysis (interaction model, CIs, predicted table,
       linear-vs-log AIC/BIC comparison)
    4. Interruption analysis — categorisation of notes into operator errors
       vs system false negatives vs clarifications; secondary covariate models
    5. Task completion and errors — duration LME, failed-stage count model,
       motion/duration correlation, num_stages control
    6. Questionnaire analysis — Friedman + post-hoc Wilcoxon + Kendall's W
       (skipped gracefully if questionnaire CSV not found)
    7. Figures — forest plot, learning curves with CI bands, per-metric
       panels, participant-level box plot, duration plot
    8. Sensitivity — leave-one-participant-out, with/without interrupted trials
    9. residual_outliers.csv        -- stage observations with |z residual| > 3

Outputs (in --output-dir):
    trial_summary.csv
    trial_report.txt
    analysis_dataset.csv
    metric_descriptives.csv
    stage_summary.csv
    lme_report.txt                 -- full primary/interaction/reduced summaries
    per_metric_results.csv         -- 4-metric table with BH-corrected p
    predicted_learning_table.csv   -- predicted skill at cycles 1/5/10/15/20
    sensitivity_loo.csv            -- leave-one-participant-out coefficients
    sensitivity_no_interrupt.csv   -- refit without interrupted trials
    duration_model_report.txt
    error_model_report.txt
    motion_duration_correlation.txt
    questionnaire_report.txt       -- if questionnaire CSV provided
    system_false_negative_rates.csv
    figures/
        fig_forest_condition.png
        fig_learning_curves.png
        fig_per_metric.png
        fig_participant_boxplot.png
        fig_duration.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Analysis is non-interactive and must be reproducible on headless systems.
# Explicitly selecting Agg prevents Matplotlib from attempting to initialise
# the local Tcl/Tk GUI backend while generating output figures.
import matplotlib
matplotlib.use("Agg")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_METRICS = [
    "jerk_index",
    "pause_ratio",
    "micro_stop_count",
    "path_inefficiency_ratio",
]

GUIDANCE_CONDITION_MAP = {
    "full_guidance_mode": "full",
    "mid_guidance_mode": "mid",
    "no_guidance_mode": "no",
}

CONDITION_ORDER = ["full", "mid", "no"]

# The clean experimental dataset used for the Chapter 11 analysis contains
# 70 trials.  This guard prevents a malformed participant--trial mapping from
# silently changing the hierarchical model's clustering structure.
EXPECTED_VALID_TRIALS = 70


@dataclass(frozen=True)
class ExperimentFiles:
    root: Path
    metadata_path: Path


# ============================================================================
# General helpers
# ============================================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_condition_map(items: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid condition map item '{item}'. "
                "Expected EXPERIMENT_NAME=condition."
            )
        key, value = item.split("=", 1)
        output[key.strip()] = value.strip()
    return output


def infer_condition(
    metadata: dict,
    experiment_name: str,
    explicit_map: dict[str, str],
) -> str:
    if experiment_name in explicit_map:
        return explicit_map[experiment_name]

    system_config = metadata.get("system_config") or {}
    guidance_mode = str(
        system_config.get("guidance_mode") or ""
    ).strip()

    if guidance_mode in GUIDANCE_CONDITION_MAP:
        return GUIDANCE_CONDITION_MAP[guidance_mode]

    lowered = experiment_name.lower()
    if re.search(r"(^|[-_])full([-_]|$)|full_guidance", lowered):
        return "full"
    if re.search(r"(^|[-_])mid([-_]|$)|medium|mid_guidance", lowered):
        return "mid"
    if re.search(r"(^|[-_])no([-_]|$)|no_guidance|control", lowered):
        return "no"
    return "unknown"


def iter_experiment_files(root: Path) -> list[ExperimentFiles]:
    return [
        ExperimentFiles(root=p.parent, metadata_path=p)
        for p in sorted(root.glob("*/metadata.json"))
    ]


def iter_stage_entries(metadata: dict):
    decisions = metadata.get("fsm_decisions") or []
    system_config = metadata.get("system_config") or {}
    metadata_task_type = (
        system_config.get("operation_mode")
        or (metadata.get("protocol") or {}).get("run_manifest", {}).get("operation_mode")
    )
    for order, decision in enumerate(decisions, start=1):
        feedback = decision.get("feedback") or {}
        decision_task_type = decision.get("operation_mode") or metadata_task_type
        if decision_task_type not in {"assembly", "dismantle"}:
            raise ValueError(
                "Missing or unsupported operation_mode for an FSM decision: "
                f"{decision_task_type!r}. Expected 'assembly' or 'dismantle'."
            )
        yield order, {
            "task_type": decision_task_type,
            "stage_name": (
                feedback.get("stage_name")
                or decision.get("stage")
            ),
            "stage_passed": bool(
                feedback.get("stage_passed", False)
            ),
            "final_score": feedback.get("final_score"),
            "fatigue_metrics": (
                feedback.get("fatigue_metrics") or {}
            ),
            "segment": decision.get("segment") or {},
        }


# ============================================================================
# Dataset construction — stage level
# ============================================================================

def build_dataset(
    *,
    experiments_root: Path,
    condition_map: dict[str, str],
    metrics: list[str],
    include_unknown_condition: bool,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []
    experiment_files = iter_experiment_files(experiments_root)

    print(f"[INFO] Found {len(experiment_files)} experiment folders.")

    excluded_unknown = 0
    excluded_empty = 0

    for files in experiment_files:
        metadata = load_json(files.metadata_path)
        experiment_name = str(
            metadata.get("experiment_name") or files.root.name
        )
        condition = infer_condition(
            metadata, experiment_name, condition_map
        )
        if condition == "unknown" and not include_unknown_condition:
            excluded_unknown += 1
            continue

        session = metadata.get("session") or {}
        subject = normalize_participant_id(
            session.get("user_id")
            or session.get("operator_id")
            or "unknown_subject"
        )
        trial_id = str(session.get("trial_id") or files.root.name)
        created_at = str(
            metadata.get("created_at")
            or metadata.get("finalized_at")
            or ""
        )

        stage_count = 0
        for stage_order, entry in iter_stage_entries(metadata):
            stage_count += 1
            fatigue = entry["fatigue_metrics"]
            row = {
                "experiment": experiment_name,
                "experiment_path": str(files.root),
                "subject": subject,
                "condition": condition,
                "task_type": entry["task_type"],
                "trial_id": trial_id,
                "created_at": created_at,
                "stage_order": stage_order,
                "stage_name": entry["stage_name"],
                "stage_passed": entry["stage_passed"],
                "final_score": safe_float(entry["final_score"]),
            }
            for metric in metrics:
                row[metric] = safe_float(fatigue.get(metric))
            rows.append(row)

        if stage_count == 0:
            excluded_empty += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No usable experiment rows found under {experiments_root}"
        )

    # Each experiment must represent exactly one operation mode.
    task_counts = df.groupby("experiment_path")["task_type"].nunique()
    inconsistent_tasks = task_counts[task_counts > 1]
    if not inconsistent_tasks.empty:
        raise ValueError(
            "An experiment contains mixed operation_mode values: "
            f"{inconsistent_tasks.index.tolist()[:5]}"
        )

    print(f"[INFO] Excluded unknown-condition experiments: {excluded_unknown}")
    print(f"[INFO] Excluded experiments with no FSM decisions: {excluded_empty}")

    df = df.sort_values(
        ["subject", "created_at", "experiment", "stage_order"]
    ).reset_index(drop=True)

    experiment_order = (
        df[["subject", "experiment", "created_at"]]
        .drop_duplicates()
        .copy()
        .sort_values(["subject", "created_at", "experiment"])
        .reset_index(drop=True)
    )
    experiment_order["cycle_global"] = (
        experiment_order.groupby("subject").cumcount() + 1
    )
    df = df.merge(
        experiment_order[["subject", "experiment", "cycle_global"]],
        on=["subject", "experiment"],
        how="left",
    )
    df["cycle_global"] = df["cycle_global"].astype(int)
    df["log_global"] = np.log(
        df["cycle_global"].astype(float).clip(lower=1.0)
    )

    return df


# ============================================================================
# Trial-level summary
# ============================================================================

def build_trial_summary(
    *,
    experiments_root: Path,
    condition_map: dict[str, str],
    include_unknown_condition: bool,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for files in iter_experiment_files(experiments_root):
        metadata = load_json(files.metadata_path)
        experiment_name = str(
            metadata.get("experiment_name") or files.root.name
        )
        condition = infer_condition(
            metadata, experiment_name, condition_map
        )
        if condition == "unknown" and not include_unknown_condition:
            continue

        session = metadata.get("session") or {}
        subject = normalize_participant_id(
            session.get("user_id")
            or session.get("operator_id")
            or "unknown_subject"
        )
        trial_id = str(session.get("trial_id") or files.root.name)
        created_at = str(
            metadata.get("created_at")
            or metadata.get("finalized_at")
            or ""
        )

        decisions = metadata.get("fsm_decisions") or []
        if not decisions:
            continue

        system_config = metadata.get("system_config") or {}
        task_type = (
            system_config.get("operation_mode")
            or (metadata.get("protocol") or {}).get("run_manifest", {}).get("operation_mode")
            or decisions[0].get("operation_mode")
        )
        if task_type not in {"assembly", "dismantle"}:
            task_type = "unknown"

        segments = [
            d.get("segment") or {}
            for d in decisions
            if d.get("segment")
        ]

        total_duration = None
        if segments:
            start_times = [
                safe_float(s.get("start_time"))
                for s in segments
                if s.get("start_time") is not None
            ]
            end_times = [
                safe_float(s.get("end_time"))
                for s in segments
                if s.get("end_time") is not None
            ]
            start_times = [x for x in start_times if x is not None]
            end_times = [x for x in end_times if x is not None]
            if start_times and end_times:
                total_duration = float(max(end_times) - min(start_times))

        num_stages = len(decisions)

        failed_decisions = []
        failed_unique_stages = set()
        for d in decisions:
            fb = d.get("feedback") or {}
            if not bool(fb.get("stage_passed", False)):
                failed_decisions.append(fb.get("stage_name") or d.get("stage"))
                failed_unique_stages.add(
                    fb.get("stage_name") or d.get("stage")
                )

        rows.append({
            "experiment": experiment_name,
            "experiment_path": str(files.root),
            "subject": subject,
            "condition": condition,
            "task_type": task_type,
            "trial_id": trial_id,
            "created_at": created_at,
            "total_duration_s": total_duration,
            "num_stages": num_stages,
            "num_failed_decisions": len(failed_decisions),
            "num_failed_unique_stages": len(failed_unique_stages),
            "num_failed_stages": len(failed_decisions),  # back-compat
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No trial rows found under {experiments_root}")
    return df.sort_values(
        ["subject", "created_at", "experiment"]
    ).reset_index(drop=True)


# ============================================================================
# Interruptions — merge and categorise
# ============================================================================

def classify_notes(notes_raw: str) -> dict[str, int]:
    """
    Categorise a free-text intervention note into counts of
    system false negatives, operator errors, and clarifications.

    Patterns are intentionally conservative: a note must contain a
    clear keyword to be counted. Multiple keywords in one note count
    once each per category.
    """
    text = (notes_raw or "").lower()

    false_neg = len(re.findall(
        r"false\s*(fail|negative|not\s*trigger)"
        r"|manually\s*passed"
        r"|false\s*negative",
        text,
    ))

    operator_err = len(re.findall(
        r"wrong\s*(part|order|box|tool|orientation|place|position|l-screwdriver|screwdriver)"
        r"|\bforgot\b"
        r"|\bmissed\b"
        r"|\bdid\s*not\s*(hold|remove|unscrew|push)"
        r"|\bmixed\b",
        text,
    ))

    clarification = len(re.findall(
        r"\bask\b|did\s*not\s*know|confus|doubts|unsure|hesitat",
        text,
    ))

    return {
        "n_false_neg": false_neg,
        "n_operator_err": operator_err,
        "n_clarification": clarification,
    }


def merge_interruptions(
    df: pd.DataFrame,
    interruptions_csv: Path,
) -> pd.DataFrame:
    """
    Merge cycle-level interruption counts AND per-cycle categorised
    components (false negatives, operator errors, clarifications).

    Required CSV columns: subject, cycle_global, interruptions.
    Optional: notes_raw (used for categorisation).
    """

    if not interruptions_csv.exists():
        print(
            f"[WARN] Interruptions CSV not found: {interruptions_csv}. "
            "Proceeding with interruptions=0."
        )
        out = df.copy()
        out["interruptions"] = 0.0
        out["n_false_neg"] = 0.0
        out["n_operator_err"] = 0.0
        out["n_clarification"] = 0.0
        return out

    intr = pd.read_csv(interruptions_csv, dtype={"subject": str})
    required = {"subject", "cycle_global", "interruptions"}
    missing = required - set(intr.columns)
    if missing:
        raise ValueError(
            f"Interruptions CSV missing required columns: {sorted(missing)}"
        )

    intr["subject"] = (
        intr["subject"].map(normalize_participant_id)
    )
    intr["cycle_global"] = intr["cycle_global"].astype(int)
    intr["interruptions"] = pd.to_numeric(
        intr["interruptions"], errors="coerce"
    )

    # Categorise notes if present.
    if "notes_raw" in intr.columns:
        categorised = intr["notes_raw"].apply(classify_notes).apply(pd.Series)
        intr = pd.concat([intr, categorised], axis=1)
    else:
        intr["n_false_neg"] = 0
        intr["n_operator_err"] = 0
        intr["n_clarification"] = 0

    keep_cols = [
        "subject", "cycle_global", "interruptions",
        "n_false_neg", "n_operator_err", "n_clarification",
    ]
    intr = (
        intr[keep_cols]
        .drop_duplicates(subset=["subject", "cycle_global"], keep="last")
    )

    out = df.copy()
    out["subject_key"] = (
        out["subject"].astype(str).str.strip().str.lower()
    )

    merged = out.merge(
        intr.rename(columns={"subject": "subject_key"}),
        on=["subject_key", "cycle_global"],
        how="left",
    )

    n_missing = int(merged["interruptions"].isna().sum())
    if n_missing > 0:
        print(
            f"[INFO] {n_missing} stage rows have no interruption entry; "
            "defaulting to 0."
        )

    for col in [
        "interruptions", "n_false_neg",
        "n_operator_err", "n_clarification",
    ]:
        merged[col] = merged[col].fillna(0).astype(float)

    merged = merged.drop(columns=["subject_key"])
    return merged


# ============================================================================
# Composite skill score
# ============================================================================

def add_motion_skill_score(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:

    out = df.copy()
    usable = []
    for metric in metrics:
        if metric not in out.columns:
            continue
        if int(out[metric].notna().sum()) >= 2:
            usable.append(metric)

    if not usable:
        raise RuntimeError(
            "No requested motion metrics had enough numeric values."
        )

    print("[INFO] Computing globally standardised skill score.")
    quality_cols: list[pd.Series] = []

    for metric in usable:
        values = pd.to_numeric(out[metric], errors="coerce")
        mu = float(values.mean())
        sd = float(values.std(ddof=0))
        if not math.isfinite(sd) or sd <= 1e-12:
            print(f"[WARN] Excluding constant metric '{metric}'.")
            continue
        z = (values - mu) / sd
        quality_cols.append((-z).rename(metric))
        print(f"        {metric}: mean={mu:.6f}, sd={sd:.6f}")

    if not quality_cols:
        raise RuntimeError("All metrics constant or invalid.")

    out["skill_score"] = pd.concat(quality_cols, axis=1).mean(
        axis=1, skipna=True
    )
    out["skill_score_source"] = "motion_quality_global_inverted_z"

    valid = int(out["skill_score"].notna().sum())
    print(f"[INFO] skill_score valid: {valid}/{len(out)}")

    smeans = out.groupby("subject")["skill_score"].mean().round(3).to_dict()
    print("[INFO] Subject mean skill_score (between-participant variance preserved):")
    for s, v in smeans.items():
        print(f"        {s}: {v}")

    return out


# ============================================================================
# MixedLM fitting
# ============================================================================

def validate_and_add_trial_uid(
    df: pd.DataFrame,
    *,
    expected_trials: int = EXPECTED_VALID_TRIALS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate stage-to-trial membership and create a nested trial ID.

    ``trial_id`` is not used because it is not globally unique.  The folder
    identity, paired with the participant identifier, identifies the actual
    recorded trial robustly even when trial labels are reused.
    """

    required = {"subject", "experiment_path", "condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot construct nested trial random effect; missing columns: "
            f"{sorted(missing)}"
        )

    output = df.copy()
    subject_key = output["subject"].astype(str).str.strip().str.lower()
    folder_key = output["experiment_path"].astype(str).str.strip().str.replace(
        "\\", "/", regex=False
    )
    if subject_key.eq("").any() or folder_key.eq("").any():
        raise ValueError("Blank subject or experiment_path prevents trial mapping.")

    output["trial_uid"] = subject_key + "::" + folder_key

    # Every trial must map to one participant and one condition, and every
    # participant--folder pair must map to one generated trial identifier.
    bad_subject = output.groupby("trial_uid")["subject"].nunique()
    bad_condition = output.groupby("trial_uid")["condition"].nunique()
    bad_uid = output.groupby(["subject", "experiment_path"])["trial_uid"].nunique()

    problems: list[str] = []
    if (bad_subject != 1).any():
        problems.append(
            "trial_uid mapped to multiple participants: "
            f"{bad_subject[bad_subject != 1].to_dict()}"
        )
    if (bad_condition != 1).any():
        problems.append(
            "trial_uid mapped to multiple conditions: "
            f"{bad_condition[bad_condition != 1].to_dict()}"
        )
    if (bad_uid != 1).any():
        problems.append(
            "participant--trial folder mapped to multiple trial_uids: "
            f"{bad_uid[bad_uid != 1].to_dict()}"
        )

    n_trial_uid = int(output["trial_uid"].nunique())
    n_subject_trial = int(
        output[["subject", "experiment_path"]].drop_duplicates().shape[0]
    )
    if n_trial_uid != n_subject_trial:
        problems.append(
            f"trial_uid count ({n_trial_uid}) differs from participant--trial "
            f"count ({n_subject_trial})."
        )
    if n_trial_uid != expected_trials:
        problems.append(
            f"Found {n_trial_uid} valid participant--trial combinations; "
            f"expected {expected_trials}."
        )

    if problems:
        raise ValueError("Trial-structure validation failed:\n- " + "\n- ".join(problems))

    return output, {
        "stage_rows": int(len(output)),
        "subjects": int(output["subject"].nunique()),
        "trial_uids": n_trial_uid,
        "subject_trial_pairs": n_subject_trial,
    }

def fit_mixedlm(
    formula: str,
    df: pd.DataFrame,
    label: str,
    *,
    include_trial_intercept: bool = True,
    include_condition_slopes: bool = False,
):
    import statsmodels.formula.api as smf

    if include_trial_intercept and "trial_uid" not in df.columns:
        raise ValueError(
            f"{label} MixedLM requires trial_uid. Run trial-structure validation first."
        )

    # The composite skill-score primary model uses Model A: participant and
    # nested trial random intercepts, without condition-slope variance
    # components. Condition slopes remain available explicitly for the
    # diagnostic B/C comparisons and the separately reported per-metric models.
    vc_formula = {}
    if include_condition_slopes:
        vc_formula.update({
            "condition_mid": "0 + I(condition == 'mid')",
            "condition_no": "0 + I(condition == 'no')",
        })
    if include_trial_intercept:
        vc_formula["trial"] = "0 + C(trial_uid)"

    if include_trial_intercept and include_condition_slopes:
        structure_label = "participant intercept + independent condition slopes + nested trial intercept"
    elif include_trial_intercept:
        structure_label = "participant + nested trial random intercepts (Model A)"
    elif include_condition_slopes:
        structure_label = "participant intercept + independent condition slopes"
    else:
        structure_label = "participant random intercept only"
    print(f"[INFO] Fitting {label} MixedLM ({structure_label})...")

    kwargs = {
        "formula": formula,
        "data": df,
        "groups": df["subject"],
        "re_formula": "1",
    }
    if vc_formula:
        kwargs["vc_formula"] = vc_formula
    model = smf.mixedlm(**kwargs)

    methods = ["lbfgs", "bfgs", "powell", "cg"]
    last_exc = None
    candidates = []

    # Try every supported optimizer and retain the best finite likelihood.
    # This is particularly important for variance-component MixedLMs, where
    # a nominally converged first optimizer can stop at a poorer local solution.
    for method in methods:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = model.fit(
                    reml=False, method=method, maxiter=5000, disp=False
                )
            conv = bool(getattr(res, "converged", False))
            llf = float(getattr(res, "llf", float("nan")))
            warning_text = " | ".join(
                str(w.message) for w in caught
                if issubclass(w.category, Warning)
            )
            if conv and math.isfinite(llf):
                candidates.append((llf, res, method, warning_text))
                print(
                    f"[INFO] {label} candidate converged "
                    f"(method={method}, llf={llf:.4f})."
                )
            else:
                print(
                    f"[WARN] {label} not accepted (method={method}, "
                    f"converged={conv}, llf={llf})."
                )
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] {label} failed (method={method}): {exc}")

    if not candidates:
        raise RuntimeError(
            f"Unable to fit {label} MixedLM. Last error: {last_exc}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_llf, best_res, best_method, best_warning = candidates[0]
    all_warnings = [
        f"{method}: {warning_text}"
        for _, _, method, warning_text in candidates if warning_text
    ]
    if len(candidates) > 1:
        spread = max(item[0] for item in candidates) - min(
            item[0] for item in candidates
        )
        if spread > 1e-4:
            all_warnings.append(
                f"Optimizer likelihood spread={spread:.6g}; selected highest llf."
            )
    if all_warnings:
        print(f"[WARN] {label} optimizer diagnostics: {' || '.join(all_warnings)}")
    best_res._analysis_optimizer = best_method
    best_res._analysis_optimizer_candidates = [
        {"method": method, "log_likelihood": float(llf)}
        for llf, _, method, _ in candidates
    ]
    best_res._analysis_optimizer_warnings = " || ".join(all_warnings)
    print(
        f"[INFO] {label} selected {best_method} "
        f"(best llf={best_llf:.4f}; {len(candidates)} converged candidates)."
    )
    return best_res


def likelihood_ratio_test(reduced, full) -> dict[str, float]:
    from scipy.stats import chi2
    ll_r = float(reduced.llf)
    ll_f = float(full.llf)
    df_r = int(len(reduced.fe_params))
    df_f = int(len(full.fe_params))
    df_diff = df_f - df_r
    lr = 2.0 * (ll_f - ll_r)
    if lr < 0 and abs(lr) < 1e-8:
        lr = 0.0
    if lr < 0:
        print(
            f"[ERROR] Full model log-likelihood ({ll_f:.6f}) is below the "
            f"reduced model ({ll_r:.6f}); the likelihood-ratio test is invalid "
            "until the optimisation issue is resolved."
        )
        return {
            "lr": float("nan"), "df": int(df_diff), "p": float("nan"),
            "ll_reduced": ll_r, "ll_full": ll_f,
        }
    p = float(chi2.sf(lr, df_diff))
    return {
        "lr": float(lr), "df": int(df_diff), "p": p,
        "ll_reduced": ll_r, "ll_full": ll_f,
    }


def compute_learning_slope_contrasts(interaction_model) -> pd.DataFrame:
    """
    Linear contrasts for the log_global slope in each condition, including
    tests of each condition-specific slope against zero.
    """
    from scipy.stats import norm

    params = interaction_model.fe_params
    cov = interaction_model.cov_params().loc[params.index, params.index]
    base_term = "log_global"
    if base_term not in params.index:
        return pd.DataFrame()
    base = float(params[base_term])
    rows = []

    # Reference condition (full)
    base_se = math.sqrt(max(float(cov.loc[base_term, base_term]), 0.0))
    base_z = base / base_se if base_se > 0 else float("nan")
    rows.append({
        "condition": "full", "slope": base, "se": base_se,
        "ci_low": base - 1.96 * base_se,
        "ci_high": base + 1.96 * base_se,
        "p_slope_vs_zero": float(2 * norm.sf(abs(base_z))) if math.isfinite(base_z) else float("nan"),
        "slope_difference_vs_full": 0.0,
        "p_difference_vs_full": np.nan,
    })

    for condition in ["mid", "no"]:
        term = f"C(condition)[T.{condition}]:log_global"
        if term not in params.index:
            continue
        delta = float(params[term])
        variance = (
            float(cov.loc[base_term, base_term])
            + float(cov.loc[term, term])
            + 2.0 * float(cov.loc[base_term, term])
        )
        se = math.sqrt(max(variance, 0.0))
        slope = base + delta
        z = slope / se if se > 0 else float("nan")
        diff_se = float(interaction_model.bse_fe[term])
        diff_p = float(interaction_model.pvalues[term])
        rows.append({
            "condition": condition, "slope": slope, "se": se,
            "ci_low": slope - 1.96 * se,
            "ci_high": slope + 1.96 * se,
            "p_slope_vs_zero": float(2 * norm.sf(abs(z))) if math.isfinite(z) else float("nan"),
            "slope_difference_vs_full": delta,
            "p_difference_vs_full": diff_p,
        })
    return pd.DataFrame(rows)


def fit_models(df: pd.DataFrame):
    """
    Fit reduced, primary, and interaction MixedLM models.

    Returns (reduced, primary, interaction, condition_test,
             interaction_test, df_used).
    """

    needed = [
        "skill_score", "condition", "log_global",
        "interruptions", "subject", "trial_uid",
        "task_type", "stage_order",
    ]
    df_used = df.dropna(subset=needed).copy()

    if df_used.empty:
        raise RuntimeError("No complete rows for MixedLM.")

    n_subj = df_used["subject"].nunique()
    n_cond = df_used["condition"].nunique()
    if n_subj < 2:
        raise RuntimeError("Need at least 2 subjects.")
    if n_cond < 2:
        raise RuntimeError("Need at least 2 conditions.")

    print(
        f"[INFO] MixedLM sample: {len(df_used)} rows, "
        f"{n_subj} subjects, {n_cond} conditions."
    )

    # Stage is represented by its observed FSM decision position, with task
    # type entered as a separate fixed effect. This avoids exact collinearity
    # between task-specific stage_name labels (Axx vs Dxx) and task_type. The
    # additive specification assumes a common stage-position adjustment
    # across task types; it does not claim semantic stage identity.
    stage_terms = "C(task_type) + C(stage_order)"
    reduced_formula = (
        f"skill_score ~ {stage_terms} + log_global + interruptions"
    )
    primary_formula = (
        f"skill_score ~ C(condition) + {stage_terms} "
        "+ log_global + interruptions"
    )
    interaction_formula = (
        f"skill_score ~ C(condition) + {stage_terms} "
        "+ log_global + interruptions + C(condition):log_global"
    )

    # All three inferential models use the same Model A random-effects
    # structure, so the ML likelihood-ratio comparisons differ only in fixed
    # effects: participant intercept + nested trial intercept.
    reduced = fit_mixedlm(
        reduced_formula, df_used, "reduced", include_condition_slopes=False
    )
    primary = fit_mixedlm(
        primary_formula, df_used, "primary", include_condition_slopes=False
    )
    interaction = fit_mixedlm(
        interaction_formula, df_used, "interaction", include_condition_slopes=False
    )
    interaction._learning_slope_contrasts = compute_learning_slope_contrasts(
        interaction
    )

    cond_test = likelihood_ratio_test(reduced, primary)
    inter_test = likelihood_ratio_test(primary, interaction)

    return reduced, primary, interaction, cond_test, inter_test, df_used


def fit_num_stages_sensitivity(
    stage_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Trial-level sensitivity: aggregate stage skill scores to one value per
    trial, then adjust for observed stage count. This gives every trial equal
    weight and avoids treating num_stages as a stage-varying predictor.
    """
    motion = (
        stage_df.groupby(["subject", "experiment"], observed=True)["skill_score"]
        .mean()
        .reset_index()
        .rename(columns={"skill_score": "mean_skill"})
    )
    trial = trial_df.merge(
        motion, on=["subject", "experiment"], how="inner"
    ).dropna(subset=["mean_skill", "num_stages", "condition", "subject"]).copy()
    trial = trial.sort_values(
        ["subject", "created_at", "experiment"]
    ).reset_index(drop=True)
    trial["cycle_global"] = trial.groupby("subject").cumcount() + 1
    trial["log_global"] = np.log(trial["cycle_global"].clip(lower=1))
    trial["condition"] = pd.Categorical(
        trial["condition"], categories=CONDITION_ORDER
    )
    trial["task_type"] = trial["task_type"].astype(str)
    formula = (
        "mean_skill ~ C(condition) + C(task_type) + log_global "
        "+ interruptions + num_stages"
    )
    report = [
        "Trial-level num_stages sensitivity analysis",
        "===========================================",
        f"Trials: {len(trial)}",
        f"Participants: {trial['subject'].nunique()}",
        "One mean skill-score observation per trial; random intercept for subject.",
        f"Formula: {formula} + (1 | subject)",
        "num_stages is treated as a trial-level sensitivity covariate. It is "
        "not included in the primary stage-level model because it may be "
        "affected by condition (and thus may be post-treatment); the "
        "trial-level analysis also gives each trial equal weight.",
        "",
    ]
    model = None
    try:
        model = fit_mixedlm(
            formula, trial, "trial-level num_stages sensitivity",
            include_trial_intercept=False,
        )
        report.append("Primary sensitivity fit: Gaussian MixedLM.")
        report.append(str(model.summary()))
        for term in ["C(condition)[T.mid]", "C(condition)[T.no]", "num_stages"]:
            if term in model.fe_params.index:
                ci = model.conf_int().loc[term]
                report.append(
                    f"{term}: beta={model.fe_params[term]:.6f}, "
                    f"SE={model.bse_fe[term]:.6f}, "
                    f"95% CI [{ci.iloc[0]:.6f}, {ci.iloc[1]:.6f}], "
                    f"p={model.pvalues[term]:.6g}"
                )
    except Exception as exc:
        report.append(f"MixedLM failed: {exc}")
        # A zero/boundary participant variance can make Gaussian MixedLM
        # numerically singular with only eight participants. Fall back to
        # Gaussian GEE with exchangeable within-participant correlation.
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            gee = smf.gee(
                formula, groups="subject", data=trial,
                cov_struct=sm.cov_struct.Exchangeable(),
                family=sm.families.Gaussian(),
            ).fit()
            model = gee
            report.append(
                "Fallback sensitivity fit: Gaussian GEE with exchangeable "
                "within-participant working correlation and robust covariance."
            )
            report.append(str(gee.summary()))
            for term in ["C(condition)[T.mid]", "C(condition)[T.no]", "num_stages"]:
                if term in gee.params.index:
                    ci = gee.conf_int().loc[term]
                    report.append(
                        f"{term}: beta={gee.params[term]:.6f}, "
                        f"SE={gee.bse[term]:.6f}, "
                        f"95% CI [{ci.iloc[0]:.6f}, {ci.iloc[1]:.6f}], "
                        f"p={gee.pvalues[term]:.6g}"
                    )
        except Exception as gee_exc:
            report.append(f"GEE fallback also failed: {gee_exc}")
    (output_dir / "num_stages_sensitivity.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    trial.to_csv(output_dir / "num_stages_sensitivity_data.csv", index=False)
    return {"model": model, "n": len(trial), "data": trial}


def fit_condition_task_interaction(
    df_model: pd.DataFrame,
) -> dict[str, Any]:
    """Sensitivity test for whether condition effects differ by task type."""
    additive_formula = (
        "skill_score ~ C(condition) + C(task_type) + C(stage_order) "
        "+ log_global + interruptions"
    )
    interaction_formula = (
        "skill_score ~ C(condition) * C(task_type) + C(stage_order) "
        "+ log_global + interruptions"
    )
    additive = fit_mixedlm(
        additive_formula, df_model, "condition-by-task additive reference",
        include_trial_intercept=True,
    )
    interaction = fit_mixedlm(
        interaction_formula, df_model, "condition-by-task interaction",
        include_trial_intercept=True,
    )
    test = likelihood_ratio_test(additive, interaction)
    return {
        "additive": additive, "interaction": interaction,
        "test": test, "formula": interaction_formula,
    }


def write_cycle_participation(
    trial_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    trial = trial_df.sort_values(
        ["subject", "created_at", "experiment"]
    ).copy()
    trial["cycle_global"] = trial.groupby("subject").cumcount() + 1
    rows = []
    for cycle, group in trial.groupby("cycle_global"):
        rows.append({
            "cycle_global": int(cycle),
            "n_trials": int(group["experiment"].nunique()),
            "n_participants": int(group["subject"].nunique()),
            "participants": ";".join(sorted(group["subject"].astype(str).unique())),
            "conditions": ";".join(
                f"{k}:{v}" for k, v in group["condition"].value_counts().items()
            ),
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "cycle_participation.csv", index=False)
    return result


def write_participant_summary(
    stage_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    stage = stage_df.groupby("subject", observed=True).agg(
        n_stage_rows=("skill_score", "size"),
        mean_skill_score=("skill_score", "mean"),
        sd_skill_score=("skill_score", "std"),
    ).reset_index()
    trials = trial_df.groupby("subject").agg(
        n_trials=("experiment", "nunique"),
        mean_num_stages=("num_stages", "mean"),
        mean_duration_s=("total_duration_s", "mean"),
    ).reset_index()
    result = stage.merge(trials, on="subject", how="outer")
    result.to_csv(output_dir / "participant_summary.csv", index=False)
    return result


def write_observed_stopping_summary(
    trial_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """
    Export observed session endpoints only. This does not infer the
    participant's subjective stopping reason; that requires a recorded
    self-report field not currently parsed by this script.
    """
    ordered = trial_df.sort_values(
        ["subject", "created_at", "experiment"]
    ).copy()
    rows = []
    for subject, group in ordered.groupby("subject"):
        last = group.iloc[-1]
        rows.append({
            "subject": subject,
            "n_observed_trials": int(group["experiment"].nunique()),
            "last_observed_cycle": int(group["cycle_global"].max()),
            "last_trial_experiment": str(last["experiment"]),
            "last_trial_condition": str(last["condition"]),
            "last_trial_date": str(last["created_at"]),
            "self_reported_stopping_criterion": "not available in parsed metadata",
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "participant_stopping_summary.csv", index=False)
    (output_dir / "stopping_criterion_note.txt").write_text(
        "The analysis script exports each participant's observed final trial "
        "and total number of recorded trials. It does not identify a "
        "self-reported stopping criterion because no such field is parsed "
        "from the experiment metadata or questionnaire in this script. "
        "Do not describe the observed final trial as proof of the reason "
        "the participant stopped. If the protocol/debrief confirms that "
        "participants stopped after reporting no further need for assistance, "
        "that criterion should be stated from the protocol or debrief record, "
        "not inferred from this CSV.",
        encoding="utf-8",
    )
    return result


def write_model_formulas(output_dir: Path) -> None:
    formulas = [
        "PRIMARY STAGE-LEVEL MODEL",
        "skill_score ~ C(condition) + C(task_type) + C(stage_order) + log_global + interruptions + (1|subject) + (1|subject:trial_uid)",
        "",
        "REDUCED MODEL",
        "skill_score ~ C(task_type) + C(stage_order) + log_global + interruptions + (1|subject) + (1|subject:trial_uid)",
        "",
        "CONDITION-BY-LEARNING INTERACTION",
        "skill_score ~ C(condition) + C(task_type) + C(stage_order) + log_global + interruptions + C(condition):log_global + (1|subject) + (1|subject:trial_uid)",
        "",
        "DURATION MODEL",
        "log_duration ~ C(condition) + C(task_type) + log_global + (1|subject)",
        "",
        "DURATION WITH STAGE-COUNT CONTROL",
        "log_duration ~ C(condition) + C(task_type) + log_global + num_stages + (1|subject)",
        "",
        "MOTION-DURATION ADJUSTED MODEL",
        "mean_skill ~ log_duration + log_global + C(condition) + C(task_type) + (1|subject)",
        "",
        "TRIAL-LEVEL NUM_STAGES SENSITIVITY",
        "mean_skill ~ C(condition) + C(task_type) + log_global + interruptions + num_stages + (1|subject)",
        "",
        "CONDITION-BY-TASK SENSITIVITY",
        "skill_score ~ C(condition)*C(task_type) + C(stage_order) + log_global + interruptions + (1|subject) + (1|subject:trial_uid)",
        "",
        "ERROR MODELS",
        "num_failed_unique_stages ~ condition + log_global + num_stages (Poisson GLM and NB2 regression; subject-clustered robust SE)",
    ]
    (output_dir / "model_formulas.txt").write_text(
        "\n".join(formulas) + "\n", encoding="utf-8"
    )


def fit_random_structure_model(
    formula: str,
    df: pd.DataFrame,
    label: str,
    *,
    include_condition_slopes: bool,
    include_trial_intercept: bool,
):
    """Fit a diagnostic random-effects structure by ML.

    Every optimizer is attempted and the converged finite-likelihood result
    with the highest likelihood is retained. This avoids accepting the first
    nominally converged but potentially inferior local solution.
    """
    import statsmodels.formula.api as smf

    vc_formula = {}
    if include_condition_slopes:
        vc_formula.update({
            "condition_mid": "0 + I(condition == 'mid')",
            "condition_no": "0 + I(condition == 'no')",
        })
    if include_trial_intercept:
        if "trial_uid" not in df.columns:
            raise ValueError(f"{label} requires trial_uid.")
        vc_formula["trial"] = "0 + C(trial_uid)"

    kwargs = {
        "formula": formula,
        "data": df,
        "groups": df["subject"],
        "re_formula": "1",
    }
    if vc_formula:
        kwargs["vc_formula"] = vc_formula
    model = smf.mixedlm(**kwargs)

    last_error = None
    attempts = []
    candidates = []
    for method in ["lbfgs", "bfgs", "powell", "cg"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=False, method=method, maxiter=5000, disp=False
                )
            llf = float(result.llf)
            converged = bool(getattr(result, "converged", False))
            attempts.append(method)
            warning_text = " | ".join(
                str(w.message) for w in caught
                if issubclass(w.category, Warning)
            )
            if converged and math.isfinite(llf):
                candidates.append((llf, result, method, warning_text))
                print(
                    f"[INFO] {label} candidate converged "
                    f"(method={method}, llf={llf:.4f})."
                )
            else:
                print(
                    f"[WARN] {label} not accepted (method={method}, "
                    f"converged={converged}, llf={llf})."
                )
        except Exception as exc:
            last_error = str(exc)
            attempts.append(method)
            print(f"[WARN] {label} failed ({method}): {exc}")

    if not candidates:
        raise RuntimeError(
            f"{label} failed all optimizers {attempts}; last error={last_error}"
        )

    # Retain the best finite likelihood across all converged optimizer runs.
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_llf, best_result, best_method, best_warnings = candidates[0]
    all_warnings = []
    for _, _, method, warning_text in candidates:
        if warning_text:
            all_warnings.append(f"{method}: {warning_text}")
    warning_text = " || ".join(all_warnings)
    if len(candidates) > 1:
        llfs = [item[0] for item in candidates]
        spread = max(llfs) - min(llfs)
        if spread > 1e-4:
            warning_text = (warning_text + " || " if warning_text else "") + (
                f"Optimizer likelihood spread={spread:.6g}; selected highest llf."
            )
    print(
        f"[INFO] {label} selected {best_method} "
        f"(best llf={best_llf:.4f}; {len(candidates)} converged candidates)."
    )
    return best_result, best_method, warning_text, attempts


def compare_random_effects_structures(df_model: pd.DataFrame) -> pd.DataFrame:
    """Compare A/B/C random-effects structures with fixed effects and rows held constant.

    A: participant intercept + nested trial intercept.
    B: participant intercept + independent condition slopes + nested trial intercept.
    C: participant intercept + independent condition slopes (no trial intercept).
    All are maximum-likelihood fits of the primary fixed-effects formula.
    """
    formula = (
        "skill_score ~ C(condition) + C(task_type) + C(stage_order) "
        "+ log_global + interruptions"
    )
    specs = [
        ("A", "participant intercept + nested trial intercept", False, True),
        ("B", "participant intercept + independent condition slopes + nested trial intercept", True, True),
        ("C", "participant intercept + independent condition slopes (no trial intercept)", True, False),
    ]
    rows = []
    for code, description, slopes, trial in specs:
        try:
            res, method, warning_text, attempts = fit_random_structure_model(
                formula, df_model, f"Random-structure model {code}",
                include_condition_slopes=slopes,
                include_trial_intercept=trial,
            )
            vc_names = list(getattr(res.model.exog_vc, "names", []))
            vc_values = np.asarray(getattr(res, "vcomp", []), dtype=float)
            vc = dict(zip(vc_names, vc_values))
            row = {
                "model": code,
                "description": description,
                "status": "converged",
                "optimizer": method,
                "attempted_optimizers": ",".join(attempts),
                "n_rows": int(len(df_model)),
                "n_subjects": int(df_model["subject"].nunique()),
                "n_trials": int(df_model["trial_uid"].nunique()),
                "n_fixed_parameters": int(len(res.fe_params)),
                "n_covariance_parameters": int(len(res.params) - len(res.fe_params)),
                "log_likelihood": float(res.llf),
                "AIC": float(res.aic),
                "BIC": float(res.bic),
                "residual_variance": float(res.scale),
                "participant_intercept_variance": float(np.asarray(res.cov_re)[0, 0]),
                "condition_mid_slope_variance": float(vc.get("condition_mid", np.nan)),
                "condition_no_slope_variance": float(vc.get("condition_no", np.nan)),
                "trial_intercept_variance": float(vc.get("trial", np.nan)),
                "warnings": warning_text,
                "condition_mid_beta": float(res.fe_params.get("C(condition)[T.mid]", np.nan)),
                "condition_mid_se": float(res.bse_fe.get("C(condition)[T.mid]", np.nan)),
                "condition_mid_p": float(res.pvalues.get("C(condition)[T.mid]", np.nan)),
                "condition_no_beta": float(res.fe_params.get("C(condition)[T.no]", np.nan)),
                "condition_no_se": float(res.bse_fe.get("C(condition)[T.no]", np.nan)),
                "condition_no_p": float(res.pvalues.get("C(condition)[T.no]", np.nan)),
            }
            rows.append(row)
        except Exception as exc:
            rows.append({
                "model": code,
                "description": description,
                "status": "failed",
                "optimizer": "",
                "attempted_optimizers": "lbfgs,bfgs,powell,cg",
                "n_rows": int(len(df_model)),
                "n_subjects": int(df_model["subject"].nunique()),
                "n_trials": int(df_model["trial_uid"].nunique()),
                "error": str(exc),
            })

    result_df = pd.DataFrame(rows)
    # B contains the same participant/trial intercept structure as A plus
    # additional non-negative variance components. Its maximised likelihood
    # should therefore not be materially below A if optimisation reaches the
    # same likelihood surface. Flag any violation rather than silently
    # interpreting the AIC/BIC ranking.
    if {"A", "B"}.issubset(set(result_df.get("model", []))):
        a_rows = result_df[result_df["model"] == "A"]
        b_rows = result_df[result_df["model"] == "B"]
        if (not a_rows.empty and not b_rows.empty
                and a_rows.iloc[0].get("status") == "converged"
                and b_rows.iloc[0].get("status") == "converged"):
            ll_a = float(a_rows.iloc[0]["log_likelihood"])
            ll_b = float(b_rows.iloc[0]["log_likelihood"])
            nesting_note = (
                "OK: B log-likelihood is not below A."
                if ll_b >= ll_a - 1e-4 else
                "WARNING: B log-likelihood is below A despite the intended nested structure; treat B fit and its AIC/BIC as optimisation-sensitive, not as evidence against random slopes."
            )
            result_df["A_vs_B_nesting_diagnostic"] = nesting_note
    return result_df


def write_random_structure_report(result_df: pd.DataFrame, output_dir: Path) -> None:
    """Write human-readable and machine-readable model-structure comparisons."""
    result_df.to_csv(output_dir / "random_effects_structure_comparison.csv", index=False)
    path = output_dir / "random_effects_structure_comparison.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Random-Effects Structure Sensitivity Comparison\n")
        f.write("=" * 48 + "\n\n")
        f.write("All models use the exact same complete-case rows and the same fixed-effects formula:\n")
        f.write("skill_score ~ C(condition) + C(task_type) + C(stage_order) + log_global + interruptions\n")
        f.write("All fits use ML (reml=False).\n\n")
        f.write("A: participant random intercept + nested trial random intercept.\n")
        f.write("B: participant random intercept + independent participant-level mid/no condition-slope variance components + nested trial random intercept (diagnostic random-slope alternative).\n")
        f.write("C: participant random intercept + independent participant-level mid/no condition-slope variance components, without a trial random intercept.\n\n")
        f.write("These are independent variance components, not correlated random intercept/slope coefficients.\n")
        f.write("For each structure, all listed optimizers are attempted; the converged finite-likelihood solution with the highest log-likelihood is retained. Optimizer disagreement and convergence warnings are recorded.\n")
        f.write("AIC/BIC are comparable because the fixed effects, response, and observations are held constant. Lower values indicate better relative fit, not proof that a structure is correct.\n")
        f.write("Model A is the primary inferential random-effects structure. Models B and C are diagnostic alternatives only. Variance-component tests lie on parameter boundaries; do not interpret a conventional chi-square LRT p-value for A vs B or B vs C as a standard test. With only eight participant clusters, treat this as a diagnostic comparison.\n")
        if "A_vs_B_nesting_diagnostic" in result_df.columns and len(result_df):
            f.write(str(result_df["A_vs_B_nesting_diagnostic"].iloc[0]) + "\n\n")
        else:
            f.write("A-vs-B nesting diagnostic unavailable (one or both models failed).\n\n")
        for _, row in result_df.iterrows():
            f.write(f"MODEL {row.get('model')} — {row.get('description')}\n")
            f.write(f"Status: {row.get('status')}\n")
            if row.get("status") == "converged":
                for key in [
                    "optimizer", "attempted_optimizers", "n_rows", "n_subjects", "n_trials",
                    "log_likelihood", "AIC", "BIC", "residual_variance",
                    "participant_intercept_variance", "condition_mid_slope_variance",
                    "condition_no_slope_variance", "trial_intercept_variance",
                    "condition_mid_beta", "condition_mid_se", "condition_mid_p",
                    "condition_no_beta", "condition_no_se", "condition_no_p", "warnings",
                ]:
                    f.write(f"{key}: {row.get(key)}\n")
            else:
                f.write(f"Error: {row.get('error')}\n")
            f.write("\n")
        successful = result_df[result_df.get("status", pd.Series(dtype=str)) == "converged"].copy()
        if len(successful):
            f.write("RANKING AMONG CONVERGED MODELS\n")
            f.write("-------------------------------\n")
            for metric in ["AIC", "BIC"]:
                if metric in successful:
                    ranked = successful.sort_values(metric)
                    f.write(metric + " (lowest first): " + ", ".join(
                        f"{r['model']}={r[metric]:.3f}" for _, r in ranked.iterrows()
                    ) + "\n")
            f.write("\nInterpret the ranking together with convergence, variance estimates, fixed-effect stability, and the scientific need to account for repeated stages within trials. Do not select solely by a tiny AIC/BIC difference.\n")


# ============================================================================
# Variance decomposition and diagnostics
# ============================================================================

def nakagawa_r2(model, df: pd.DataFrame) -> dict[str, float]:
    """Observation-level Nakagawa-style R² for the fitted stage-level model.

    The primary composite model is Model A: participant random intercept plus
    nested trial random intercept. The function also recognises optional
    independent condition-slope variance components so it can be reused for
    diagnostics without assuming a fixed variance-component ordering.
    """
    X = np.asarray(model.model.exog, dtype=float)
    beta = np.asarray(model.fe_params, dtype=float)
    var_fixed = float(np.var(X @ beta))

    G = np.asarray(model.cov_re, dtype=float)
    if G.shape != (1, 1):
        raise RuntimeError(
            "Expected a participant random-intercept covariance matrix of shape (1, 1)."
        )
    var_intercept = float(G[0, 0])

    vc_names = list(getattr(model.model.exog_vc, "names", []))
    vc_values = np.asarray(getattr(model, "vcomp", []), dtype=float)
    if len(vc_names) != len(vc_values):
        raise RuntimeError(
            "Could not map fitted variance components to their names: "
            f"{len(vc_names)} names vs {len(vc_values)} estimates."
        )
    vc = dict(zip(vc_names, vc_values))

    conditions = df["condition"].astype(str)
    p_mid = float((conditions == "mid").mean())
    p_no = float((conditions == "no").mean())
    var_condition_mid = float(vc.get("condition_mid", 0.0))
    var_condition_no = float(vc.get("condition_no", 0.0))
    var_subject = (
        var_intercept
        + p_mid * var_condition_mid
        + p_no * var_condition_no
    )
    var_trial = float(vc.get("trial", 0.0))
    var_random = var_subject + var_trial
    var_resid = float(model.scale)
    total = var_fixed + var_random + var_resid
    if total <= 0:
        return {"marginal_r2": float("nan"), "conditional_r2": float("nan")}
    return {
        "marginal_r2": var_fixed / total,
        "conditional_r2": (var_fixed + var_random) / total,
        "var_fixed": var_fixed,
        "var_subject": float(var_subject),
        "var_subject_intercept": var_intercept,
        "var_condition_mid_slope": var_condition_mid,
        "var_condition_no_slope": var_condition_no,
        "has_condition_slopes": bool("condition_mid" in vc or "condition_no" in vc),
        "var_trial": var_trial,
        "has_trial_intercept": bool("trial" in vc),
        "var_random": var_random,
        "var_resid": var_resid,
        "trial_var_fraction_of_residual": var_trial / var_resid if var_resid > 0 else float("nan"),
        "trial_variance_near_boundary": bool(
            "trial" in vc and var_trial <= max(1e-6, 0.01 * var_resid)
        ),
    }


def residual_diagnostics(model, df: pd.DataFrame) -> dict[str, float]:
    resid = np.asarray(model.resid, dtype=float)
    # Standardised residuals
    sd_resid = float(np.std(resid, ddof=1))
    z = resid / (sd_resid if sd_resid > 0 else 1.0)
    return {
        "n": int(len(resid)),
        "mean": float(np.mean(resid)),
        "sd": sd_resid,
        "max_abs_z": float(np.max(np.abs(z))),
        "n_abs_z_gt_3": int(np.sum(np.abs(z) > 3.0)),
    }


def leave_one_participant_out(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for held_out in sorted(df["subject"].unique()):
        sub = df[df["subject"] != held_out].copy()
        if sub["subject"].nunique() < 2:
            continue
        try:
            m = fit_mixedlm(
                "skill_score ~ C(condition) + C(task_type) + C(stage_order) "
                "+ log_global + interruptions",
                sub,
                f"LOO (without {held_out})",
                include_condition_slopes=False,
            )
            results.append({
                "held_out": held_out,
                "n_rows": len(sub),
                "beta_mid": m.fe_params.get("C(condition)[T.mid]", np.nan),
                "beta_no": m.fe_params.get("C(condition)[T.no]", np.nan),
                "p_mid": m.pvalues.get("C(condition)[T.mid]", np.nan),
                "p_no": m.pvalues.get("C(condition)[T.no]", np.nan),
            })
        except Exception as exc:
            print(f"[WARN] LOO failed for {held_out}: {exc}")

    return pd.DataFrame(results)


# ============================================================================
# Per-metric analysis
# ============================================================================

def fit_per_metric_model(
    df: pd.DataFrame,
    metric: str,
    transform_name: str,
) -> dict[str, Any]:
    """
    Fit a per-metric MixedLM with an appropriate transformation.

    transform_name:
        'log1p'  -> log(1 + x)          (jerk_index)
        'log'    -> log(x) with clip>0  (path_inefficiency_ratio)
        'logit'  -> logit(clip x)       (pause_ratio)
        'negative_binomial' -> count regression for micro_stop_count
        'none'   -> x
    """
    import statsmodels.formula.api as smf

    sub = df.dropna(
        subset=[metric, "condition", "log_global",
                "interruptions", "subject", "task_type", "stage_order"]
    ).copy()

    if len(sub) < 10:
        return {"metric": metric, "error": "insufficient rows"}

    x = pd.to_numeric(sub[metric], errors="coerce")
    if transform_name == "negative_binomial":
        # The observed micro-stop counts are strongly overdispersed, so a
        # Gaussian model of log(count + 1) is not the primary analysis.  Fit
        # an NB2 count model and use subject-clustered standard errors to
        # respect repeated stage observations within participant.
        try:
            m = smf.negativebinomial(
                "micro_stop_count ~ C(condition) + C(task_type) + "
                "C(stage_order) + log_global + interruptions",
                sub,
            ).fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": sub["subject"]},
            )
        except Exception as exc:
            return {"metric": metric, "error": str(exc)}

        from scipy.stats import chi2
        condition_names = [
            name for name in ["C(condition)[T.mid]", "C(condition)[T.no]"]
            if name in m.params.index
        ]
        beta = np.asarray(m.params[condition_names], dtype=float)
        covariance = np.asarray(
            m.cov_params().loc[condition_names, condition_names], dtype=float
        )
        wald = float(beta @ np.linalg.pinv(covariance) @ beta)
        p_condition = float(chi2.sf(wald, len(condition_names)))

        return {
            "metric": metric,
            "model": "negative_binomial (subject-clustered SE)",
            "random_effects_note": (
                "Separate per-metric negative-binomial model with subject-clustered "
                "robust SE only (8 clusters); not equivalent to the composite "
                "skill-score Model A subject-plus-trial hierarchical model."
            ),
            "transform": transform_name,
            "n": int(len(sub)),
            "beta_mid": float(m.params.get("C(condition)[T.mid]", np.nan)),
            "se_mid": float(m.bse.get("C(condition)[T.mid]", np.nan)),
            "p_mid": float(m.pvalues.get("C(condition)[T.mid]", np.nan)),
            "beta_no": float(m.params.get("C(condition)[T.no]", np.nan)),
            "se_no": float(m.bse.get("C(condition)[T.no]", np.nan)),
            "p_no": float(m.pvalues.get("C(condition)[T.no]", np.nan)),
            "lr_condition": wald,
            "p_condition_raw": p_condition,
            "dispersion_alpha": float(m.params.get("alpha", np.nan)),
        }

    if transform_name == "log1p":
        y = np.log1p(x.clip(lower=0))
    elif transform_name == "log":
        y = np.log(x.clip(lower=1e-9))
    elif transform_name == "logit":
        p = x.clip(0.01, 0.99)
        y = np.log(p / (1 - p))
    elif transform_name == "none":
        y = x
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    sub["_y"] = y.values

    try:
        m = fit_mixedlm(
            "_y ~ C(condition) + C(task_type) + C(stage_order) + log_global + interruptions",
            sub,
            f"{metric} full",
            include_condition_slopes=True,
        )
        m0 = fit_mixedlm(
            "_y ~ C(task_type) + C(stage_order) + log_global + interruptions",
            sub,
            f"{metric} reduced",
            include_condition_slopes=True,
        )
    except Exception as exc:
        return {"metric": metric, "error": str(exc)}

    from scipy.stats import chi2
    lr = 2 * (m.llf - m0.llf)
    if lr < 0 and abs(lr) < 1e-8:
        lr = 0.0
    p = float(chi2.sf(lr, 2))

    return {
        "metric": metric,
            "model": "MixedLM: participant intercept + independent condition slopes + nested trial intercept",
        "random_effects_note": (
            "Separate per-metric model: participant random intercept + "
            "independent condition-mid/no variance components + nested trial "
            "random intercept; this is not the composite skill-score Model A."
        ),
        "transform": transform_name,
        "n": int(len(sub)),
        "beta_mid": float(m.fe_params.get("C(condition)[T.mid]", np.nan)),
        "se_mid": float(m.bse_fe.get("C(condition)[T.mid]", np.nan)),
        "p_mid": float(m.pvalues.get("C(condition)[T.mid]", np.nan)),
        "beta_no": float(m.fe_params.get("C(condition)[T.no]", np.nan)),
        "se_no": float(m.bse_fe.get("C(condition)[T.no]", np.nan)),
        "p_no": float(m.pvalues.get("C(condition)[T.no]", np.nan)),
        "lr_condition": float(lr),
        "p_condition_raw": p,
    }


def per_metric_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the per-metric analysis with Benjamini-Hochberg correction
    across the four omnibus condition-effect p-values. Gaussian MixedLM
    metrics use likelihood-ratio tests; micro_stop_count uses a joint Wald
    test from its subject-clustered negative-binomial regression.
    """
    specs = [
        ("jerk_index", "log1p"),
        ("pause_ratio", "logit"),
        ("micro_stop_count", "negative_binomial"),
        ("path_inefficiency_ratio", "log"),
    ]

    rows = []
    for metric, transform in specs:
        if metric not in df.columns:
            print(f"[WARN] metric {metric} not in dataset; skipping.")
            continue
        print(f"[INFO] Per-metric model for {metric} ({transform}).")
        rows.append(fit_per_metric_model(df, metric, transform))

    result = pd.DataFrame(rows)

    if "transform" in result.columns:
        result["condition_test"] = np.where(
            result["transform"] == "negative_binomial",
            "joint Wald test (negative binomial; subject-clustered SE)",
            "likelihood-ratio test (nested MixedLM)",
        )

    if "p_condition_raw" in result.columns:
        # Benjamini-Hochberg
        p = result["p_condition_raw"].values
        n = len(p)
        order = np.argsort(p)
        ranked = np.empty(n)
        prev = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            adj = p[idx] * n / (n - rank + 1)
            prev = min(prev, adj)
            ranked[idx] = prev
        result["p_condition_BH"] = ranked
        result["significant_BH_05"] = result["p_condition_BH"] < 0.05

    return result


# ============================================================================
# Learning-trajectory analysis
# ============================================================================

def predict_learning_grid(
    interaction_model,
    df: pd.DataFrame,
    cycles: list[int] | None = None,
    mean_interruptions: float | None = None,
) -> pd.DataFrame:
    """
    Fixed-effects prediction on a grid of cycles, per condition.
    Returns a long DataFrame with prediction and 95% CI.
    """
    from patsy import dmatrix

    conditions = [
        c for c in CONDITION_ORDER
        if c in set(df["condition"])
    ]
    if cycles is None:
        max_cycle = max(1, int(df["cycle_global"].max()))
        cycles = sorted(set(
            [1, 2, 3, 5, 8, 10, 12, 15, 20]
            + list(range(1, max_cycle + 1))
        ))
        cycles = [c for c in cycles if c <= max_cycle + 2]

    if mean_interruptions is None:
        mean_interruptions = float(df["interruptions"].mean())

    # Average predictions over the observed task/stage distribution.
    # This produces marginal learning curves rather than predictions for an
    # arbitrary reference stage.
    stage_distribution = (
        df[["task_type", "stage_order"]]
        .dropna()
        .value_counts(normalize=True)
        .rename("weight")
        .reset_index()
    )
    rows = []
    for cond in conditions:
        for cycle in cycles:
            for _, stage_row in stage_distribution.iterrows():
                rows.append({
                    "condition": cond,
                    "cycle_global": cycle,
                    "log_global": math.log(cycle),
                    "interruptions": mean_interruptions,
                    "task_type": stage_row["task_type"],
                    "stage_order": stage_row["stage_order"],
                    "weight": stage_row["weight"],
                })
    prediction_rows = pd.DataFrame(rows)

    # See nakagawa_r2: in statsmodels 0.15 DesignInfo belongs to orig_exog.
    design = interaction_model.model.data.orig_exog.design_info
    X = dmatrix(design, prediction_rows, return_type="dataframe")

    coef = np.asarray(interaction_model.fe_params.values, dtype=float)
    # cov_params also contains the random-effect variance parameter.  The
    # fixed-effect prediction uncertainty must use only the fixed-effect block.
    fixed_names = list(interaction_model.fe_params.index)
    cov = np.asarray(
        interaction_model.cov_params().loc[fixed_names, fixed_names],
        dtype=float,
    )

    # Weighted mean design row per condition/cycle, then compute CI for
    # that marginal fixed-effect prediction.
    prediction_rows["_row"] = np.arange(len(prediction_rows))
    X_values = X.values
    output = []
    cursor = 0
    for cond in conditions:
        for cycle in cycles:
            subset = prediction_rows[
                (prediction_rows["condition"] == cond)
                & (prediction_rows["cycle_global"] == cycle)
            ]
            idx = subset["_row"].to_numpy(dtype=int)
            weights = subset["weight"].to_numpy(dtype=float)
            weights = weights / weights.sum()
            x_bar = np.average(X_values[idx], axis=0, weights=weights)
            point = float(x_bar @ coef)
            se = float(np.sqrt(max(0.0, x_bar @ cov @ x_bar)))
            output.append({
                "condition": cond,
                "cycle_global": cycle,
                "log_global": math.log(cycle),
                "interruptions": mean_interruptions,
                "prediction": point,
                "se": se,
                "ci_low": point - 1.96 * se,
                "ci_high": point + 1.96 * se,
            })
    return pd.DataFrame(output)


def linear_vs_log_specification(df: pd.DataFrame) -> dict[str, Any]:
    """
    Fit both specifications (linear cycle_global vs log) and
    report AIC/BIC for comparison.
    """
    sub = df.dropna(
        subset=["skill_score", "condition", "log_global",
                "cycle_global", "interruptions", "subject"]
    ).copy()

    m_log = fit_mixedlm(
        "skill_score ~ C(condition) + C(task_type) + C(stage_order) + log_global + interruptions",
        sub, "logarithmic learning specification",
        include_condition_slopes=False,
    )
    m_lin = fit_mixedlm(
        "skill_score ~ C(condition) + C(task_type) + C(stage_order) + cycle_global + interruptions",
        sub, "linear learning specification",
        include_condition_slopes=False,
    )

    return {
        "log_spec_aic": float(m_log.aic),
        "log_spec_bic": float(m_log.bic),
        "log_spec_llf": float(m_log.llf),
        "lin_spec_aic": float(m_lin.aic),
        "lin_spec_bic": float(m_lin.bic),
        "lin_spec_llf": float(m_lin.llf),
        "preferred_by_aic": "log" if m_log.aic < m_lin.aic else "linear",
        "preferred_by_bic": "log" if m_log.bic < m_lin.bic else "linear",
    }


# ============================================================================
# Interruption-covariate secondary models
# ============================================================================

def fit_covariate_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit three models, each replacing the raw interruption count with
    a categorised component:
        A) n_operator_err
        B) n_false_neg
        C) n_clarification
    and report the condition coefficients.
    """
    results = []
    base_need = [
        "skill_score", "condition", "log_global",
        "interruptions", "subject",
    ]
    for covariate in [
        "n_operator_err", "n_false_neg", "n_clarification",
    ]:
        if covariate not in df.columns:
            print(f"[WARN] {covariate} not in dataset; skipping.")
            continue
        sub = df.dropna(subset=base_need + [covariate]).copy()
        if len(sub) < 10:
            continue
        formula = (
            f"skill_score ~ C(condition) + C(task_type) + C(stage_order) "
            f"+ log_global + {covariate}"
        )
        try:
            m = fit_mixedlm(
                formula, sub, f"{covariate} covariate",
                include_condition_slopes=False,
            )
            results.append({
                "covariate": covariate,
                "n": len(sub),
                "beta_cov": float(m.fe_params.get(covariate, np.nan)),
                "se_cov": float(m.bse_fe.get(covariate, np.nan)),
                "p_cov": float(m.pvalues.get(covariate, np.nan)),
                "beta_mid": float(
                    m.fe_params.get("C(condition)[T.mid]", np.nan)
                ),
                "p_mid": float(
                    m.pvalues.get("C(condition)[T.mid]", np.nan)
                ),
                "beta_no": float(
                    m.fe_params.get("C(condition)[T.no]", np.nan)
                ),
                "p_no": float(
                    m.pvalues.get("C(condition)[T.no]", np.nan)
                ),
            })
        except Exception as exc:
            print(f"[WARN] Covariate model '{covariate}' failed: {exc}")

    return pd.DataFrame(results)


def false_negative_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-condition false-negative rate, using the categorised
    n_false_neg column (propagated to every stage row of the cycle).
    """
    if "n_false_neg" not in df.columns:
        return pd.DataFrame()

    if "trial_uid" not in df.columns:
        raise ValueError(
            "false_negative_rates requires trial_uid to count trials safely."
        )

    # Aggregate per participant--trial, rather than by experiment label,
    # because experiment names need not be globally unique.
    trial = (
        df.groupby(["trial_uid", "condition"], observed=True)
        .agg(
            n_stages=("stage_name", "size"),
            n_false_neg=("n_false_neg", "max"),
        )
        .reset_index()
    )
    by_cond = (
        trial.groupby("condition", observed=True)
        .agg(
            n_trials=("trial_uid", "nunique"),
            n_stages=("n_stages", "sum"),
            n_false_neg=("n_false_neg", "sum"),
        )
        .reset_index()
    )
    by_cond["false_neg_per_100_stages"] = (
        100 * by_cond["n_false_neg"] / by_cond["n_stages"]
    )
    return by_cond


# ============================================================================
# Duration, error, and correlation analyses
# ============================================================================

def analyse_duration(
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Duration LME on log(duration), adjusted for condition, task type, and
    trial progression. A separate sensitivity model adds num_stages.
    Requires at least 2 subjects and 2 conditions.
    """
    trial_all = trial_df.copy()
    if "cycle_global" not in trial_all.columns:
        trial_all = trial_all.sort_values(
            ["subject", "created_at", "experiment"]
        ).reset_index(drop=True)
        trial_all["cycle_global"] = trial_all.groupby("subject").cumcount() + 1
    trial = trial_all.dropna(subset=["total_duration_s"]).copy()
    trial = trial[trial["total_duration_s"] > 0].copy()
    trial["log_duration"] = np.log(trial["total_duration_s"])
    trial["cycle_global"] = pd.to_numeric(
        trial["cycle_global"], errors="coerce"
    )
    trial = trial.dropna(subset=["cycle_global"]).copy()
    trial["log_global"] = np.log(
        trial["cycle_global"].astype(float).clip(lower=1.0)
    )

    # Merge interruptions if available per-trial
    # Interruptions are cycle-level; merge by subject+cycle_global.
    # We rely on trial_df being already merged in main().

    report_lines = [
        "Duration Analysis",
        "=================",
        f"Trials with non-null duration: {len(trial)}",
        f"Subjects: {trial['subject'].nunique()}",
        f"Conditions: {dict(trial['condition'].value_counts())}",
        "",
    ]

    if trial["subject"].nunique() < 2:
        report_lines.append("Insufficient subjects for MixedLM.")
        (output_dir / "duration_model_report.txt").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        return {"n": len(trial), "model": None}

    # Model 1: duration ~ condition + log_global
    try:
        m_dur = fit_mixedlm(
            "log_duration ~ C(condition) + C(task_type) + log_global",
            trial,
            "duration",
            include_trial_intercept=False,
        )
        report_lines.append(
            "Model A: log_duration ~ C(condition) + C(task_type) + "
            "log_global + (1|subject)"
        )
        report_lines.append(str(m_dur.summary()))
        report_lines.append("")
    except Exception as exc:
        report_lines.append(f"Model A failed: {exc}")
        m_dur = None

    # Model 2: with num_stages control
    m_dur_ctrl = None
    try:
        m_dur_ctrl = fit_mixedlm(
            "log_duration ~ C(condition) + C(task_type) + log_global + num_stages",
            trial,
            "duration with stage-count control",
            include_trial_intercept=False,
        )
        report_lines.append(
            "Model B: log_duration ~ C(condition) + C(task_type) + "
            "log_global + num_stages + (1|subject)"
        )
        report_lines.append(str(m_dur_ctrl.summary()))
        report_lines.append("")
        report_lines.append(
            "Interpretation: model B is a sensitivity adjustment for the "
            "observed number of stages. Because stage count may itself be "
            "affected by guidance condition, it is not treated as a primary "
            "causal adjustment."
        )
    except Exception as exc:
        report_lines.append(f"Model B failed: {exc}")

    (output_dir / "duration_model_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return {"n": len(trial), "model": m_dur, "model_controlled": m_dur_ctrl}


def analyse_errors(
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Failed-stage count analysis: Poisson GLM with participant-clustered
    robust SE and NB2 regression sensitivity. Neither is a GLMM.
    """
    import statsmodels.api as sm
    import statsmodels.discrete.discrete_model as sm_dm

    trial = trial_df.dropna(
        subset=["num_failed_unique_stages", "condition", "subject"]
    ).copy()
    trial = trial.sort_values(
        ["subject", "created_at", "experiment"]
    ).reset_index(drop=True)
    trial["cycle_global"] = trial.groupby("subject").cumcount() + 1
    trial["log_global"] = np.log(trial["cycle_global"].astype(float).clip(lower=1.0))

    report_lines = [
        "Failed-Stage Analysis", "=====================",
        f"Trials: {len(trial)}", f"Subjects: {trial['subject'].nunique()}",
        f"Conditions: {dict(trial['condition'].value_counts())}", "",
        "The Poisson analysis is a GLM with participant-clustered robust "
        "standard errors, not a Poisson GLMM. A Poisson GLMM is not fitted "
        "here. With eight participant clusters, cluster-robust inference is "
        "approximate and should be interpreted cautiously.", "",
    ]
    y = trial["num_failed_unique_stages"].astype(float).values
    X = pd.DataFrame(index=trial.index)
    cond_dummies = pd.get_dummies(
        trial["condition"], prefix="cond", drop_first=True
    ).astype(float)
    X = pd.concat([X, cond_dummies], axis=1)
    X["log_global"] = trial["log_global"].values
    X["num_stages"] = pd.to_numeric(trial["num_stages"], errors="coerce").values
    X = sm.add_constant(X, has_constant="add")

    pois = None
    nb = None
    dispersion = float("nan")
    try:
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit(
            cov_type="cluster",
            cov_kwds={"groups": trial["subject"].values},
        )
        dispersion = float(pois.pearson_chi2 / pois.df_resid)
        report_lines += [
            "Poisson GLM with participant-clustered robust SE:",
            str(pois.summary()),
            f"\nPearson dispersion: {dispersion:.3f}.", "",
        ]
    except Exception as exc:
        report_lines.append(f"Poisson GLM failed: {exc}")

    # Always fit NB2 as a distributional sensitivity, not only after a
    # thresholded overdispersion test.
    try:
        with warnings.catch_warnings(record=True) as nb_warnings:
            warnings.simplefilter("always")
            nb = sm_dm.NegativeBinomial(
                y, X, loglike_method="nb2"
            ).fit(
                disp=False, cov_type="cluster",
                cov_kwds={"groups": trial["subject"].values}, maxiter=2000,
            )
        nb_converged = bool(
            getattr(nb, "mle_retvals", {}).get("converged", False)
        )
        report_lines += [
            "NB2 negative-binomial regression sensitivity with "
            "participant-clustered robust SE (not a mixed model):",
            f"Optimizer convergence flag: {nb_converged}",
            "Warnings: " + (
                " | ".join(str(w.message) for w in nb_warnings)
                if nb_warnings else "none"
            ),
            str(nb.summary()), "",
            "NB2 allows overdispersion relative to Poisson. Both models use "
            "the same trial-level predictors. With eight clusters, robust "
            "standard errors remain small-sample sensitive. If convergence "
            "is false or Hessian/standard-error warnings occur, treat NB2 "
            "as an unsuccessful sensitivity fit rather than supporting "
            "evidence.",
        ]
    except Exception as exc:
        nb = None
        report_lines.append(f"NB2 sensitivity model failed: {exc}")

    (output_dir / "error_model_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return {"n": len(trial), "poisson": pois, "negative_binomial": nb,
            "pearson_dispersion": dispersion}


def motion_duration_correlation(
    df: pd.DataFrame,
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Descriptive unadjusted correlations plus a participant-level MixedLM
    testing duration–motion association while controlling for trial
    progression, condition, and task type. Motion quality is averaged within
    trial, giving one observation per trial.
    """
    from scipy.stats import spearmanr, pearsonr

    trial_base = trial_df.copy()
    if "cycle_global" not in trial_base.columns:
        trial_base = trial_base.sort_values(
            ["subject", "created_at", "experiment"]
        ).reset_index(drop=True)
        trial_base["cycle_global"] = (
            trial_base.groupby("subject").cumcount() + 1
        )
    motion = (
        df.groupby(["subject", "experiment"], observed=True)["skill_score"]
        .mean().reset_index().rename(columns={"skill_score": "mean_skill"})
    )
    merged = trial_base.merge(
        motion, on=["subject", "experiment"], how="inner"
    ).copy()
    merged["total_duration_s"] = pd.to_numeric(
        merged["total_duration_s"], errors="coerce"
    )
    merged = merged.dropna(
        subset=["mean_skill", "total_duration_s", "subject", "condition"]
    )
    merged = merged[merged["total_duration_s"] > 0].copy()
    merged = merged.sort_values(
        ["subject", "created_at", "experiment"]
    ).reset_index(drop=True)
    merged["cycle_global"] = pd.to_numeric(
        merged["cycle_global"], errors="coerce"
    )
    merged = merged.dropna(subset=["cycle_global"]).copy()
    merged["log_global"] = np.log(merged["cycle_global"].clip(lower=1))
    merged["log_duration"] = np.log(merged["total_duration_s"])

    report_lines = [
        "Motion–Duration Association", "=" * 27,
        f"Matched trials: {len(merged)}",
        f"Participants: {merged['subject'].nunique()}", "",
        "UNADJUSTED DESCRIPTIVE CORRELATIONS",
        "-----------------------------------",
        "Pooled descriptive correlations only; they do not account for "
        "participant clustering or learning.",
    ]
    if len(merged) >= 3:
        rho, p_rho = spearmanr(merged["mean_skill"], merged["total_duration_s"])
        r, p_r = pearsonr(merged["mean_skill"], merged["total_duration_s"])
        report_lines += [
            f"Unadjusted Spearman rho: {rho:.4f}, p = {p_rho:.4g}",
            f"Unadjusted Pearson r: {r:.4f}, p = {p_r:.4g}", "",
        ]
    else:
        rho = p_rho = r = p_r = float("nan")
        report_lines.append("Insufficient matched trials for correlations.")

    report_lines += [
        "ADJUSTED REPEATED-MEASURES MODEL",
        "--------------------------------",
        "One observation per trial: mean_skill ~ log_duration + log_global "
        "+ C(condition) + C(task_type) + (1 | subject).",
        "The duration coefficient is adjusted for trial progression, "
        "condition, and task type. The participant random intercept accounts "
        "for repeated trials within participants.",
        "Only eight participant clusters are available; interpret "
        "asymptotic p-values and confidence intervals cautiously.",
    ]
    model = None
    adjusted = {k: float("nan") for k in
                ["beta_duration", "se_duration", "ci_low", "ci_high", "p_duration"]}
    try:
        model_data = merged.copy()
        if "task_type" not in model_data:
            model_data["task_type"] = "unknown"
        model_data["condition"] = pd.Categorical(
            model_data["condition"], categories=CONDITION_ORDER
        )
        model_data["task_type"] = model_data["task_type"].astype(str)
        if model_data["task_type"].nunique() < 2 or (
            model_data["task_type"] == "unknown"
        ).all():
            formula = "mean_skill ~ log_duration + log_global + C(condition)"
        else:
            formula = (
                "mean_skill ~ log_duration + log_global + C(condition) "
                "+ C(task_type)"
            )
        model = fit_mixedlm(
            formula, model_data, "motion-duration adjusted",
            include_trial_intercept=False,
        )
        term = "log_duration"
        beta = float(model.fe_params.get(term, np.nan))
        se = float(model.bse_fe.get(term, np.nan))
        pval = float(model.pvalues.get(term, np.nan))
        ci = model.conf_int().loc[term]
        adjusted = {
            "beta_duration": beta, "se_duration": se,
            "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
            "p_duration": pval,
        }
        report_lines += [
            "", f"Adjusted log-duration coefficient: beta={beta:.6f}",
            f"SE={se:.6f}; 95% CI [{ci.iloc[0]:.6f}, {ci.iloc[1]:.6f}]; "
            f"p={pval:.6g}", "", str(model.summary()),
        ]
    except Exception as exc:
        report_lines.append(f"Adjusted MixedLM failed: {exc}")

    report_lines += [
        "",
        "Interpretation: use the adjusted MixedLM as the inferential analysis. "
        "The pooled Spearman/Pearson coefficients above are descriptive only. "
        "A positive duration coefficient means longer duration is associated "
        "with higher mean_skill after adjustment (higher skill_score means "
        "better motion quality).",
    ]
    (output_dir / "motion_duration_correlation.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    merged.to_csv(output_dir / "motion_duration_trial_data.csv", index=False)
    return {
        "n": len(merged), "spearman_rho": float(rho),
        "spearman_p": float(p_rho), "pearson_r": float(r),
        "pearson_p": float(p_r), "adjusted_model": model, **adjusted,
    }


# Questionnaire
# ============================================================================

# ---------------------------------------------------------------------------
# Module-level constants (add near the top of the file, next to other configs)
# ---------------------------------------------------------------------------

def normalize_participant_id(value: Any) -> str:
    """Require the same numeric release ID in metadata and questionnaire tables.

    The private identity crosswalk is deliberately absent from this release.
    Canonicalize CSV representations such as 001 or 1.0 to the JSON ID 1.
    """
    value_text = str(value).strip()
    if not re.fullmatch(r"[0-9]+(?:\.0+)?", value_text):
        raise ValueError(
            "Analysis inputs must use consistent numeric participant IDs. "
            "Prepare de-identified copies before running this release script."
        )
    return str(int(value_text.split(".")[0]))


REFERRED_LABEL_TO_INT = {
    "Never": 1,
    "Rare": 2,
    "Sometimes": 4,
    "Often": 6,
    "Constantly": 7,
}


def analyse_questionnaire(
    questionnaire_csv: Path,
    output_dir: Path,
    trial_conditions_csv: Path | None = None,
) -> None:
    """
    Analyse the wide-format per-trial Likert questionnaire.

    Join numeric questionnaire participant IDs directly to the same numeric
    subjects in interruptions.csv. No identity crosswalk is accepted or stored.
    Trial positions and response-label scoring follow the original analysis.
    """

    if not questionnaire_csv.exists():
        print(
            f"[INFO] Questionnaire CSV not found: {questionnaire_csv}. "
            "Skipping questionnaire analysis."
        )
        return

    from scipy.stats import friedmanchisquare, rankdata, wilcoxon

    q = pd.read_csv(questionnaire_csv, encoding="utf-8-sig", dtype=str)

    # --------------------------------------------------------------
    # 1) Identify the four repeated item column groups
    # --------------------------------------------------------------
    item_labels = ["helpful", "easy", "memory", "referred"]
    item_patterns = [
        r"^The assistance provided during this trial helped me complete",
        r"^\s*The assistance was easy to understand",
        r"^I could perform this trial from memory",
        r"^How often did you refer to the assistance",
    ]

    grouped_cols: dict[str, list[str]] = {label: [] for label in item_labels}
    for label, pattern in zip(item_labels, item_patterns):
        for col in q.columns:
            if re.search(pattern, str(col), flags=re.IGNORECASE):
                grouped_cols[label].append(col)

    lengths = {label: len(cols) for label, cols in grouped_cols.items()}
    print(f"[INFO] Questionnaire trial-column counts: {lengths}")
    if any(v == 0 for v in lengths.values()):
        print(
            "[WARN] At least one item group has zero matching columns. "
            "Check the column labels against the item_patterns."
        )
        return

    min_len = min(lengths.values())
    for label in item_labels:
        grouped_cols[label] = grouped_cols[label][:min_len]
    n_trials = min_len
    print(f"[INFO] Using {n_trials} trials per item.")

    # --------------------------------------------------------------
    # 2) Identify participant ID column
    # --------------------------------------------------------------
    pid_col = None
    for candidate in [
        "Participant ID", "participant_id", "subject", "Subject",
    ]:
        if candidate in q.columns:
            pid_col = candidate
            break
    if pid_col is None:
        print("[WARN] No participant ID column found; aborting.")
        return

    # Use the same numeric release ID across all data sources.
    q["participant_id"] = q[pid_col].map(normalize_participant_id)
    q["subject"] = q["participant_id"]

    # --------------------------------------------------------------
    # 4) Load condition labels from interruptions.csv
    # --------------------------------------------------------------
    # Prefer the CSV next to the questionnaire, then the one next to
    # the trial-summary, then a fallback path.
    candidate_intr_paths = [
        questionnaire_csv.parent / "interruptions.csv",
        Path("experiments/interruptions.csv"),
        Path("data/interruptions.csv"),
    ]
    if trial_conditions_csv is not None:
        candidate_intr_paths.insert(
            0, Path(trial_conditions_csv).parent / "interruptions.csv"
        )

    intr_path = next(
        (p for p in candidate_intr_paths if p.exists()), None
    )

    condition_lookup: dict[tuple[str, int], str] = {}
    if intr_path is not None:
        intr = pd.read_csv(intr_path, dtype=str)
        # Normalise
        intr["subject"] = intr["subject"].map(normalize_participant_id)
        intr["cycle_global"] = pd.to_numeric(
            intr["cycle_global"], errors="coerce"
        )
        if "condition" not in intr.columns:
            print(
                f"[WARN] {intr_path} has no 'condition' column. "
                "Cannot map trials to conditions from this file."
            )
        else:
            intr = intr.dropna(subset=["cycle_global"])
            for _, row in intr.iterrows():
                key = (row["subject"], int(row["cycle_global"]))
                condition_lookup[key] = str(row["condition"]).strip()
            print(
                f"[INFO] Loaded {len(condition_lookup)} (subject, cycle) "
                f"-> condition pairs from {intr_path}."
            )
    else:
        print(
            "[WARN] No interruptions.csv found; every trial will be "
            "marked 'unknown' and the Friedman test will be skipped."
        )

    # --------------------------------------------------------------
    # 5) Reshape to long format, applying both mappings
    # --------------------------------------------------------------
    long_rows: list[dict[str, Any]] = []
    for _, row in q.iterrows():
        subj = row["subject"]
        pid = row["participant_id"]
        for trial_idx in range(n_trials):
            # Map trial position (1-indexed) -> condition
            if subj is not None:
                cond = condition_lookup.get(
                    (str(subj).lower(), trial_idx + 1), "unknown"
                )
            else:
                cond = "unknown"

            # Google Forms supplies the default block (1, 1, 1, "Never")
            # for uncompleted trailing trial fields.  In this dataset those
            # blocks also lack a corresponding trial-condition mapping.  Do
            # not discard a possible genuine all-minimum response from a
            # recorded trial; only remove this sentinel when it is unmapped.
            def _as_int(value: Any) -> int | None:
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None

            raw_helpful = row.get(
                grouped_cols["helpful"][trial_idx], np.nan
            )
            raw_easy = row.get(
                grouped_cols["easy"][trial_idx], np.nan
            )
            raw_memory = row.get(
                grouped_cols["memory"][trial_idx], np.nan
            )
            raw_referred = row.get(
                grouped_cols["referred"][trial_idx], np.nan
            )
            is_padding_block = (
                _as_int(raw_helpful) == 1
                and _as_int(raw_easy) == 1
                and _as_int(raw_memory) == 1
                and str(raw_referred).strip().lower() in ("never", "")
            )
            if cond == "unknown" and is_padding_block:
                continue

            for item_label, cols in grouped_cols.items():
                col = cols[trial_idx]
                raw = row.get(col, np.nan)

                # Referred is a categorical label; all others are numeric.
                if item_label == "referred":
                    label = str(raw).strip() if pd.notna(raw) else ""
                    value = REFERRED_LABEL_TO_INT.get(label, np.nan)
                else:
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        value = np.nan

                if pd.isna(value):
                    continue

                long_rows.append({
                    "participant_id": pid,
                    "subject": subj,
                    "trial_number": trial_idx + 1,
                    "condition": cond,
                    "item": item_label,
                    "value": float(value),
                })

    long_df = pd.DataFrame(long_rows)
    print(f"[INFO] Long-format questionnaire rows: {len(long_df)}")

    # --------------------------------------------------------------
    # 6) Sanity check on the mapping
    # --------------------------------------------------------------
    if not long_df.empty:
        mapping_check = (
            long_df.groupby(["subject", "condition"])
            .size()
            .reset_index(name="n")
        )
        print("[INFO] Rows per (subject, condition):")
        print(mapping_check.to_string(index=False))
        n_unknown = int(
            (long_df["condition"] == "unknown").sum()
        )
        if n_unknown > 0:
            print(
                f"[WARN] {n_unknown} long-format rows have "
                f"condition='unknown'. These will be excluded from "
                f"the Friedman test."
            )

    # --------------------------------------------------------------
    # 7) Descriptive summary (all conditions incl. unknown)
    # --------------------------------------------------------------
    report: list[str] = [
        "Questionnaire Analysis",
        "=" * 24,
        "",
        f"Participants: {q['subject'].nunique(dropna=True)}",
        f"Trials per participant (max): {n_trials}",
        f"Long-format rows: {len(long_df)}",
        "",
    ]

    # Descriptives are restricted to analysed condition mappings.  Memory and
    # referral are meaningful only where assistance exists, so present those
    # items for full and mid guidance only.
    descriptive_df = long_df[
        long_df["condition"].isin(["full", "mid", "no"])
    ].copy()
    descriptive_df = descriptive_df[
        ~(
            descriptive_df["item"].isin(["memory", "referred"])
            & (descriptive_df["condition"] == "no")
        )
    ]
    desc = (
        descriptive_df.groupby(["item", "condition"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    report.append("Descriptives (mean ± SD, n):")
    for _, r in desc.iterrows():
        report.append(
            f"  {r['item']:>10s} | {r['condition']:>7s}: "
            f"{r['mean']:.2f} ± {r['std']:.2f} (n={int(r['count'])})"
        )
    report.append("")

    # --------------------------------------------------------------
    # 8) Friedman tests for items meaningful in all three conditions.
    # --------------------------------------------------------------
    item_label_to_question = {
        "helpful": "Assistance helped complete the task",
        "easy": "Assistance was easy to understand",
        "memory": "Could perform from memory",
        "referred": "How often referred to assistance",
    }

    for item_label in ["helpful", "easy"]:
        sub = long_df[
            (long_df["item"] == item_label)
            & (long_df["condition"].isin(["full", "mid", "no"]))
        ]
        if sub.empty:
            report.append(
                f"Item '{item_label}': no data with known conditions."
            )
            continue

        # One mean per (participant, condition) is the correct unit
        # for a matched Friedman test.
        per_subj_cond = (
            sub.groupby(["subject", "condition"])["value"]
            .mean()
            .unstack(fill_value=np.nan)
            .dropna()
        )
        n = len(per_subj_cond)
        k = 3
        if n < 2:
            report.append(
                f"Item '{item_label}': insufficient participants "
                f"with all 3 conditions (n={n})."
            )
            continue

        stat, p = friedmanchisquare(
            per_subj_cond["full"],
            per_subj_cond["mid"],
            per_subj_cond["no"],
        )
        W = stat / (n * (k - 1)) if n * (k - 1) > 0 else float("nan")
        report.append(
            f"\nItem '{item_label}' ({item_label_to_question[item_label]}):"
        )
        report.append(
            f"  Friedman chi2 = {stat:.3f}, df = {k - 1}, "
            f"p = {p:.4g}, Kendall's W = {W:.3f}, n = {n}"
        )

        # Post-hoc Wilcoxon with Holm
        pairs = [("full", "mid"), ("full", "no"), ("mid", "no")]
        pair_results = []
        for ci, cj in pairs:
            a = per_subj_cond[ci].values
            b = per_subj_cond[cj].values
            if np.allclose(a, b):
                continue
            try:
                s, p_pair = wilcoxon(a, b)
                pair_results.append((ci, cj, s, p_pair))
            except Exception:
                pass

        sorted_pairs = sorted(pair_results, key=lambda x: x[3])
        m = len(sorted_pairs)
        prev = 0.0
        if m:
            report.append("  Post-hoc Wilcoxon (Holm-corrected):")
            for idx, (ci, cj, s, p_raw) in enumerate(sorted_pairs):
                adj = min(1.0, p_raw * (m - idx))
                adj = max(adj, prev)
                prev = adj
                # Rank-biserial correlation: difference between the shares
                # of positive and negative signed ranks.  This is an effect
                # size, unlike z/sqrt(n), which is a standardized test
                # statistic and must not be labelled rank-biserial.
                # Reconstruct the values for the currently reported pair;
                # ``a`` and ``b`` from the preceding loop would otherwise
                # refer only to its final comparison.
                pair_a = per_subj_cond[ci].values
                pair_b = per_subj_cond[cj].values
                differences = pair_a - pair_b
                nonzero = differences != 0
                if np.any(nonzero):
                    ranks = rankdata(np.abs(differences[nonzero]))
                    signed_rank_sum = float(
                        ranks[differences[nonzero] > 0].sum()
                        - ranks[differences[nonzero] < 0].sum()
                    )
                    r_rb = signed_rank_sum / float(ranks.sum())
                else:
                    r_rb = 0.0
                report.append(
                    f"    {ci} vs {cj}: p_raw = {p_raw:.4g}, "
                    f"p_Holm = {adj:.4g}, r_rb = {r_rb:.3f}"
                )

    report.append("")
    report.append(
        "Memory and referral were not tested across all three conditions: "
        "in the no-guidance condition there was no assistance to refer to. "
        "Their descriptive values for full and mid guidance are reported "
        "above; no three-condition Friedman or post-hoc tests were run."
    )

    # --------------------------------------------------------------
    # 9) Debrief frequencies
    # --------------------------------------------------------------
    debrief_cols = [
        "1) Which assistance level best supported your learning of the assembly process?",
        "2. If you were learning this assembly again for the first time, which assistance level would you prefer?",
        "3. Approximately after which trial did you feel confident performing the assembly without relying on the assistance?",
        "4. If you were to perform this assembly every day as part of your job, which assistance level would you prefer?",
        "5. Which assistance level did you find least distracting?",
        "6. Overall, how mentally demanding was the experiment?",
    ]
    report.append("\nDebrief responses (frequencies):")
    for col in debrief_cols:
        match = next(
            (c for c in q.columns if c.strip() == col.strip()), None
        )
        if match is None:
            continue
        vc = q[match].value_counts(dropna=False).to_dict()
        report.append(f"  {col[:80]}")
        for k, v in vc.items():
            report.append(f"    {k}: {v}")
        report.append("")

    (output_dir / "questionnaire_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("[INFO] Wrote questionnaire_report.txt")

# ============================================================================
# Descriptive summaries
# ============================================================================

def write_metric_descriptives(
    df: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
) -> None:
    rows = []
    for condition, group in df.groupby("condition"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "condition": condition,
                "metric": metric,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    pd.DataFrame(rows).to_csv(
        output_dir / "metric_descriptives.csv", index=False
    )


def write_stage_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for (condition, task_type, stage_order, stage_name), group in df.groupby(
        ["condition", "task_type", "stage_order", "stage_name"], dropna=False
    ):
        fs = pd.to_numeric(group["final_score"], errors="coerce").dropna()
        rows.append({
            "condition": condition,
            "task_type": task_type,
            "stage_order": stage_order,
            "stage_name": stage_name,
            "n_trials": int(group["trial_uid"].nunique()),
            "pass_rate": float(group["stage_passed"].mean()),
            "mean_final_score": (
                float(fs.mean()) if not fs.empty else None
            ),
        })
    pd.DataFrame(rows).to_csv(
        output_dir / "stage_summary.csv", index=False
    )


def write_trial_report(
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    path = output_dir / "trial_report.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Trial-Level Summary for Chapter 11\n")
        f.write("==================================\n\n")
        f.write(f"Trials: {len(trial_df)}\n")
        f.write(f"Subjects: {trial_df['subject'].nunique()}\n")
        f.write(
            f"Conditions: {dict(trial_df['condition'].value_counts())}\n\n"
        )
        for cond in CONDITION_ORDER:
            if cond not in set(trial_df["condition"]):
                continue
            g = trial_df[trial_df["condition"] == cond]
            f.write(f"--- Condition: {cond} ---\n")
            f.write(f"  Trials: {len(g)}\n")
            dur = pd.to_numeric(
                g["total_duration_s"], errors="coerce"
            ).dropna()
            if not dur.empty:
                f.write(
                    f"  Duration (s): mean={dur.mean():.2f}, "
                    f"sd={dur.std(ddof=1):.2f}, "
                    f"median={dur.median():.2f}\n"
                )
            f.write(
                f"  Failed decisions per trial: "
                f"mean={g['num_failed_decisions'].mean():.2f}, "
                f"sd={g['num_failed_decisions'].std(ddof=1):.2f}, "
                f"total={int(g['num_failed_decisions'].sum())}\n"
            )
            f.write(
                f"  Failed unique stages per trial: "
                f"mean={g['num_failed_unique_stages'].mean():.2f}, "
                f"sd={g['num_failed_unique_stages'].std(ddof=1):.2f}, "
                f"total={int(g['num_failed_unique_stages'].sum())}\n"
            )
            f.write(
                f"  Stages per trial: mean={g['num_stages'].mean():.2f}\n\n"
            )


# ============================================================================
# Reports — primary model
# ============================================================================

def format_p(v: float) -> str:
    if not math.isfinite(v):
        return "NA"
    if v < 0.001:
        return "<0.001"
    return f"{v:.4f}"


def _coef_ci_table(model, effect_names: list[str]) -> pd.DataFrame:
    rows = []
    ci = model.conf_int()
    for name in effect_names:
        if name in model.fe_params.index:
            rows.append({
                "term": name,
                "beta": float(model.fe_params[name]),
                "se": float(model.bse_fe[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
                "z": float(model.tvalues[name]),
                "p": float(model.pvalues[name]),
            })
    return pd.DataFrame(rows)


def write_lme_report(
    *,
    output_dir: Path,
    df: pd.DataFrame,
    reduced,
    primary,
    interaction,
    condition_test: dict[str, float],
    interaction_test: dict[str, float],
    r2: dict[str, float],
    r2_interaction: dict[str, float],
    task_interaction: dict[str, Any] | None,
    diag: dict[str, float],
    linear_vs_log: dict[str, Any],
    grid: pd.DataFrame,
) -> None:
    path = output_dir / "lme_report.txt"

    with path.open("w", encoding="utf-8") as f:
        f.write("Mixed-Effects Guidance Analysis\n")
        f.write("================================\n\n")

        f.write("DATASET\n-------\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Subjects: {df['subject'].nunique()}\n")
        f.write(f"Experiments: {df['experiment'].nunique()}\n")
        f.write(f"Nested trial IDs: {df['trial_uid'].nunique()}\n")
        f.write(f"Conditions: {dict(df['condition'].value_counts())}\n\n")

        f.write(
            "Skill-score construction: globally standardised "
            "four-metric composite with sign inversion "
            "(higher = better motion quality).\n\n"
        )

        # Variance decomposition
        f.write("VARIANCE DECOMPOSITION (Nakagawa)\n")
        f.write("---------------------------------\n")
        f.write(
            f"Variance of fixed effects: {r2.get('var_fixed', float('nan')):.6f}\n"
        )
        f.write(
            "Participant random-intercept variance: "
            f"{r2.get('var_subject', float('nan')):.6f}\n"
        )
        if r2.get("has_condition_slopes", False):
            f.write(
                "Participant condition-mid slope variance: "
                f"{r2.get('var_condition_mid_slope', float('nan')):.6f}\n"
            )
            f.write(
                "Participant condition-no slope variance: "
                f"{r2.get('var_condition_no_slope', float('nan')):.6f}\n"
            )
        f.write(
            "Trial-within-participant random-intercept variance: "
            f"{r2.get('var_trial', float('nan')):.6f}\n"
        )
        f.write(
            "Total random-effects variance (participant + trial): "
            f"{r2.get('var_random', float('nan')):.6f}\n"
        )
        f.write(
            f"Residual variance:         {r2.get('var_resid', float('nan')):.6f}\n"
        )
        f.write(
            f"Marginal R² (fixed only):  {r2.get('marginal_r2', float('nan')):.4f}\n"
        )
        f.write(
            f"Conditional R² (fixed+re): {r2.get('conditional_r2', float('nan')):.4f}\n\n"
        )

        # Record optimizer selection for the inferential model family.
        f.write("MODEL OPTIMIZATION DIAGNOSTICS\n")
        f.write("-------------------------------\n")
        for model_label, fitted_model in [
            ("Reduced", reduced), ("Primary", primary), ("Interaction", interaction)
        ]:
            candidates = getattr(fitted_model, "_analysis_optimizer_candidates", [])
            f.write(
                f"{model_label}: selected={getattr(fitted_model, '_analysis_optimizer', 'unknown')}; "
                f"converged candidates={candidates}\n"
            )
            fit_warnings = getattr(fitted_model, "_analysis_optimizer_warnings", "")
            if fit_warnings:
                f.write(f"  warnings: {fit_warnings}\n")
        f.write("\n")

        # Diagnostics
        f.write("RESIDUAL DIAGNOSTICS\n")
        f.write("--------------------\n")
        f.write(f"n = {diag['n']}\n")
        f.write(f"mean = {diag['mean']:.6f}\n")
        f.write(f"sd = {diag['sd']:.6f}\n")
        f.write(f"max |z| = {diag['max_abs_z']:.3f}\n")
        f.write(f"count |z| > 3 = {diag['n_abs_z_gt_3']}\n\n")

        # Primary coefficients
        f.write(
            "NOTE: The per-metric analyses are separate secondary models with "
            "their own transformations and random-effects specifications; "
            "micro_stop_count uses a negative-binomial model with subject-clustered "
            "robust standard errors. See per_metric_results.csv.\n\n"
        )

        f.write("PRIMARY MODEL\n")
        f.write("-------------\n")
        f.write(
            "skill_score ~ C(condition) + C(task_type) + C(stage_order) "
            "+ log_global + interruptions + (1 | subject) "
            "+ (1 | subject:trial_uid)\n"
            "(Model A: participant random intercept and nested trial random intercept; "
            "no participant condition slopes.)\n"
            "Stage order is the observed FSM decision position, not a semantic "
            "component identity; task type is controlled separately.\n"
            "The primary model does not include num_stages; this trial-level "
            "quantity is examined in a separate equal-trial-weight sensitivity "
            "model because it may be affected by guidance condition. A trial "
            "random intercept accounts for residual clustering but does not "
            "mathematically make num_stages redundant.\n\n"
        )
        f.write(str(primary.summary()))
        f.write("\n\n")
        f.write("Primary coefficients with 95% CI:\n")
        tbl = _coef_ci_table(
            primary,
            list(primary.fe_params.index),
        )
        f.write(tbl.to_string(index=False))
        f.write("\n\n")

        # LRT
        f.write("LIKELIHOOD-RATIO TESTS (ML)\n")
        f.write("---------------------------\n")
        f.write("Condition effect (primary vs reduced):\n")
        f.write(f"  LR = {condition_test['lr']:.4f}\n")
        f.write(f"  df = {condition_test['df']}\n")
        f.write(f"  p  = {condition_test['p']:.6g}\n\n")
        f.write("Condition × learning interaction (interaction vs primary):\n")
        f.write(f"  LR = {interaction_test['lr']:.4f}\n")
        f.write(f"  df = {interaction_test['df']}\n")
        f.write(f"  p  = {interaction_test['p']:.6g}\n\n")

        # Interaction model coefficients
        f.write("INTERACTION MODEL\n")
        f.write("-----------------\n")
        f.write(
            "skill_score ~ C(condition) + C(task_type) + C(stage_order) "
            "+ log_global + interruptions + C(condition):log_global "
            "+ (1 | subject) + (1 | subject:trial_uid)\n"
            "(Model A random-effects structure.)\n\n"
        )
        f.write(str(interaction.summary()))
        f.write("\n\n")
        f.write("Interaction coefficients with 95% CI:\n")
        tbl = _coef_ci_table(
            interaction,
            list(interaction.fe_params.index),
        )
        f.write(tbl.to_string(index=False))
        f.write("\n\n")
        f.write("Interaction-model variance decomposition (Nakagawa-style):\n")
        f.write(
            f"Marginal R² = {r2_interaction.get('marginal_r2', float('nan')):.4f}; "
            f"Conditional R² = {r2_interaction.get('conditional_r2', float('nan')):.4f}\n"
        )
        f.write(
            "Both R² values include the participant and nested-trial random "
            "intercepts in the conditional component.\n\n"
        )

        # Condition-specific learning slopes and the mid-vs-full slope test.
        f.write("CONDITION-SPECIFIC LEARNING SLOPES\n")
        f.write("----------------------------------\n")
        slope_table = getattr(
            interaction, "_learning_slope_contrasts", pd.DataFrame()
        )
        if not slope_table.empty:
            f.write(slope_table.to_string(index=False))
            f.write("\n")
        else:
            f.write("Learning-slope contrasts unavailable.\n")
        f.write(
            "The joint condition-by-learning LRT above is the primary test of "
            "whether learning slopes differ by condition. Individual slope "
            "tests against zero are secondary contrasts.\n\n"
        )

        # Condition x task interaction sensitivity.
        f.write("CONDITION-BY-TASK INTERACTION SENSITIVITY\n")
        f.write("-----------------------------------------\n")
        if task_interaction is not None:
            t = task_interaction["test"]
            f.write(
                f"LR={t['lr']:.6f}, df={t['df']}, p={t['p']:.6g}\n"
            )
            f.write(str(task_interaction["interaction"].summary()))
        else:
            f.write("Condition-by-task interaction model unavailable.\n")
        f.write("\n")

        # Predicted learning table
        f.write("PREDICTED SKILL SCORE AT SELECTED CYCLES\n")
        f.write("----------------------------------------\n")
        show_cycles = [1, 2, 3, 5, 8, 10, 12, 15, 20]
        for cond in CONDITION_ORDER:
            sub = grid[grid["condition"] == cond]
            sub = sub[sub["cycle_global"].isin(show_cycles)]
            if sub.empty:
                continue
            f.write(f"Condition: {cond}\n")
            for _, row in sub.iterrows():
                f.write(
                    f"  cycle={int(row['cycle_global']):>2}  "
                    f"pred={row['prediction']:+.3f}  "
                    f"95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]\n"
                )
            f.write("\n")

        # Linear vs log
        f.write("LEARNING-FUNCTION SPECIFICATION\n")
        f.write("-------------------------------\n")
        f.write(
            f"Logarithmic: AIC = {linear_vs_log['log_spec_aic']:.3f}, "
            f"BIC = {linear_vs_log['log_spec_bic']:.3f}, "
            f"llf = {linear_vs_log['log_spec_llf']:.3f}\n"
        )
        f.write(
            f"Linear:      AIC = {linear_vs_log['lin_spec_aic']:.3f}, "
            f"BIC = {linear_vs_log['lin_spec_bic']:.3f}, "
            f"llf = {linear_vs_log['lin_spec_llf']:.3f}\n"
        )
        f.write(
            f"Preferred by AIC: {linear_vs_log['preferred_by_aic']}\n"
        )
        f.write(
            f"Preferred by BIC: {linear_vs_log['preferred_by_bic']}\n\n"
        )

        # Interruption covariate note
        f.write("INTERRUPTION COVARIATE\n")
        f.write("----------------------\n")
        if "interruptions" in primary.params.index:
            f.write(
                f"β = {float(primary.params['interruptions']):.6f}, "
                f"SE = {float(primary.bse_fe['interruptions']):.6f}, "
                f"p = {float(primary.pvalues['interruptions']):.6g}\n"
            )
            f.write(
                "Interruptions are cycle-level and propagated to every "
                "stage of the cycle. See interruption_components.csv "
                "for categorised components and covariate_models.csv "
                "for false-negative / operator-error / clarification "
                "sub-models.\n"
            )
        else:
            f.write("(interruptions not in model)\n")


def write_sensitivity_report(
    *,
    output_dir: Path,
    loo_df: pd.DataFrame,
    no_interrupt_coefs: dict[str, Any],
    baseline_coefs: dict[str, Any],
) -> None:
    path = output_dir / "sensitivity_report.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Sensitivity Analyses\n")
        f.write("====================\n\n")
        f.write(
            "Composite skill-score sensitivity refits use Model A: participant "
            "random intercept plus nested trial random intercept. Per-metric "
            "models are reported separately with their own random-effects "
            "specifications in per_metric_results.csv.\n\n"
        )

        f.write("Leave-one-participant-out (LOO)\n")
        f.write("-------------------------------\n")
        if loo_df.empty:
            f.write("(no LOO fits completed)\n\n")
        else:
            f.write(loo_df.to_string(index=False))
            f.write("\n\n")
            for col in ["beta_mid", "beta_no"]:
                v = loo_df[col].dropna()
                if not v.empty:
                    f.write(
                        f"{col}: range [{v.min():+.4f}, {v.max():+.4f}], "
                        f"all same sign = "
                        f"{'yes' if (v > 0).all() or (v < 0).all() else 'no'}\n"
                    )
            f.write("\n")

        f.write("Refit without interrupted trials\n")
        f.write("--------------------------------\n")
        if baseline_coefs and no_interrupt_coefs:
            for name in ["mid", "no"]:
                f.write(
                    f"β_{name}: baseline = {baseline_coefs.get(f'beta_{name}', float('nan')):+.4f}, "
                    f"no-interrupt = {no_interrupt_coefs.get(f'beta_{name}', float('nan')):+.4f}\n"
                )
                f.write(
                    f"p_{name}: baseline = {baseline_coefs.get(f'p_{name}', float('nan')):.4g}, "
                    f"no-interrupt = {no_interrupt_coefs.get(f'p_{name}', float('nan')):.4g}\n"
                )
        else:
            f.write("(refit failed or not performed)\n")


def fit_no_interruptions_model(df: pd.DataFrame) -> dict[str, Any]:
    sub = df[df["interruptions"] == 0].copy()
    sub = sub.dropna(subset=[
        "skill_score", "condition", "log_global", "subject"
    ])
    if sub["subject"].nunique() < 2 or sub["condition"].nunique() < 2:
        return {}
    try:
        m = fit_mixedlm(
            "skill_score ~ C(condition) + C(task_type) + C(stage_order) + log_global",
            sub,
            "no-interruptions sensitivity",
            include_condition_slopes=False,
        )
        return {
            "beta_mid": float(m.fe_params.get("C(condition)[T.mid]", np.nan)),
            "p_mid": float(m.pvalues.get("C(condition)[T.mid]", np.nan)),
            "beta_no": float(m.fe_params.get("C(condition)[T.no]", np.nan)),
            "p_no": float(m.pvalues.get("C(condition)[T.no]", np.nan)),
            "n": len(sub),
        }
    except Exception as exc:
        print(f"[WARN] No-interruptions refit failed: {exc}")
        return {}


# ============================================================================
# Figures
# ============================================================================

def _ensure_fig_dir(output_dir: Path) -> Path:
    fig = output_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    return fig


def plot_forest_condition(
    primary, output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    ci = primary.conf_int()
    names = ["C(condition)[T.mid]", "C(condition)[T.no]"]
    labels = ["Mid vs Full", "No vs Full"]
    betas, los, his = [], [], []
    for n in names:
        if n in primary.fe_params.index:
            betas.append(float(primary.fe_params[n]))
            los.append(float(ci.loc[n, 0]))
            his.append(float(ci.loc[n, 1]))

    if not betas:
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(betas))
    ax.errorbar(
        betas, y,
        xerr=[np.array(betas) - np.array(los),
              np.array(his) - np.array(betas)],
        fmt="o", capsize=4, color="C0",
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Difference in skill score (vs full guidance)")
    ax.set_title("Primary condition effect with 95% CI")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(output_dir) / "fig_forest_condition.png", dpi=160)
    plt.close(fig)


def plot_learning_curves(
    grid: pd.DataFrame,
    output_dir: Path,
    mean_interruptions: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in CONDITION_ORDER:
        sub = grid[grid["condition"] == cond]
        if sub.empty:
            continue
        ax.plot(
            sub["cycle_global"], sub["prediction"],
            marker="o", label=cond,
        )
        ax.fill_between(
            sub["cycle_global"], sub["ci_low"], sub["ci_high"],
            alpha=0.20,
        )
    ax.set_xlabel("Global cycle")
    ax.set_ylabel("Predicted skill score")
    ax.set_title(
        "Learning-adjusted guidance curves "
        f"(interruptions = {mean_interruptions:.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(output_dir) / "fig_learning_curves.png", dpi=160)
    plt.close(fig)


def plot_per_metric(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        "jerk_index", "pause_ratio",
        "micro_stop_count", "path_inefficiency_ratio",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means, errs, conds = [], [], []
        for cond in CONDITION_ORDER:
            sub = df[df["condition"] == cond][metric].dropna()
            if sub.empty:
                continue
            conds.append(cond)
            means.append(float(sub.mean()))
            errs.append(float(sub.std(ddof=1) / math.sqrt(len(sub))))
        ax.bar(conds, means, yerr=errs, capsize=3)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(output_dir) / "fig_per_metric.png", dpi=160)
    plt.close(fig)


def plot_participant_boxplot(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    subjects = sorted(df["subject"].unique())
    if not subjects:
        return

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(subjects)), 4))
    positions = []
    data = []
    colors = []
    labels = []
    cmap = {"full": "C0", "mid": "C1", "no": "C2"}
    pos = 1.0
    for subj in subjects:
        for cond in CONDITION_ORDER:
            g = df[(df["subject"] == subj) & (df["condition"] == cond)]
            if g.empty:
                continue
            data.append(g["skill_score"].dropna().values)
            positions.append(pos)
            colors.append(cmap[cond])
            labels.append("")
            pos += 0.8
        pos += 0.6
    if not data:
        return
    bp = ax.boxplot(data, positions=positions, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks([p for p, l in zip(positions, labels) if l])
    ax.set_ylabel("Skill score")
    ax.set_title("Participant-level skill score by condition")
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(output_dir) / "fig_participant_boxplot.png", dpi=160)
    plt.close(fig)


def plot_duration_by_condition(
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    trial = trial_df.dropna(subset=["total_duration_s"]).copy()
    if trial.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))
    data = []
    labels = []
    for cond in CONDITION_ORDER:
        sub = trial[trial["condition"] == cond]["total_duration_s"]
        if sub.empty:
            continue
        data.append(sub.values)
        labels.append(cond)
    if not data:
        return
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Task completion time (s)")
    # Raw seconds are plotted; inferential duration models use log_duration.
    ax.set_title("Task completion time by condition (raw seconds)")
    fig.tight_layout()
    fig.savefig(_ensure_fig_dir(output_dir) / "fig_duration.png", dpi=160)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 11 statistical analysis (final Model A)."
    )
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--output-dir", default="results/guidance_stats_v2")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--condition-map", action="append", default=[])
    parser.add_argument("--include-unknown-condition", action="store_true")
    parser.add_argument(
        "--interruptions-csv",
        default="experiments/interruptions.csv",
    )
    parser.add_argument(
        "--questionnaire-csv",
        default="experiments/questionnaire.csv",
        help="Optional wide-format per-trial questionnaire with numeric Participant ID "
             "matching the metadata and interruptions subject IDs. See docs/data_contract.md.",
    )
    args = parser.parse_args()

    print()
    print("=" * 80)
    print("CHAPTER 11 GUIDANCE ANALYSIS — FINAL MODEL A")
    print("=" * 80)
    print(f"[INFO] Experiments root: {args.experiments_root}")
    print(f"[INFO] Output directory: {args.output_dir}")

    metrics = [
        m.strip() for m in str(args.metrics).split(",") if m.strip()
    ]
    print(f"[INFO] Motion metrics: {metrics}")

    try:
        import statsmodels
        import scipy
        print(f"[INFO] statsmodels {statsmodels.__version__}, "
              f"scipy {scipy.__version__}")
    except ImportError as exc:
        raise RuntimeError(
            "Requires statsmodels and scipy. "
            "Install: pip install statsmodels scipy"
        ) from exc

    condition_map = parse_condition_map(args.condition_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Trial-level data
    # ------------------------------------------------------------------
    print("\n[STEP 1] Building trial summary...")
    trial_df = build_trial_summary(
        experiments_root=Path(args.experiments_root),
        condition_map=condition_map,
        include_unknown_condition=bool(args.include_unknown_condition),
    )

    # Merge interruptions at trial level too (for duration/error control)
    intr_path = Path(args.interruptions_csv) if args.interruptions_csv else None
    if intr_path and intr_path.exists():
        # Build a trial-level interruptions frame directly from CSV
        intr = pd.read_csv(intr_path, dtype={"subject": str})
        intr["subject"] = intr["subject"].map(normalize_participant_id)
        intr["cycle_global"] = intr["cycle_global"].astype(int)
        intr["interruptions"] = pd.to_numeric(
            intr["interruptions"], errors="coerce"
        )
        trial_df["subject_key"] = (
            trial_df["subject"].astype(str).str.strip().str.lower()
        )
        trial_df = trial_df.sort_values(
            ["subject", "created_at", "experiment"]
        ).reset_index(drop=True)
        trial_df["cycle_global"] = trial_df.groupby("subject").cumcount() + 1
        trial_df = trial_df.merge(
            intr.rename(columns={"subject": "subject_key"})[
                ["subject_key", "cycle_global", "interruptions"]
            ],
            on=["subject_key", "cycle_global"],
            how="left",
        )
        trial_df["interruptions"] = (
            trial_df["interruptions"].fillna(0).astype(float)
        )
        trial_df = trial_df.drop(columns=["subject_key"])

    # Assign within-participant cycle order regardless of whether an
    # interruption file was supplied; this is the common learning index.
    trial_df = trial_df.sort_values(
        ["subject", "created_at", "experiment"]
    ).reset_index(drop=True)
    trial_df["cycle_global"] = trial_df.groupby("subject").cumcount() + 1
    trial_df["log_global"] = np.log(trial_df["cycle_global"].clip(lower=1))
    write_cycle_participation(trial_df, output_dir)
    write_observed_stopping_summary(trial_df, output_dir)
    trial_df.to_csv(output_dir / "trial_summary.csv", index=False)
    write_trial_report(trial_df, output_dir)

    # ------------------------------------------------------------------
    # Step 2: Stage-level data
    # ------------------------------------------------------------------
    print("\n[STEP 2] Building stage-level dataset...")
    df = build_dataset(
        experiments_root=Path(args.experiments_root),
        condition_map=condition_map,
        metrics=metrics,
        include_unknown_condition=bool(args.include_unknown_condition),
    )
    print(f"[INFO] Raw stage rows: {len(df)}")

    metric_cols = [m for m in metrics if m in df.columns]
    before = len(df)
    df = df.dropna(subset=metric_cols, how="any").copy()
    print(f"[INFO] Dropped {before - len(df)} rows with missing metrics; "
          f"{len(df)} retained.")

    # ------------------------------------------------------------------
    # Step 3: Interruptions (categorised)
    # ------------------------------------------------------------------
    print("\n[STEP 3] Merging categorised interruption data...")
    if intr_path and intr_path.exists():
        df = merge_interruptions(df, intr_path)
        print(f"[INFO] Mean interruptions per stage: {df['interruptions'].mean():.4f}")
        for col in ["n_false_neg", "n_operator_err", "n_clarification"]:
            if col in df.columns:
                print(f"        {col} mean: {df[col].mean():.4f}")
    else:
        df["interruptions"] = 0.0
        df["n_false_neg"] = 0.0
        df["n_operator_err"] = 0.0
        df["n_clarification"] = 0.0

    # ------------------------------------------------------------------
    # Step 4: Skill score
    # ------------------------------------------------------------------
    print("\n[STEP 4] Computing composite skill score...")
    df = add_motion_skill_score(df, metrics)

    # Explicit categorical variables for the stage-level model.
    # The two tasks have different named stages; stage_order is therefore
    # modelled categorically within task_type (task_type * stage_order).
    df["task_type"] = df["task_type"].astype(str).str.strip().str.lower()
    invalid_task_types = sorted(set(df["task_type"]) - {"assembly", "dismantle"})
    if invalid_task_types:
        raise ValueError(f"Unsupported task_type values: {invalid_task_types}")
    df["task_type"] = pd.Categorical(
        df["task_type"], categories=["assembly", "dismantle"], ordered=False
    )
    df["stage_order"] = pd.to_numeric(df["stage_order"], errors="coerce")
    df["stage_order"] = pd.Categorical(
        df["stage_order"], categories=sorted(df["stage_order"].dropna().unique()),
        ordered=False
    )

    # Explicit categorical reference level for condition
    df["condition"] = pd.Categorical(
        df["condition"],
        categories=CONDITION_ORDER,
        ordered=False,
    )

    # Drop any remaining invalid modelling rows
    before = len(df)
    df = df.dropna(
        subset=[
            "skill_score", "condition", "log_global", "subject",
            "task_type", "stage_order",
        ]
    ).copy()
    if before != len(df):
        print(f"[INFO] Dropped {before - len(df)} rows with missing "
              "modelling variables.")

    # Validate the nested participant--trial structure before fitting any
    # stage-level Gaussian mixed model.  This deliberately fails loudly if a
    # folder, subject, or condition was merged inconsistently.
    df, trial_structure = validate_and_add_trial_uid(df)
    print(
        "[INFO] Validated nested trial structure: "
        f"{trial_structure['stage_rows']} stage rows, "
        f"{trial_structure['subjects']} subjects, "
        f"{trial_structure['trial_uids']} participant--trial combinations."
    )
    (output_dir / "trial_structure_report.txt").write_text(
        "Nested Trial Structure Validation\n"
        "===============================\n\n"
        "trial_uid = normalized subject + normalized experiment_path\n"
        "All trial_uids map to exactly one subject and one condition.\n"
        "All participant--trial folders map to exactly one trial_uid.\n"
        f"Stage rows: {trial_structure['stage_rows']}\n"
        f"Subjects: {trial_structure['subjects']}\n"
        f"Participant--trial combinations: {trial_structure['trial_uids']}\n"
        f"Expected valid trials: {EXPECTED_VALID_TRIALS}\n",
        encoding="utf-8",
    )

    df.to_csv(output_dir / "analysis_dataset.csv", index=False)
    write_participant_summary(df, trial_df, output_dir)

    # Descriptive outputs
    write_metric_descriptives(df, metrics, output_dir)
    write_stage_summary(df, output_dir)

    # ------------------------------------------------------------------
    # Step 5: Mixed-effects models
    # ------------------------------------------------------------------
    print("\n[STEP 5] Fitting MixedLM models...")
    (reduced, primary, interaction,
     condition_test, interaction_test, df_model) = fit_models(df)
    getattr(
        interaction, "_learning_slope_contrasts", pd.DataFrame()
    ).to_csv(output_dir / "learning_slope_contrasts.csv", index=False)

    # Diagnostic comparison of random-effects structures A/B/C. This uses
    # df_model exactly as constructed by fit_models and does not alter any
    # downstream analysis, questionnaire processing, figures, or metadata.
    print("\n[STEP 5A] Comparing random-effects structures A/B/C...")
    structure_comparison = compare_random_effects_structures(df_model)
    write_random_structure_report(structure_comparison, output_dir)

    # Variance decomposition for both primary and learning-interaction models.
    r2 = nakagawa_r2(primary, df_model)
    r2_interaction = nakagawa_r2(interaction, df_model)

    # Trial-level sensitivity for unequal observed stage counts. Each trial
    # contributes one aggregated motion-quality observation.
    print("\n[STEP 5B] Trial-level num_stages sensitivity...")
    num_stages_sensitivity = fit_num_stages_sensitivity(
        df_model, trial_df, output_dir
    )

    # Check whether the guidance-condition effect differs between assembly
    # and dismantling. This is a secondary interaction sensitivity.
    print("\n[STEP 5C] Condition-by-task interaction sensitivity...")
    try:
        task_interaction = fit_condition_task_interaction(df_model)
    except Exception as exc:
        print(f"[WARN] Condition-by-task interaction failed: {exc}")
        task_interaction = None

    # Export extreme conditional residuals for data-quality/model diagnostics.
    # This is diagnostic only: it is computed after fitting and never changes
    # the fitted model or any inferential result.
    resid = np.asarray(primary.resid, dtype=float)
    sd_resid = float(np.std(resid, ddof=1))
    if math.isfinite(sd_resid) and sd_resid > 0:
        z_scores = resid / sd_resid
        outlier_positions = np.flatnonzero(
            np.isfinite(z_scores) & (np.abs(z_scores) > 3.0)
        )
    else:
        z_scores = np.full(len(resid), np.nan)
        outlier_positions = np.array([], dtype=int)
        print("[WARN] Residual SD is zero or invalid; no residual outliers saved.")

    outliers_df = df_model.iloc[outlier_positions].copy()
    outliers_df["z_resid"] = z_scores[outlier_positions]
    outliers_df[
        ["subject", "condition", "task_type", "stage_order",
         "stage_name", "skill_score", "z_resid"]
    ].to_csv(output_dir / "residual_outliers.csv", index=False)
    print(
        f"[INFO] Saved {len(outliers_df)} residual outliers "
        "to residual_outliers.csv"
    )

    diag = residual_diagnostics(primary, df_model)

    # Learning prediction grid
    grid = predict_learning_grid(interaction, df_model)

    # Linear vs log specification
    lin_vs_log = linear_vs_log_specification(df_model)

    write_lme_report(
        output_dir=output_dir,
        df=df_model,
        reduced=reduced,
        primary=primary,
        interaction=interaction,
        condition_test=condition_test,
        interaction_test=interaction_test,
        r2=r2,
        r2_interaction=r2_interaction,
        task_interaction=task_interaction,
        diag=diag,
        linear_vs_log=lin_vs_log,
        grid=grid,
    )
    grid.to_csv(output_dir / "predicted_learning_table.csv", index=False)
    write_model_formulas(output_dir)

    # ------------------------------------------------------------------
    # Step 6: Per-metric analysis
    # ------------------------------------------------------------------
    print("\n[STEP 6] Per-metric analysis...")
    per_metric_df = per_metric_analysis(df_model)
    per_metric_df.to_csv(
        output_dir / "per_metric_results.csv", index=False
    )

    # ------------------------------------------------------------------
    # Step 7: Interruption covariate models
    # ------------------------------------------------------------------
    print("\n[STEP 7] Interruption covariate models...")
    covariate_df = fit_covariate_models(df_model)
    covariate_df.to_csv(
        output_dir / "covariate_models.csv", index=False
    )
    fn_rates = false_negative_rates(df_model)
    if not fn_rates.empty:
        fn_rates.to_csv(
            output_dir / "system_false_negative_rates.csv", index=False
        )

    # ------------------------------------------------------------------
    # Step 8: Sensitivity
    # ------------------------------------------------------------------
    print("\n[STEP 8] Sensitivity analyses...")
    loo_df = leave_one_participant_out(df_model)
    loo_df.to_csv(output_dir / "sensitivity_loo.csv", index=False)

    baseline_coefs = {
        "beta_mid": float(primary.fe_params.get("C(condition)[T.mid]", np.nan)),
        "p_mid": float(primary.pvalues.get("C(condition)[T.mid]", np.nan)),
        "beta_no": float(primary.fe_params.get("C(condition)[T.no]", np.nan)),
        "p_no": float(primary.pvalues.get("C(condition)[T.no]", np.nan)),
    }
    no_intr_coefs = fit_no_interruptions_model(df_model)

    write_sensitivity_report(
        output_dir=output_dir,
        loo_df=loo_df,
        no_interrupt_coefs=no_intr_coefs,
        baseline_coefs=baseline_coefs,
    )

    # ------------------------------------------------------------------
    # Step 9: Duration and error analyses
    # ------------------------------------------------------------------
    print("\n[STEP 9] Duration and error analyses...")
    try:
        duration_result = analyse_duration(trial_df, output_dir)
        print(f"[INFO] Duration analysis N = {duration_result.get('n', 0)}")
    except Exception as exc:
        print(f"[WARN] Duration analysis failed: {exc}")

    try:
        error_result = analyse_errors(trial_df, output_dir)
        print(f"[INFO] Error analysis N = {error_result.get('n', 0)}")
    except Exception as exc:
        print(f"[WARN] Error analysis failed: {exc}")

    try:
        corr = motion_duration_correlation(df, trial_df, output_dir)
        if corr:
            print(
                f"[INFO] Unadjusted motion–duration Spearman rho = "
                f"{corr['spearman_rho']:.3f}, p = {corr['spearman_p']:.4g}"
            )
            print(
                f"[INFO] Adjusted motion–duration MixedLM beta = "
                f"{corr.get('beta_duration', float('nan')):.4f}, "
                f"p = {corr.get('p_duration', float('nan')):.4g}"
            )
    except Exception as exc:
        print(f"[WARN] Motion–duration correlation failed: {exc}")

    # ------------------------------------------------------------------
    # Step 10: Questionnaire (optional)
    # ------------------------------------------------------------------
    if args.questionnaire_csv:
        analyse_questionnaire(
            Path(args.questionnaire_csv), output_dir
        )

    # ------------------------------------------------------------------
    # Step 11: Figures
    # ------------------------------------------------------------------
    print("\n[STEP 11] Generating figures...")
    try:
        plot_forest_condition(primary, output_dir)
        plot_learning_curves(
            grid, output_dir,
            mean_interruptions=float(df_model["interruptions"].mean()),
        )
        plot_per_metric(df_model, output_dir)
        plot_participant_boxplot(df_model, output_dir)
        plot_duration_by_condition(trial_df, output_dir)
    except Exception as exc:
        print(f"[WARN] Figure generation failed: {exc}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Trials: {trial_df['experiment'].nunique()}")
    print(f"Subjects: {trial_df['subject'].nunique()}")
    print(f"Stage rows in MixedLM: {len(df_model)}")
    print(f"Conditions: {dict(df_model['condition'].value_counts())}")
    print()
    print("Primary condition LRT:")
    print(f"  LR = {condition_test['lr']:.4f}, "
          f"df = {condition_test['df']}, "
          f"p = {format_p(condition_test['p'])}")
    print("Learning interaction LRT:")
    print(f"  LR = {interaction_test['lr']:.4f}, "
          f"df = {interaction_test['df']}, "
          f"p = {format_p(interaction_test['p'])}")
    print()
    print(f"Marginal R²: {r2.get('marginal_r2', float('nan')):.3f}")
    print(f"Conditional R²: {r2.get('conditional_r2', float('nan')):.3f}")
    print(f"Participant random-intercept variance: "
          f"{r2.get('var_subject', float('nan')):.6f}")
    print(f"Trial random-intercept variance: "
          f"{r2.get('var_trial', float('nan')):.6f}")
    print(f"Total random-effects variance (participant + trial): "
          f"{r2.get('var_random', float('nan')):.6f}")
    print(f"Residual variance: {r2.get('var_resid', float('nan')):.6f}")
    print()
    print("Outputs written to:", output_dir)
    for fname in [
        "trial_summary.csv", "trial_report.txt",
        "analysis_dataset.csv", "trial_structure_report.txt",
        "cycle_participation.csv", "participant_summary.csv",
        "participant_stopping_summary.csv", "stopping_criterion_note.txt",
        "model_formulas.txt", "learning_slope_contrasts.csv",
        "num_stages_sensitivity.txt",
        "num_stages_sensitivity_data.csv", "motion_duration_trial_data.csv",
        "metric_descriptives.csv",
        "stage_summary.csv", "lme_report.txt",
        "random_effects_structure_comparison.txt",
        "random_effects_structure_comparison.csv",
        "residual_outliers.csv",
        "per_metric_results.csv", "predicted_learning_table.csv",
        "covariate_models.csv", "system_false_negative_rates.csv",
        "sensitivity_loo.csv", "sensitivity_report.txt",
        "duration_model_report.txt", "error_model_report.txt",
        "motion_duration_correlation.txt",
        "questionnaire_report.txt",
    ]:
        p = output_dir / fname
        if p.exists():
            print(f"  {p}")
    print(f"  {output_dir / 'figures'}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
