"""Verify de-identified inputs join without a private identity crosswalk."""

import json

import pandas as pd
import pytest

from scripts import guidance_stats_final_analysis_model_B as analysis


def test_false_negative_rates_count_only_observed_trial_conditions():
    frame = pd.DataFrame({
        'trial_uid': ['1', '1', '2', '3'],
        'condition': pd.Categorical(['full', 'full', 'mid', 'no'],
                                    categories=['full', 'mid', 'no']),
        'stage_name': ['A01', 'A02', 'A01', 'A01'],
        'n_false_neg': [1, 1, 0, 0],
    })
    rates = analysis.false_negative_rates(frame).set_index('condition')
    assert rates['n_trials'].to_dict() == {'full': 1, 'mid': 1, 'no': 1}
    assert rates.loc['full', 'n_stages'] == 2
    assert rates.loc['full', 'false_neg_per_100_stages'] == 50


def test_numeric_metadata_and_intervention_ids_join(tmp_path):
    trial = tmp_path / "trial_1"
    trial.mkdir()
    (trial / "metadata.json").write_text(json.dumps({
        "experiment_name": "trial_1",
        "session": {"user_id": "001", "trial_id": "1"},
        "created_at": "2000-01-01T00:00:00",
        "system_config": {"guidance_mode": "full_guidance_mode", "operation_mode": "assembly"},
        "fsm_decisions": [{
            "stage": "A01_place_bottom_tie_rod_nuts",
            "feedback": {"stage_passed": True, "fatigue_metrics": {
                metric: 1.0 for metric in analysis.DEFAULT_METRICS
            }},
        }],
    }))
    frame = analysis.build_dataset(
        experiments_root=tmp_path,
        condition_map={},
        metrics=analysis.DEFAULT_METRICS,
        include_unknown_condition=False,
    )
    assert frame["subject"].tolist() == ["1"]
    frame["cycle_global"] = 1
    interruptions = tmp_path / "interruptions.csv"
    pd.DataFrame([{"subject": "1.0", "cycle_global": 1, "interruptions": 2}]).to_csv(
        interruptions, index=False,
    )
    merged = analysis.merge_interruptions(frame, interruptions)
    assert merged["interruptions"].tolist() == [2.0]


def test_questionnaire_numeric_ids_match_conditions(tmp_path):
    questions = [
        "The assistance provided during this trial helped me complete the task.",
        "The assistance was easy to understand and follow.",
        "I could perform this trial from memory without relying on the assistance.",
        "How often did you refer to the assistance during this trial?",
    ]
    responses = []
    interventions = []
    for participant in (1, 2):
        response = {"Participant ID": f"{participant:03d}"}
        for cycle, condition in enumerate(("full", "mid", "no"), start=1):
            suffix = "" if cycle == 1 else f".{cycle - 1}"
            for question in questions:
                response[question + suffix] = (
                    ["Often", "Sometimes", "Never"][cycle - 1]
                    if question.startswith("How often") else 8 - cycle - participant
                )
            interventions.append({
                "subject": f"{participant}.0", "cycle_global": cycle,
                "condition": condition, "interruptions": 0,
            })
        responses.append(response)
    csv_path = tmp_path / "questionnaire.csv"
    pd.DataFrame(responses).to_csv(csv_path, index=False)
    pd.DataFrame(interventions).to_csv(tmp_path / "interruptions.csv", index=False)
    output = tmp_path / "output"
    output.mkdir()
    analysis.analyse_questionnaire(csv_path, output)
    report = (output / "questionnaire_report.txt").read_text()
    assert "Participants: 2" in report
    assert "Long-format rows: 24" in report
    assert "Friedman" in report


@pytest.mark.parametrize("value", ["operator_name", "", None, "nan", "1.5", "1e3"])
def test_identifying_or_invalid_subject_ids_are_rejected(value):
    with pytest.raises(ValueError, match="numeric participant IDs"):
        analysis.normalize_participant_id(value)
