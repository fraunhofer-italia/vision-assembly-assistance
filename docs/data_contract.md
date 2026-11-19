# Input formats

Download datasets and weights from the [Hugging Face repositories](artifacts.md). Study observations are stored separately from the code.

## Trial metadata

Each trial directory contains `metadata.json` with `experiment_name`, `session`, `created_at`, `system_config` and `fsm_decisions`. The released analysis uses 70 valid trials.

Use the same numeric participant ID in JSON `session.user_id`, intervention CSV `subject` and questionnaire `Participant ID`. Values such as `001`, `1` and `1.0` resolve to `1`. Chronological order determines the global cycle index; preserve it when preparing inputs.

## Interventions

`interruptions.csv` requires `subject`, `cycle_global` and `interruptions`. Conditions are `full`, `mid`, `no`. Use one row per participant/cycle. `condition`, `interruptions_severity` and `notes_raw` provide condition mapping and event categories. Unmatched counts default to zero.

## Questionnaire

The CSV uses numeric `Participant ID` and repeated blocks of four columns, in trial order:

1. `The assistance provided during this trial helped me complete the task.`
2. `The assistance was easy to understand and follow.`
3. `I could perform this trial from memory without relying on the assistance.`
4. `How often did you refer to the assistance during this trial?`

Keep response order and scoring. pandas adds suffixes such as `.1`, `.2` to repeated headers. Per-cycle conditions come from `interruptions.csv`.

## Vision inputs

Pass video, calibration and model paths through the relevant CLI options. RF-DETR requires its checkpoint and the matching COCO annotations/class map. Pose classifiers use part crops and the class mapping supplied with each model. Numerical DTW references are provided with the study dataset.
