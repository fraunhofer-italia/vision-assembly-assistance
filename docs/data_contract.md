# Inputs for the release analysis script

No participant observations are distributed in this code repository. Prepare cleared copies outside the repository and mount/copy them into the local `experiments/` directory when needed.

## IDs and joins

Use the same numeric participant ID in JSON `session.user_id`, CSV `subject` and questionnaire `Participant ID`. The analysis canonicalizes `001`, `1` and `1.0` to `1`. Participant names, arbitrary labels, missing IDs and noninteger values are rejected. Do not publish an identity crosswalk. Assign IDs before creating release filenames and outputs; numeric IDs alone do not remove identifying content from free text or video.

Each trial directory needs `metadata.json`, containing a nonidentifying `experiment_name`, `session` identifiers, relative or appropriately cleared `created_at` chronology, `system_config.guidance_mode` and `operation_mode`, and `fsm_decisions`. Each decision carries its stage, feedback/motion metrics, segment times and FSM transitions as in the existing metadata schema. Chronological order must be preserved because global trial/cycle numbers are derived by sorting participant records. Replacing dates must not change that order.

The original script validates 70 valid participant trials for this paper. Preserve the intended exclusions and document missing metrics; an arbitrary new study is not automatically compatible with that guard.

## Interruptions

`experiments/interruptions.csv` requires `subject`, `cycle_global`, `interruptions`; `condition`, `interruptions_severity` and `notes_raw` support questionnaire mapping and interruption categorization. Conditions are `full`, `mid`, `no`. Use one row per participant/cycle. Sanitize notes without deleting scientifically relevant event codes. The historical script defaults unmatched counts to zero and uses notes to categorize event types; that behavior is preserved, so missing rows must be resolved/documented before scientific reuse.

## Questionnaire

The final script expects a **wide-format** questionnaire with numeric `Participant ID` and repeated blocks of these four columns in chronological trial order:

1. `The assistance provided during this trial helped me complete the task.`
2. `The assistance was easy to understand and follow.`
3. `I could perform this trial from memory without relying on the assistance.`
4. `How often did you refer to the assistance during this trial?`

Repeated column headers acquire pandas suffixes such as `.1`, `.2`. Keep the original item ordering/scoring, six debrief questions when available and the matching per-cycle condition lookup in `interruptions.csv`. The script is not a long-format questionnaire loader. Uncompleted form blocks must retain their documented handling; do not convert genuine minimum responses to missing data.

Do not include participant-name fields, submission dates, unused health/lifestyle responses or the original form-to-name mapping in public data. No data sanitization was performed by preparing this code copy. Review consent and the final data export separately, then regenerate cleared results with numeric labels.

## Vision assets

Reference/observed videos, homography arrays, CAD meshes, detector image annotations, crop splits and trained weights are external assets. Their paths remain explicit in configuration/CLI options. Do not use the generic checkpoints from a development workspace as substitutes for study models. RF-DETR currently needs its COCO class map/annotations as well as a checkpoint. CAD references preserve the expected `dependencies/cad_synth/...` relative layout; generator code and mesh rights must be supplied separately.
