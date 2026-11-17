# Vision-based assembly verification and operator guidance

Code for single-overhead-camera pneumatic piston assembly/disassembly verification: MediaPipe hand tracking, regional interaction triggers, MotionFSM/StageFSM, FastDTW reference comparison, RF-DETR detection, task-specific pose classification and full/mid/no guidance.

This is a **code-only release candidate** in the private company repository. The project's original source code is licensed under the [MIT License](LICENSE). No data, models, videos, private submodules or original Git history are included. Missing study assets and third-party/model terms remain documented in [release status](docs/release_status.md). There is no public release or HF download URL yet.

## Install and checks

Use Python 3.11 for the verification environment. Run commands from this repository root. `python` below must refer to the activated environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/verification.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q -p no:cacheprovider
python scripts/validate_release.py
```

The pinned verification environment was prepared for this candidate. It is **not** evidence of the original paper's software versions. Optional camera, MediaPipe, RF-DETR and Ultralytics packages are separate extras; GPU/model-dependent execution has not been verified with this asset-free copy. See [dependencies and licensing](docs/licensing.md).

## Statistical analysis

```bash
python -m pip install -e '.[analysis]'
python scripts/guidance_stats_final_analysis_model_B.py --help
python scripts/guidance_stats_final_analysis_model_B.py --output-dir results/reproduced_model_B_v2
```

Before the last command, supply cleared `experiments/<numeric-trial>/metadata.json`, `experiments/interruptions.csv` and the optional wide questionnaire described in [the data contract](docs/data_contract.md). They are not included here. The candidate accepts numeric participant IDs directly, without a private identity crosswalk. Original data using participant names must be de-identified separately before use.

The script retains its historical filename. Its primary composite model is the participant-plus-nested-trial random-intercept **Model A**; separate secondary models and B/C diagnostics remain in the script. Model formulas, fitting/optimizer selection and correction procedures were retained. This candidate has not reproduced the paper's full model fitting.

## Camera-free synthetic example

```bash
python scripts/main.py
```

This deterministic example segments and verifies two generated trajectories using legacy MotionFSM, StageFSM and real FastDTW scoring. It prints a JSON summary, requires no assets and writes no experiment files. It does not reproduce the study's camera pipeline.

## Recorded video and live guidance

Offline segmentation is available through `scripts/main_full_pipeline.py --mode offline`; playback in the guidance UI uses `scripts/main_guidance_ui.py --input-video VIDEO.mp4`. Use `--help` for their full argument lists. Both require matching expert references, calibration/homography, task settings and any enabled trained models. None is downloaded automatically by the release preparation.

Live guidance defaults to an OAK-D RGB stream and needs `.[vision,camera]` plus the matching workstation configuration. The older `--mode depthai` offline route additionally requires a separately cleared hand-tracker dependency, which is not bundled. The MediaPipe video route does not require that dependency. The UI preserves the original advisory pose default; `--enforce-pose-gate` is an explicit behavioral option.

The existing batch templates in `configs/piston_assembly_v1.json` and `configs/piston_dismantle_v1.json` illustrate the task; their external file names and reference counts are not an exact study manifest. Supply the intended reference set before running them. Use numeric `user_id` values for new metadata destined for the released statistical script.

RF-DETR training uses `scripts/train_rfdetr_detector.py`; pose crop extraction, sequence splitting and training use `extract_pose_classifier_rois.py`, `build_pose_classifier_dataset.py` and `train_yolo_pose_classifier.py`. Install the relevant `rfdetr` or `pose` extras separately and record the resolved training environment. Existing model-selection and alternative statistical experiments have been excluded.

## Contents and provenance

- `src/assembly_assistant/`: runtime modules reached by the retained entrypoints and tests.
- `scripts/`: execution, final analysis, training and data preparation utilities.
- `configs/`: task/ROI/guidance settings and anonymized example identifiers.
- `data/mocks/`: synthetic generators used by the example and tests; no observations.
- `tests/`: relevant existing unit tests and checks for numeric participant joins.
- `docs/source_manifest.json`: original file hashes and candidate modifications.

The core perception/FSM algorithms are preserved. Changes and remaining assets are listed in [release status](docs/release_status.md). MIT covers the project's original code; third-party dependencies and model weights retain their own licenses. In particular, the optional Ultralytics path remains subject to its AGPL-3.0 or applicable Enterprise terms; see [licensing](docs/licensing.md).
