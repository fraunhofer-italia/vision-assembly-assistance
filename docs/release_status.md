# Release candidate status

Prepared 17 November 2026 for the private company repository `fraunhofer-italia/vision-assembly-assistance`. The initial code-only import is authorized by the owner. No study data, weights or original repository history are included.

## Selection and changes

The retained entrypoints cover the study UI, offline pipeline, calibration, the final statistical script, RF-DETR training, pose crop/split/training utilities and a camera-free synthetic example. Local import dependencies and relevant unit tests were retained. Legacy MotionFSM and a pose-pilot bridge remain where current imports require them; unused standalone pilots, alternative detectors/statistics, media, weights, results, caches and source Git history were excluded.

Original copied-file hashes are recorded in `source_manifest.json`. Text line endings were normalized to LF. Candidate-specific changes are:

- Replace the hard-coded private questionnaire identity mapping with shared numeric participant IDs; canonicalize integer-like IDs across JSON/CSV joins. Reject identifying IDs without printing them in validation errors.
- Keep all statistical formulas, ML fitting, optimizer selection, exclusions and test/correction procedures. Clarify the CLI questionnaire format. No scientific discrepancy was silently corrected.
- Remove unused commented DTW comparison alternatives and their declared dependencies; keep the FastDTW implementation unchanged.
- Use numeric example participant IDs and a neutral calibration directory in the batch templates; no actual study data were transformed.
- Add package configuration, separate optional extras, a pinned verification environment, documentation, a release-content checker and numeric-ID regression tests.
- Update stale unit expectations to the existing canonical tool names, supply the existing `active_tool` field in a trajectory fixture and accept current optional inference context in the segmentation test double. Production tool/FSM logic was not changed to satisfy tests.
- Replace the broken random synthetic entrypoint with a deterministic, asset-free two-stage example using the actual MotionFSM/StageFSM/FastDTW modules. The old example could leave the first trajectory empty and index a reference for the wrong stage; the new demo prints a summary without creating experiment data. The study's runtime algorithms were not changed.

## Open items

- The original analysis environment is unavailable. The pinned verification versions are a newly prepared check environment, not recovered study versions. Optional RF-DETR/YOLO/MediaPipe/DepthAI execution needs its own resolved environment and asset-dependent verification.
- Missing external assets: actual study/expert recordings and full telemetry, calibration, CAD/generator, real/synthetic detector data and splits, pose crops/splits, deployed detector/classifier weights and looping guidance clips. Their sizes, consent and rights remain to be checked.
- The owner selected MIT for the project's original code. External dependency/model and CAD rights remain subject to their respective terms; see `licensing.md`.
- The source paper/configuration differences identified in the internal release plan remain open: advisory pose gate in study metadata, manual advancement logging provenance, baseline versus mixed detector data, one missing interruption entry, and the outlier-participant count. Current defaults and analyses were preserved rather than silently changed.
- A synthetic demo validates only its supported subset. Successful unit checks do not establish full camera/model operation or reproduction of the paper's results.

## Repository settings

The owner created `https://github.com/fraunhofer-italia/vision-assembly-assistance`, verified private and empty before the initial import, and subsequently selected MIT for the original project source. Only this candidate folder belongs in that repository. The surrounding workspace, archives, original history and internal release plan are excluded.

The owner authorized the initial private upload by supplying this repository URL after reviewing the prepared candidate. Public publication and any data/model release remain separate decisions; keep the repository private until those decisions and the missing-asset issues are resolved.
