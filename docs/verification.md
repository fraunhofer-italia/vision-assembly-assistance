# Verification

Verified on 19 November 2026:

| Check | Result |
| --- | --- |
| Unit and regression tests | 135 passed |
| CLI help/import checks | 20 passed |
| Camera-free example | Two stages completed |
| Package build and installation | Passed |
| Full study analysis | Completed with numeric participant IDs, interventions and questionnaires |
| RF-DETR Small inference | Used for pose crop extraction |
| Remote GitHub/HF contents | File inventories, hashes, README and licenses matched the prepared release |

The analysis environment is pinned in `requirements/verification.lock.txt`; pose extraction uses `requirements/pose-dataset.txt`. Descriptives and learning predictions match the supplied outputs within numerical tolerance; some optimizer fits have small numerical differences.
