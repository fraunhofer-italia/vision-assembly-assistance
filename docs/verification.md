# Candidate verification

Checked on 17 November 2026 in an isolated Python 3.11.5 environment. Package versions are recorded in `requirements/verification.lock.txt`; this is a new verification environment, not the historical analysis environment.

| Check | Result |
| --- | --- |
| Relevant unit and release regression tests | 134 passed. Thirteen Matplotlib/pyparsing deprecation warnings; no test failures. |
| Shared numeric participant IDs | JSON/intervention joins and wide-questionnaire condition joins verified; invalid/named IDs rejected. |
| Camera-free example | Two complete 40-point segments, both stage checks passed, task complete; no data/model assets required. |
| CLI argument/help loading | 19 retained CLI tools returned successfully without camera/model dependencies. Calibration is checked through its import closure and package imports; no calibration capture was run. |
| Package build/install | Wheel built and installed into an isolated environment; runtime, mock generators and packaged task configurations imported successfully outside the checkout. The wheel is a runtime library; execution scripts remain in the source checkout. |
| Dependency consistency | `pip check` reported no broken requirements. Optional camera/training extras were not installed. |
| Local import closure | No missing retained local imports; the documented optional external DepthAI hand-tracker route is not bundled. |
| Statistical source comparison | 48 existing top-level functions have identical ASTs to the supplied source. Only dataset/trial subject-ID handling, interruption joins, questionnaire IDs and CLI/help/input handling differ; the added ID normalizer contains no identity map. |
| Release content | Static Python/JSON parsing and configured token, private-address, user-path and code-only asset checks pass. Known source participant names and internal project links were also checked separately without adding an identifying denylist to the public candidate. |
| Original source | Original supplied files were checked against the pre-preparation hash snapshot; preparation was confined to the new candidate and temporary verification artifacts. |

No full paper-analysis fit, model training, model inference, video/audio audit, live camera run, original study-version recovery or legal/privacy clearance is implied by these checks. Those require the missing external inputs and approvals listed in `release_status.md`.
