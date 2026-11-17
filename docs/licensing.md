# Licensing decisions still required

This code-only candidate has no assigned project license. Keep the initial repository private until the company approves source ownership, redistribution and the applicable license. No third-party weights or dependency source are bundled.

| Execution path | Dependency licenses to retain/review |
| --- | --- |
| Motion/DTW and analysis | FastDTW: MIT; NumPy/SciPy/pandas/statsmodels: BSD; PyYAML: MIT; Matplotlib: its upstream license. Exact wheel/transitive notices still need inventory. |
| MediaPipe and OpenCV | Apache-2.0 core licenses; inspect packaged assets, contrib and codec notices. |
| RF-DETR Small | Apache-2.0 candidate for the exact code/pretraining/checkpoint chain; verify the model-specific terms. |
| YOLO pose training/inference | Ultralytics AGPL-3.0 or an applicable Enterprise arrangement; review the integrated code and trained weights before selecting the project license. |
| OAK-D | DepthAI Python MIT; the separate hand-tracker dependency and bundled assets require their own review. |
| CAD/virtual projections | Meshes/generator are absent; their rights cannot be inferred from this code. |

The unused `dtw-python` and `tslearn` dependencies and commented alternative comparison blocks were removed; the executable FastDTW path was retained. FoundationPose source, model data and comparison scripts were excluded. A small imported pose-pilot bridge remains because current pipeline modules import it; it does not bundle FoundationPose.

Possible project license direction is AGPL-3.0 for a company-approved integrated open-source release. Separately licensed independent components need explicit boundaries and review. Neither moving model weights to HF nor using an HTTP inference process settles copyleft obligations. If an Enterprise license exists, verify its public code/model redistribution terms rather than assuming permissive relicensing.

Original source provenance is recorded in `source_manifest.json`; no copyright holder was invented. Add approved LICENSE/NOTICE/copyright attribution and cite the paper once its final identifier is available.

Upstream references: [Ultralytics](https://www.ultralytics.com/license), [RF-DETR](https://github.com/roboflow/rf-detr/blob/develop/LICENSE), [RF-DETR model terms](https://roboflow.com/model-licenses/rf-detr), [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/LICENSE), [FastDTW](https://github.com/slaypni/fastdtw/blob/master/LICENSE).
