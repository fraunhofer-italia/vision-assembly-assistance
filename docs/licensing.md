# Project and third-party licensing

The owner selected the **MIT License** for the project's original source code on 17 November 2026. The full text is in the root `LICENSE` file and the package metadata references it. No third-party weights or dependency source are bundled. This selection does not relicense external libraries, pretrained models or CAD assets.

| Execution path | Dependency licenses to retain/review |
| --- | --- |
| Motion/DTW and analysis | FastDTW: MIT; NumPy/SciPy/pandas/statsmodels: BSD; PyYAML: MIT; Matplotlib: its upstream license. Exact wheel/transitive notices still need inventory. |
| MediaPipe and OpenCV | Apache-2.0 core licenses; inspect packaged assets, contrib and codec notices. |
| RF-DETR Small | Apache-2.0 for Small according to upstream model terms; include the upstream license with the checkpoint. |
| YOLO pose training/inference | Ultralytics AGPL-3.0 or an applicable Enterprise arrangement; review obligations for the integrated application and trained weights. The project's MIT license does not replace these terms. |
| OAK-D | DepthAI Python MIT; the separate hand-tracker dependency and bundled assets require their own review. |
| CAD/virtual projections | Meshes/generator are absent; their rights cannot be inferred from this code. |

The unused `dtw-python` and `tslearn` dependencies and commented alternative comparison blocks were removed; the executable FastDTW path was retained. FoundationPose source, model data and comparison scripts were excluded. A small imported pose-pilot bridge remains because current pipeline modules import it; it does not bundle FoundationPose.

MIT permits reuse of the project's original code. Combining it with AGPL-covered software can impose additional copyleft obligations on the combined work; do not describe the complete Ultralytics-based stack or its weights as MIT-only. Neither moving model weights to HF nor using an HTTP inference process settles those obligations. If an Enterprise license exists, verify its code/model redistribution terms rather than assuming permissive relicensing.

Original source provenance is recorded in `source_manifest.json`. The MIT notice credits the project contributors collectively without assigning exclusive ownership to a named individual or institution. Preserve applicable third-party attribution and cite the paper once its final identifier is available.

Upstream references: [Ultralytics](https://www.ultralytics.com/license), [RF-DETR](https://github.com/roboflow/rf-detr/blob/develop/LICENSE), [RF-DETR model terms](https://roboflow.com/model-licenses/rf-detr), [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/v0.10.14/LICENSE), [FastDTW](https://github.com/slaypni/fastdtw/blob/master/LICENSE).
