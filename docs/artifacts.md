# Data and models

| Artifact | Hugging Face repository |
| --- | --- |
| Study metadata, interventions, questionnaires, analysis, homographies and DTW references | [piston-assembly-study](https://huggingface.co/datasets/fhi/piston-assembly-study) |
| Study recordings | [piston-assembly-recordings](https://huggingface.co/datasets/fhi/piston-assembly-recordings) |
| COCO detector images and annotations | [piston-parts-detection](https://huggingface.co/datasets/fhi/piston-parts-detection) |
| Pose classifier dataset | [piston-pose-classification](https://huggingface.co/datasets/fhi/piston-pose-classification) |
| RF-DETR Small detector | [piston-rfdetr-small](https://huggingface.co/fhi/piston-rfdetr-small) |
| Front cover YOLOv8n-cls | [piston-front-cover-pose](https://huggingface.co/fhi/piston-front-cover-pose) |
| Rear cover YOLOv8n-cls | [piston-rear-cover-pose](https://huggingface.co/fhi/piston-rear-cover-pose) |
| Experimental piston rod YOLOv8n-cls | [piston-rod-pose-experimental](https://huggingface.co/fhi/piston-rod-pose-experimental) |

The repositories are publicly accessible. Code and datasets use MIT. RF-DETR weights use Apache 2.0; Ultralytics weights retain AGPL-3.0 terms. Repository cards list usage, classes and evaluation results.

To run the study analysis, download `piston-assembly-study` to `data/study` and run:

```bash
python scripts/guidance_stats_final_analysis_model_B.py --experiments-root data/study/experiments --interruptions-csv data/study/experiments/interruptions.csv --questionnaire-csv data/study/experiments/questionnaire.csv --output-dir results/analysis
```

For pose crop preparation, install `requirements/pose-dataset.txt` in a separate environment and see `scripts/prepare_pose_dataset.py --help`. The dataset card documents the frame filter, video-level splits and background-crop exclusions.
