from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

from training_utils import (
    GPUMemoryMonitor,
    apply_common_training_overrides,
    load_comparison_config,
    prepare_mixed_coco_dataset,
    write_training_run_report,
)


MODEL_MAP = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _build_model(size: str, pretrain_weights: str | None):
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    except Exception as exc:
        raise SystemExit(
            "RF-DETR is not installed. Install it first with: `python3 -m pip install -r requirements-rfdetr.txt`"
        ) from exc

    cls_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    kwargs = {}
    if pretrain_weights:
        kwargs["pretrain_weights"] = pretrain_weights
    return cls_map[size](**kwargs)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(description="Train an RF-DETR detector on a COCO-format dataset.")
    parser.add_argument(
        "--dataset-dir",
        default="data/detector_dataset_v1/Assembly_parts_detection.v2i.coco",
        help="Root directory of the COCO-format dataset containing train/valid/test folders",
    )
    parser.add_argument(
        "--size",
        default="small",
        choices=["nano", "small", "medium", "large"],
        help="RF-DETR model size",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="0", help="Training device, informational for comparison tracking")
    parser.add_argument("--output-dir", default="data/detector_dataset_v1/runs/rfdetr", help="Output directory")
    parser.add_argument("--run-name", default="piston_rfdetr", help="Run-name suffix under output-dir")
    parser.add_argument(
        "--comparison-config",
        default=None,
        help="Optional JSON file that defines common comparison settings",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON output path for training time and GPU-memory tracking",
    )
    parser.add_argument(
        "--pretrain-weights",
        default=None,
        help="Optional path to pretrained RF-DETR weights directory/checkpoint if supported by installed version",
    )
    parser.add_argument(
        "--mix-synthetic-coco-dir",
        default=None,
        help="Optional synthetic COCO dataset root to mix with the real dataset",
    )
    parser.add_argument(
        "--mixed-dataset-output-dir",
        default="data/detector_dataset_v1/mixed_coco_cache",
        help="Output directory where mixed COCO dataset will be materialized",
    )
    parser.add_argument(
        "--mix-val-test",
        action="store_true",
        help="If set, include synthetic data also in val/test. Default keeps val/test real-only.",
    )
    parser.add_argument(
        "--synthetic-train-fraction",
        type=float,
        default=1.0,
        help="Fraction of synthetic TRAIN images to include when mixing (0.0-1.0)",
    )
    parser.add_argument(
        "--synthetic-sample-seed",
        type=int,
        default=42,
        help="Random seed for synthetic train subsampling",
    )
    args = parser.parse_args()

    comparison_config = load_comparison_config(args.comparison_config)
    apply_common_training_overrides(args, comparison_config, backend="rfdetr")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if args.mix_synthetic_coco_dir:
        synthetic_dir = Path(args.mix_synthetic_coco_dir)
        if not synthetic_dir.exists():
            raise SystemExit(f"Synthetic COCO dataset directory not found: {synthetic_dir}")
        mixed_out = Path(args.mixed_dataset_output_dir) / f"{args.run_name}_mixed"
        dataset_dir = prepare_mixed_coco_dataset(
            real_dataset_dir=dataset_dir,
            synthetic_dataset_dir=synthetic_dir,
            output_dir=mixed_out,
            real_val_split="valid",
            synthetic_val_split="val",
            mix_val_test=bool(args.mix_val_test),
            synthetic_train_fraction=float(args.synthetic_train_fraction),
            sample_seed=int(args.synthetic_sample_seed),
        )

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(args.size, args.pretrain_weights)
    started_at = time.time()
    monitor = GPUMemoryMonitor(enabled=str(args.device).lower() != "cpu")
    monitor.start()
    results = model.train(
        dataset_dir=str(dataset_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        lr=float(args.lr),
        output_dir=str(output_dir),
    )
    monitor.stop()
    ended_at = time.time()
    report_path = Path(args.report_path) if args.report_path else output_dir / "training_report.json"
    write_training_run_report(
        report_path,
        model_family="RF-DETR",
        model_name=str(args.size),
        dataset=str(dataset_dir),
        run_name=str(args.run_name),
        started_at_unix=started_at,
        ended_at_unix=ended_at,
        peak_gpu_memory_mib=monitor.peak_memory_mib,
        extra={
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "effective_batch": int(args.batch_size) * int(args.grad_accum_steps),
            "lr": float(args.lr),
            "device": str(args.device),
            "comparison_config": str(args.comparison_config) if args.comparison_config else None,
            "mix_synthetic_coco_dir": str(args.mix_synthetic_coco_dir) if args.mix_synthetic_coco_dir else None,
            "synthetic_train_fraction": float(args.synthetic_train_fraction),
            "synthetic_sample_seed": int(args.synthetic_sample_seed),
        },
    )
    print(
        {
            "model_family": "RF-DETR",
            "model_size": args.size,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "results": str(results),
        }
    )


if __name__ == "__main__":
    main()
