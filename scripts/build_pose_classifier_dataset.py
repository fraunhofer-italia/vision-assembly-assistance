#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


POSE_LABELS = ("correct_pose", "side_pose", "flipped_pose")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _copy_or_link(src: Path, dst: Path, *, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _split_sequences(sequences: list[Path], train: float, val: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(int(seed))
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    n_train = int(round(n * float(train)))
    n_val = int(round(n * float(val)))
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)
    elif n == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _collect_sequences(raw_root: Path, part_label: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    part_root = raw_root / part_label
    for label in POSE_LABELS:
        label_root = part_root / label
        if not label_root.exists():
            out[label] = []
            continue
        out[label] = sorted([p for p in label_root.iterdir() if p.is_dir()])
    return out


def _sequence_images(sequence_dir: Path) -> list[Path]:
    image_dir = sequence_dir / "images"
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Ultralytics classification dataset from pose-classifier raw sequences with video-level splits."
    )
    parser.add_argument("--raw-root", default="data/classifier_dataset_v1/raw_sequences")
    parser.add_argument("--part-label", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symlink", action="store_true", help="Symlink images instead of copying them.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    part_label = str(args.part_label).strip()
    output_dir = Path(args.output_dir or f"data/classifier_dataset_v1/ultralytics/{part_label}").resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences_by_label = _collect_sequences(raw_root, part_label)
    manifest = {
        "part_label": part_label,
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "split_policy": "video_sequence_level",
        "seed": int(args.seed),
        "classes": list(POSE_LABELS),
        "splits": {split: [] for split in ("train", "val", "test")},
    }

    for label, sequences in sequences_by_label.items():
        split_map = _split_sequences(sequences, float(args.train_ratio), float(args.val_ratio), int(args.seed))
        for split, split_sequences in split_map.items():
            for sequence in split_sequences:
                images = _sequence_images(sequence)
                manifest["splits"][split].append(
                    {
                        "label": label,
                        "sequence": sequence.name,
                        "source_dir": str(sequence),
                        "num_images": len(images),
                    }
                )
                for image in images:
                    dst_name = f"{sequence.name}__{image.name}"
                    _copy_or_link(image, output_dir / split / label / dst_name, symlink=bool(args.symlink))

    yaml_text = "\n".join(
        [
            f"path: {output_dir}",
            "train: train",
            "val: val",
            "test: test",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(POSE_LABELS)],
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
