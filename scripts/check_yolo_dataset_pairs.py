import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _stem_map(directory: Path, allowed_exts: set[str]) -> dict[str, Path]:
    items: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_exts:
            continue
        items[path.stem] = path
    return items


def _check_split(dataset_root: Path, split: str) -> dict:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Missing labels directory: {labels_dir}")

    images = _stem_map(images_dir, IMAGE_EXTS)
    labels = _stem_map(labels_dir, {".txt"})

    image_stems = set(images)
    label_stems = set(labels)

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    matched = sorted(image_stems & label_stems)

    empty_labels = []
    malformed_labels = []
    non_normalized_values = []

    for stem in matched:
        label_path = labels[stem]
        content = label_path.read_text().splitlines()
        non_empty_lines = [line.strip() for line in content if line.strip()]
        if not non_empty_lines:
            empty_labels.append(label_path.name)
            continue
        for idx, line in enumerate(non_empty_lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                malformed_labels.append({"file": label_path.name, "line": idx, "content": line})
                continue
            try:
                class_id = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
            except ValueError:
                malformed_labels.append({"file": label_path.name, "line": idx, "content": line})
                continue
            if class_id < 0:
                malformed_labels.append({"file": label_path.name, "line": idx, "content": line})
                continue
            if any(v < 0.0 or v > 1.0 for v in values):
                non_normalized_values.append({"file": label_path.name, "line": idx, "content": line})

    return {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": len(images),
        "label_count": len(labels),
        "matched_pairs": len(matched),
        "missing_labels_count": len(missing_labels),
        "missing_labels_examples": missing_labels[:10],
        "orphan_labels_count": len(orphan_labels),
        "orphan_labels_examples": orphan_labels[:10],
        "empty_labels_count": len(empty_labels),
        "empty_labels_examples": empty_labels[:10],
        "malformed_labels_count": len(malformed_labels),
        "malformed_labels_examples": malformed_labels[:10],
        "non_normalized_values_count": len(non_normalized_values),
        "non_normalized_values_examples": non_normalized_values[:10],
        "healthy": (
            len(missing_labels) == 0
            and len(orphan_labels) == 0
            and len(malformed_labels) == 0
            and len(non_normalized_values) == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that YOLO image/label pairs line up correctly across train/val/test splits."
    )
    parser.add_argument("--dataset-root", default="data/detector_dataset_v1", help="Dataset root")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to check, e.g. train,val,test",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise SystemExit("No splits provided.")

    reports = [_check_split(dataset_root, split) for split in splits]
    overall_healthy = all(report["healthy"] for report in reports)

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "splits": reports,
                "overall_healthy": overall_healthy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
