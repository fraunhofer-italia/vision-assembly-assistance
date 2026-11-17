import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_classes(classes_path: Path) -> list[str]:
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not classes:
        raise SystemExit(f"No classes found in {classes_path}")
    return classes


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _convert_bbox_xywh_to_yolo(bbox: list[float], image_w: int, image_h: int) -> tuple[float, float, float, float]:
    x_min, y_min, width, height = [float(v) for v in bbox]
    x_max = x_min + width
    y_max = y_min + height

    x_min = max(0.0, min(x_min, float(image_w)))
    y_min = max(0.0, min(y_min, float(image_h)))
    x_max = max(0.0, min(x_max, float(image_w)))
    y_max = max(0.0, min(y_max, float(image_h)))

    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    return (
        _clip01(x_center / float(image_w)),
        _clip01(y_center / float(image_h)),
        _clip01(width / float(image_w)),
        _clip01(height / float(image_h)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a COCO JSON export and convert it into standard YOLO detection labels."
    )
    parser.add_argument("--coco-json", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--classes", default="data/detector_dataset_v1/classes.txt", help="Line-based class list")
    parser.add_argument("--output-labels-dir", help="Where to write YOLO .txt labels")
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty .txt files for images with no annotations mapped to the class list",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only validate/report the COCO file, do not write labels",
    )
    args = parser.parse_args()

    coco_path = Path(args.coco_json)
    if not coco_path.exists():
        raise SystemExit(f"COCO JSON not found: {coco_path}")

    classes_path = Path(args.classes)
    if not classes_path.exists():
        raise SystemExit(f"Classes file not found: {classes_path}")

    classes = _load_classes(classes_path)
    class_to_index = {name: idx for idx, name in enumerate(classes)}

    with coco_path.open("r") as f:
        payload = json.load(f)

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise SystemExit("COCO JSON is missing one of the required list fields: images, annotations, categories")

    image_by_id = {item["id"]: item for item in images if "id" in item}
    category_id_to_name = {item["id"]: item["name"] for item in categories if "id" in item and "name" in item}

    missing_category_names = sorted(set(class_to_index) - set(category_id_to_name.values()))
    extra_category_names = sorted(set(category_id_to_name.values()) - set(class_to_index))

    ann_by_image_id: dict[int, list[dict]] = defaultdict(list)
    counts_by_class = Counter()
    skipped_annotations = 0
    invalid_boxes = 0
    missing_images = 0

    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if image_id not in image_by_id:
            missing_images += 1
            continue
        if category_id not in category_id_to_name:
            skipped_annotations += 1
            continue
        category_name = category_id_to_name[category_id]
        if category_name not in class_to_index:
            skipped_annotations += 1
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_boxes += 1
            continue
        ann_by_image_id[image_id].append(ann)
        counts_by_class[category_name] += 1

    summary = {
        "coco_json": str(coco_path),
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "mapped_categories": sorted(name for name in category_id_to_name.values() if name in class_to_index),
        "missing_category_names_from_coco": missing_category_names,
        "extra_category_names_not_in_classes_txt": extra_category_names,
        "skipped_annotations": skipped_annotations,
        "invalid_boxes": invalid_boxes,
        "annotations_with_missing_image": missing_images,
        "counts_by_class": dict(counts_by_class),
    }

    if args.report_only:
        print(json.dumps(summary, indent=2))
        return

    if not args.output_labels_dir:
        raise SystemExit("--output-labels-dir is required unless --report-only is used")

    output_dir = Path(args.output_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = 0
    written_annotations = 0

    for image in images:
        image_id = image.get("id")
        if image_id is None:
            continue
        image_w = int(image.get("width", 0))
        image_h = int(image.get("height", 0))
        file_name = Path(image.get("file_name", ""))
        if image_w <= 0 or image_h <= 0 or not file_name.name:
            continue

        lines = []
        for ann in ann_by_image_id.get(image_id, []):
            category_name = category_id_to_name[ann["category_id"]]
            class_id = class_to_index[category_name]
            x_center, y_center, width, height = _convert_bbox_xywh_to_yolo(ann["bbox"], image_w, image_h)
            if width <= 0.0 or height <= 0.0:
                continue
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            written_annotations += 1

        if not lines and not args.write_empty:
            continue

        label_path = output_dir / f"{file_name.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written_files += 1

    summary.update(
        {
            "output_labels_dir": str(output_dir),
            "written_label_files": written_files,
            "written_annotations": written_annotations,
            "write_empty": bool(args.write_empty),
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
