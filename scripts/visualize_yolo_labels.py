import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_classes(classes_path: Path) -> list[str]:
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not classes:
        raise SystemExit(f"No classes found in {classes_path}")
    return classes


def _find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _draw_label(image, label_path: Path, class_names: list[str]) -> None:
    height, width = image.shape[:2]
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x_center, y_center, box_w, box_h = [float(v) for v in parts[1:]]

        x_min = int((x_center - box_w / 2.0) * width)
        y_min = int((y_center - box_h / 2.0) * height)
        x_max = int((x_center + box_w / 2.0) * width)
        y_max = int((y_center + box_h / 2.0) * height)

        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(0, min(x_max, width - 1))
        y_max = max(0, min(y_max, height - 1))

        color = (0, 255, 0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        text_x = x_min
        text_y = max(text_h + 4, y_min - 4)
        cv2.rectangle(
            image,
            (text_x, text_y - text_h - baseline - 4),
            (text_x + text_w + 6, text_y + baseline - 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (text_x + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay YOLO detection labels on dataset images for visual QA.")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", required=True, help="Directory containing YOLO .txt labels")
    parser.add_argument("--classes", default="data/detector_dataset_v1/classes.txt", help="Line-based class list")
    parser.add_argument("--output-dir", required=True, help="Directory to write annotated preview images")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of annotated previews to generate")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    classes_path = Path(args.classes)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not classes_path.exists():
        raise SystemExit(f"Classes file not found: {classes_path}")

    class_names = _load_classes(classes_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_images = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        if written >= int(args.limit):
            break
        image_path = _find_image_path(images_dir, label_path.stem)
        if image_path is None:
            missing_images.append(label_path.stem)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        _draw_label(image, label_path, class_names)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), image)
        written += 1

    print(
        {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "output_dir": str(output_dir),
            "written": written,
            "missing_images_for_labels": missing_images[:10],
            "missing_images_for_labels_count": len(missing_images),
        }
    )


if __name__ == "__main__":
    main()
