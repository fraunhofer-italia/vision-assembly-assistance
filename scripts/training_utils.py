from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path


def load_comparison_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Comparison config not found: {config_path}")
    with config_path.open("r") as f:
        return json.load(f)


def apply_common_training_overrides(args, config: dict, backend: str) -> None:
    common = config.get("common", {})
    backend_cfg = config.get(backend, {})

    if hasattr(args, "epochs") and common.get("epochs") is not None:
        args.epochs = int(common["epochs"])
    if hasattr(args, "imgsz") and common.get("imgsz") is not None:
        args.imgsz = int(common["imgsz"])
    if hasattr(args, "batch") and common.get("batch") is not None:
        args.batch = int(common["batch"])
    if hasattr(args, "device") and common.get("device") is not None:
        args.device = str(common["device"])

    for key, value in backend_cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)


class GPUMemoryMonitor:
    def __init__(self, enabled: bool = True, interval_sec: float = 2.0):
        self.enabled = enabled
        self.interval_sec = interval_sec
        self.peak_memory_mib = None
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._available = None

    def _query(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(float(line)))
            except ValueError:
                continue
        return max(values) if values else None

    def _loop(self):
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self.samples.append(value)
                self.peak_memory_mib = max(self.samples)
                self._available = True
            elif self._available is None:
                self._available = False
            self._stop.wait(self.interval_sec)

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_sec + 1.0)


def write_training_run_report(
    output_path: str | Path,
    *,
    model_family: str,
    model_name: str,
    dataset: str,
    run_name: str,
    started_at_unix: float,
    ended_at_unix: float,
    peak_gpu_memory_mib,
    extra: dict | None = None,
) -> None:
    report = {
        "model_family": model_family,
        "model_name": model_name,
        "dataset": dataset,
        "run_name": run_name,
        "started_at_unix": started_at_unix,
        "ended_at_unix": ended_at_unix,
        "wall_time_sec": ended_at_unix - started_at_unix,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
    }
    if extra:
        report.update(extra)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _to_list(value) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return [value]


def extract_ultralytics_per_class_metrics(metrics, class_names: dict | list | None = None) -> dict:
    """
    Best-effort extraction of per-class metrics from an Ultralytics validation result object.
    This is defensive across versions because metric object attributes have changed.
    """
    box = getattr(metrics, "box", None)
    if box is None:
        return {"per_class_metrics": {}, "overall": {}}

    maps = _to_list(getattr(box, "maps", None))
    names = getattr(metrics, "names", None)
    if names is None:
        names = class_names
    if names is None:
        names = {}

    if isinstance(names, list):
        names = {idx: name for idx, name in enumerate(names)}

    ap50_list = []
    for attr in ("ap50", "all_ap50", "p50"):
        value = getattr(box, attr, None)
        if value is not None:
            ap50_list = _to_list(value)
            if ap50_list:
                break

    precision_list = []
    for attr in ("p", "precision"):
        value = getattr(box, attr, None)
        if value is not None:
            precision_list = _to_list(value)
            if precision_list:
                break

    recall_list = []
    for attr in ("r", "recall"):
        value = getattr(box, attr, None)
        if value is not None:
            recall_list = _to_list(value)
            if recall_list:
                break

    per_class = {}
    count = max(len(maps), len(ap50_list), len(precision_list), len(recall_list), len(names))
    for idx in range(count):
        class_name = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
        per_class[class_name] = {
            "AP50": _safe_float(ap50_list[idx]) if idx < len(ap50_list) else None,
            "AP50-95": _safe_float(maps[idx]) if idx < len(maps) else None,
            "precision": _safe_float(precision_list[idx]) if idx < len(precision_list) else None,
            "recall": _safe_float(recall_list[idx]) if idx < len(recall_list) else None,
        }

    overall = {
        "mAP50": _safe_float(getattr(box, "map50", None)),
        "mAP50-95": _safe_float(getattr(box, "map", None)),
        "precision": _safe_float(getattr(box, "mp", None)),
        "recall": _safe_float(getattr(box, "mr", None)),
    }
    return {"per_class_metrics": per_class, "overall": overall}


def write_per_class_metrics_report(output_path: str | Path, metrics, class_names: dict | list | None = None) -> None:
    report = extract_ultralytics_per_class_metrics(metrics, class_names=class_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def _resolve_coco_image_path(split_dir: Path, file_name: str) -> Path | None:
    candidates = [
        split_dir / file_name,
        split_dir / "images" / file_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _merge_coco_split(
    out_split_dir: Path,
    *,
    real_split_dir: Path,
    synth_split_dir: Path,
    real_prefix: str,
    synth_prefix: str,
    category_id_by_name: dict,
    root_supercategory_name: str,
    synthetic_image_fraction: float = 1.0,
    sample_seed: int = 42,
) -> None:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    merged_images = []
    merged_annotations = []
    next_image_id = 1
    next_ann_id = 1

    def ingest(source_ann_path: Path, source_split_dir: Path, prefix: str) -> None:
        nonlocal next_image_id, next_ann_id
        data = _load_json(source_ann_path)
        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = data.get("categories", [])
        src_cat_name_by_id = {int(c["id"]): c["name"] for c in cats if c.get("id") is not None and c.get("name")}

        if prefix == synth_prefix and images:
            frac = float(synthetic_image_fraction)
            if frac <= 0.0:
                images = []
                anns = []
            elif frac < 1.0:
                keep_count = max(1, int(round(len(images) * frac)))
                keep_count = min(keep_count, len(images))
                rng = random.Random(int(sample_seed))
                selected_image_ids = {int(img["id"]) for img in rng.sample(images, keep_count)}
                images = [img for img in images if int(img.get("id", -1)) in selected_image_ids]
                anns = [ann for ann in anns if int(ann.get("image_id", -1)) in selected_image_ids]

        image_id_remap = {}
        for img in images:
            old_id = int(img["id"])
            src_name = img.get("file_name")
            if not src_name:
                continue
            src_img = _resolve_coco_image_path(source_split_dir, src_name)
            if src_img is None:
                continue
            dst_name = f"{prefix}_{src_img.name}"
            dst_img = out_split_dir / dst_name
            _safe_symlink_or_copy(src_img, dst_img)
            image_id_remap[old_id] = next_image_id
            merged_images.append(
                {
                    "id": next_image_id,
                    "file_name": dst_name,
                    "width": int(img.get("width", 0)),
                    "height": int(img.get("height", 0)),
                }
            )
            next_image_id += 1

        for ann in anns:
            old_img_id = int(ann.get("image_id", -1))
            old_cat_id = int(ann.get("category_id", -1))
            if old_img_id not in image_id_remap or old_cat_id not in src_cat_name_by_id:
                continue
            cat_name = src_cat_name_by_id[old_cat_id]
            if cat_name not in category_id_by_name:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_remap[old_img_id]
            new_ann["category_id"] = category_id_by_name[cat_name]
            merged_annotations.append(new_ann)
            next_ann_id += 1

    ingest(real_split_dir / "_annotations.coco.json", real_split_dir, real_prefix)
    ingest(synth_split_dir / "_annotations.coco.json", synth_split_dir, synth_prefix)

    categories = [{"id": 0, "name": root_supercategory_name, "supercategory": "none"}] + [
        {"id": cid, "name": name, "supercategory": root_supercategory_name}
        for name, cid in sorted(category_id_by_name.items(), key=lambda x: x[1])
    ]
    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }
    with (out_split_dir / "_annotations.coco.json").open("w") as f:
        json.dump(merged, f)


def prepare_mixed_coco_dataset(
    *,
    real_dataset_dir: Path,
    synthetic_dataset_dir: Path,
    output_dir: Path,
    real_val_split: str = "valid",
    synthetic_val_split: str = "val",
    mix_val_test: bool = False,
    synthetic_train_fraction: float = 1.0,
    sample_seed: int = 42,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    real_train = _load_json(real_dataset_dir / "train" / "_annotations.coco.json")
    synth_train = _load_json(synthetic_dataset_dir / "train" / "_annotations.coco.json")

    real_categories = [c for c in real_train.get("categories", []) if c.get("name")]
    root_category = next((c for c in real_categories if str(c.get("supercategory", "")) == "none"), None)
    root_supercategory_name = root_category.get("name") if root_category else "object"

    # Keep only real dataset classes (exclude the root category), preserving real class IDs/order.
    category_id_by_name = {}
    for cat in sorted(real_categories, key=lambda x: int(x.get("id", 0))):
        if str(cat.get("supercategory", "")) == "none":
            continue
        cid = int(cat.get("id"))
        cname = str(cat.get("name"))
        if cid <= 0:
            continue
        category_id_by_name[cname] = cid

    if not category_id_by_name:
        raise SystemExit("Could not infer valid detector classes from real COCO train annotations")

    _merge_coco_split(
        output_dir / "train",
        real_split_dir=real_dataset_dir / "train",
        synth_split_dir=synthetic_dataset_dir / "train",
        real_prefix="real",
        synth_prefix="synth",
        category_id_by_name=category_id_by_name,
        root_supercategory_name=root_supercategory_name,
        synthetic_image_fraction=float(synthetic_train_fraction),
        sample_seed=int(sample_seed),
    )
    if mix_val_test:
        _merge_coco_split(
            output_dir / "valid",
            real_split_dir=real_dataset_dir / real_val_split,
            synth_split_dir=synthetic_dataset_dir / synthetic_val_split,
            real_prefix="real",
            synth_prefix="synth",
            category_id_by_name=category_id_by_name,
            root_supercategory_name=root_supercategory_name,
            synthetic_image_fraction=1.0,
            sample_seed=int(sample_seed),
        )
        _merge_coco_split(
            output_dir / "test",
            real_split_dir=real_dataset_dir / "test",
            synth_split_dir=synthetic_dataset_dir / "test",
            real_prefix="real",
            synth_prefix="synth",
            category_id_by_name=category_id_by_name,
            root_supercategory_name=root_supercategory_name,
            synthetic_image_fraction=1.0,
            sample_seed=int(sample_seed),
        )
    else:
        # Write valid/test from real dataset only, remapped to the shared category IDs.
        for split_name, real_split_name in (("valid", real_val_split), ("test", "test")):
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            real_ann = _load_json(real_dataset_dir / real_split_name / "_annotations.coco.json")
            real_images = []
            real_anns = []
            image_id_remap = {}
            next_image_id = 1
            next_ann_id = 1

            for img in real_ann.get("images", []):
                old_id = int(img["id"])
                src_name = img.get("file_name")
                if not src_name:
                    continue
                src_img = _resolve_coco_image_path(real_dataset_dir / real_split_name, src_name)
                if src_img is None:
                    continue
                dst_name = f"real_{Path(src_name).name}"
                _safe_symlink_or_copy(src_img, split_dir / dst_name)
                image_id_remap[old_id] = next_image_id
                real_images.append(
                    {
                        "id": next_image_id,
                        "file_name": dst_name,
                        "width": int(img.get("width", 0)),
                        "height": int(img.get("height", 0)),
                    }
                )
                next_image_id += 1

            src_cat_name_by_id = {
                int(c["id"]): c["name"]
                for c in real_ann.get("categories", [])
                if c.get("id") is not None and c.get("name")
            }
            for ann in real_ann.get("annotations", []):
                old_img_id = int(ann.get("image_id", -1))
                old_cat_id = int(ann.get("category_id", -1))
                if old_img_id not in image_id_remap or old_cat_id not in src_cat_name_by_id:
                    continue
                cat_name = src_cat_name_by_id[old_cat_id]
                if cat_name not in category_id_by_name:
                    continue
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = image_id_remap[old_img_id]
                new_ann["category_id"] = category_id_by_name[cat_name]
                real_anns.append(new_ann)
                next_ann_id += 1

            categories = [
                {"id": 0, "name": root_supercategory_name, "supercategory": "none"}
            ] + [
                {"id": cid, "name": name, "supercategory": root_supercategory_name}
                for name, cid in sorted(category_id_by_name.items(), key=lambda x: x[1])
            ]
            with (split_dir / "_annotations.coco.json").open("w") as f:
                json.dump({"images": real_images, "annotations": real_anns, "categories": categories}, f)
    return output_dir
