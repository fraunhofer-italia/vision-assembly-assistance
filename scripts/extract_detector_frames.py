import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def _laplacian_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_difference_score(a, b):
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(diff)) / 255.0


def extract_frames(
    video_path: Path,
    *,
    stride: int,
    max_frames: int | None,
    min_diff: float,
    min_blur: float,
    start_time_s: float,
    end_time_s: float | None,
    min_frame_gap: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    start_frame = max(0, int(start_time_s * fps))
    end_frame = None if end_time_s is None else max(start_frame, int(end_time_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples = []
    prev_kept = None
    prev_kept_frame_id = None
    frame_id = start_frame
    kept = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if end_frame is not None and frame_id > end_frame:
            break
        if (frame_id - start_frame) % stride != 0:
            frame_id += 1
            continue

        blur_score = _laplacian_variance(frame)
        if blur_score < min_blur:
            frame_id += 1
            continue

        diff_score = None
        if prev_kept is not None:
            diff_score = _frame_difference_score(prev_kept, frame)
            gap_ok = prev_kept_frame_id is None or (frame_id - prev_kept_frame_id) >= max(1, min_frame_gap)
            if diff_score < min_diff and not gap_ok:
                frame_id += 1
                continue

        out_name = f"{video_path.stem}_f{frame_id:06d}.jpg"
        samples.append(
            {
                "filename": out_name,
                "frame": frame.copy(),
                "source_video": str(video_path),
                "frame_id": int(frame_id),
                "time_s": float(frame_id / fps),
                "blur_score": float(blur_score),
                "diff_score_from_prev_kept": None if diff_score is None else float(diff_score),
            }
        )
        prev_kept = frame.copy()
        prev_kept_frame_id = frame_id
        kept += 1
        frame_id += 1
        if max_frames is not None and kept >= max_frames:
            break

    cap.release()
    return samples


def _parse_ratios(text_value: str):
    parts = [float(x.strip()) for x in str(text_value).split(",") if x.strip()]
    if len(parts) != 3:
        raise SystemExit("--ratios must contain exactly 3 comma-separated values: train,val,test")
    total = sum(parts)
    if total <= 0:
        raise SystemExit("--ratios must sum to a positive value")
    return [p / total for p in parts]


def _assign_auto_splits(samples, ratios, rng):
    if not samples:
        return {"train": [], "val": [], "test": []}
    ordered = sorted(samples, key=lambda item: int(item["frame_id"]))
    total = len(ordered)
    train_n = int(round(total * ratios[0]))
    val_n = int(round(total * ratios[1]))
    if total >= 3:
        train_n = max(1, min(total - 2, train_n))
        val_n = max(1, min(total - train_n - 1, val_n))
    test_n = total - train_n - val_n
    if total >= 3 and test_n <= 0:
        test_n = 1
        if train_n >= val_n and train_n > 1:
            train_n -= 1
        elif val_n > 1:
            val_n -= 1
    train_cut = train_n
    val_cut = train_n + val_n
    train_pool = ordered[:train_cut]
    val_pool = ordered[train_cut:val_cut]
    test_pool = ordered[val_cut:]
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    rng.shuffle(test_pool)
    return {"train": train_pool, "val": val_pool, "test": test_pool}


def _write_samples(samples, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        out_path = output_dir / sample["filename"]
        cv2.imwrite(str(out_path), sample["frame"])
        records.append(
            {
                "image_path": str(out_path),
                "source_video": sample["source_video"],
                "frame_id": int(sample["frame_id"]),
                "time_s": float(sample["time_s"]),
                "blur_score": float(sample["blur_score"]),
                "diff_score_from_prev_kept": sample["diff_score_from_prev_kept"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract candidate detector training frames from videos.")
    parser.add_argument("--input", required=True, nargs="+", help="One or more input video paths")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "auto"], help="Dataset split for extracted frames; use auto to randomly distribute one video across train/val/test")
    parser.add_argument("--output-root", default="data/detector_dataset_v1/extracted_frames", help="Root output folder for extracted frames")
    parser.add_argument("--manifest-dir", default="data/detector_dataset_v1/manifests", help="Folder to write extraction manifests")
    parser.add_argument("--stride", type=int, default=15, help="Keep every Nth frame before duplicate filtering")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum kept frames per video")
    parser.add_argument("--min-diff", type=float, default=0.03, help="Minimum normalized grayscale difference from previous kept frame")
    parser.add_argument("--min-blur", type=float, default=60.0, help="Minimum Laplacian variance; lower values are likely blurry")
    parser.add_argument("--start-time-s", type=float, default=0.0, help="Optional start time in seconds")
    parser.add_argument("--end-time-s", type=float, default=None, help="Optional end time in seconds")
    parser.add_argument("--min-frame-gap", type=int, default=30, help="Minimum frame gap between consecutive kept frames when difference is low")
    parser.add_argument("--ratios", default="0.7,0.15,0.15", help="When --split auto is used: train,val,test ratios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for auto split assignment")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    ratios = _parse_ratios(args.ratios)
    split_records = {"train": [], "val": [], "test": []}

    for input_item in args.input:
        video_path = Path(input_item)
        samples = extract_frames(
            video_path,
            stride=max(1, int(args.stride)),
            max_frames=args.max_frames,
            min_diff=float(args.min_diff),
            min_blur=float(args.min_blur),
            start_time_s=float(args.start_time_s),
            end_time_s=args.end_time_s,
            min_frame_gap=max(1, int(args.min_frame_gap)),
        )
        if args.split == "auto":
            assigned = _assign_auto_splits(samples, ratios, rng)
            for split_name, split_samples in assigned.items():
                records = _write_samples(split_samples, output_root / split_name)
                split_records[split_name].extend(records)
        else:
            records = _write_samples(samples, output_root / args.split)
            split_records[args.split].extend(records)

    if args.split == "auto":
        manifest_path = manifest_dir / "extract_auto.json"
        payload = {
            "split_mode": "auto",
            "videos": args.input,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "seed": int(args.seed),
            "records_by_split": split_records,
        }
        with open(manifest_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps({
            "split": "auto",
            "videos": args.input,
            "kept_frames": sum(len(v) for v in split_records.values()),
            "counts_by_split": {k: len(v) for k, v in split_records.items()},
            "output_root": str(output_root),
            "manifest": str(manifest_path),
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "seed": int(args.seed),
            "min_frame_gap": int(args.min_frame_gap),
        }, indent=2))
    else:
        manifest_path = manifest_dir / f"extract_{args.split}.json"
        with open(manifest_path, "w") as f:
            json.dump(split_records[args.split], f, indent=2)
        print(json.dumps({
            "split": args.split,
            "videos": args.input,
            "kept_frames": len(split_records[args.split]),
            "output_dir": str(output_root / args.split),
            "manifest": str(manifest_path),
            "min_frame_gap": int(args.min_frame_gap),
        }, indent=2))


if __name__ == "__main__":
    main()
