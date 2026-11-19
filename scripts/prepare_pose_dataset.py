#!/usr/bin/env python3
"""Prepare the pose classifier dataset from single-pose RGB videos."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--min-blur', type=float, default=60.)
    parser.add_argument('--min-diff', type=float, default=.03)
    parser.add_argument('--min-frame-gap', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--bbox-pad', type=float, default=.20)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from rfdetr import RFDETRSmall

    # RF-DETR 1.3.0 requests an unrestricted loader internally. Restrict it to
    # tensors/primitives plus the stdlib Namespace used by this checkpoint.
    original_load = torch.load
    def safe_load(*load_args, **kwargs):
        kwargs['weights_only'] = True
        with torch.serialization.safe_globals([argparse.Namespace]):
            return original_load(*load_args, **kwargs)
    torch.load = safe_load
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    ann = json.loads(args.annotations.read_text())
    used = {a['category_id'] for a in ann['annotations']}
    ids = {c['name']: c['id'] for c in ann['categories'] if c['id'] in used}
    model = RFDETRSmall(pretrain_weights=str(args.checkpoint), device='cpu', num_classes=len(used))
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    settings.update({'detector': 'RF-DETR Small', 'rfdetr_version': '1.3.0',
                     'checkpoint_sha256': sha256(args.checkpoint),
                     'input_color_order': 'RGB', 'min_crop_size': 24,
                     'split_policy': 'video_sequence_level',
                     'class_names': ['correct_pose', 'flipped_pose', 'side_pose']})
    write_json(args.output / 'preparation_config.json', settings)
    inventory = []
    jobs = []
    for folder, part in [('front_cover_v2', 'front_cover'), ('rear_cover', 'rear_cover'), ('piston_rod', 'piston_rod')]:
        for source_pose, pose in [('correct_pose', 'correct_pose'), ('flip_pose', 'flipped_pose'), ('side_pose', 'side_pose'), ('test', None)]:
            videos = sorted((args.source / folder / source_pose).glob('*.mp4'))
            if len(videos) != (1 if pose is None else 3):
                raise RuntimeError('Unexpected video count: ' + str((folder, source_pose)))
            assignments = list(videos)
            random.Random(args.seed).shuffle(assignments)
            split_map = {v: s for v, s in zip(assignments, ['train', 'val', 'test'])}
            for video in videos:
                rel = Path('source_videos') / part / (pose or 'unlabelled_test') / video.name
                dest = args.output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    try:
                        os.link(video, dest)
                    except OSError:
                        shutil.copy2(video, dest)
                if sha256(video) != sha256(dest):
                    raise RuntimeError('Source video mismatch')
                item = {'path': rel.as_posix(), 'sha256': sha256(dest), 'bytes': dest.stat().st_size,
                        'part': part, 'pose': pose, 'split': split_map[video] if pose else None,
                        'sequence': video.stem}
                inventory.append(item)
                if pose:
                    jobs.append((video, item))
    write_json(args.output / 'source_manifest.json', inventory)
    for index, (video, item) in enumerate(jobs, 1):
        record_path = args.output / 'sequences' / item['part'] / item['pose'] / (item['sequence'] + '.json')
        if record_path.exists():
            print('SKIP completed', item['sequence'], flush=True)
            continue
        t0 = time.time()
        print('START', index, '/', len(jobs), item['sequence'], flush=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError('Cannot read video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        records = []
        counts = collections.Counter()
        previous = None
        previous_id = None
        frame_id = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % args.stride:
                continue
            counts['sampled'] += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur < args.min_blur:
                counts['blur_rejected'] += 1
                continue
            diff = None if previous is None else float(np.mean(cv2.absdiff(previous, gray))) / 255.
            if previous is not None and diff < args.min_diff and frame_id - previous_id < args.min_frame_gap:
                counts['similar_rejected'] += 1
                continue
            previous, previous_id = gray.copy(), frame_id
            det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.threshold)
            candidates = np.flatnonzero(det.class_id == ids[item['part']])
            if not len(candidates):
                counts['no_target'] += 1
                continue
            k = int(candidates[np.argmax(det.confidence[candidates])])
            x1, y1, x2, y2 = [float(v) for v in det.xyxy[k]]
            pad = max(x2 - x1, y2 - y1) * args.bbox_pad
            height, width = frame.shape[:2]
            box = [max(0, math.floor(x1-pad)), max(0, math.floor(y1-pad)),
                   min(width, math.ceil(x2+pad)), min(height, math.ceil(y2+pad))]
            a, b, c, d = box
            crop = frame[b:d, a:c]
            if min(crop.shape[:2]) < 24:
                counts['tiny_rejected'] += 1
                continue
            image_rel = Path('datasets') / item['part'] / item['split'] / item['pose'] / f"{item['sequence']}_f{frame_id:06d}.jpg"
            out = args.output / image_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), crop):
                raise RuntimeError('Failed to write crop')
            records.append({'image': image_rel.as_posix(), 'sha256': sha256(out),
                            'frame_id': frame_id, 'time_s': frame_id / fps,
                            'bbox_xyxy': [x1,y1,x2,y2], 'bbox_xyxy_padded': box,
                            'detection_confidence': float(det.confidence[k]),
                            'frame_blur_score': blur, 'frame_difference': diff})
            counts['kept'] += 1
        cap.release()
        write_json(record_path, dict(item, fps=fps, counts=dict(counts), images=records))
        print('DONE', item['sequence'], dict(counts), 'seconds', round(time.time()-t0,1), flush=True)
    print('ALL SEQUENCES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
