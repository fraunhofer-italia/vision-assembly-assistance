#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.pipeline.region_registry import RegionRegistry


def _load_frame(*, video: str | None, image: str | None, frame_index: int):
    if bool(video) == bool(image):
        raise SystemExit("Provide exactly one input: --video or --image")
    if image:
        frame = cv2.imread(image)
        if frame is None:
            raise SystemExit(f"Failed to read image: {image}")
        return frame
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Failed to read frame {frame_index} from video: {video}")
    return frame


def _load_homography(calibration: str | None):
    if calibration is None:
        return None
    p = Path(calibration)
    if p.is_dir():
        h_path = p / "homography.npy"
        if not h_path.exists():
            raise SystemExit(f"Missing homography.npy in {p}")
        H = np.load(str(h_path)).astype(float)
        return H
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        H = np.array(data.get("homography"), dtype=float)
        if H.shape != (3, 3):
            raise SystemExit("Calibration JSON missing homography 3x3")
        return H
    raise SystemExit("--calibration must be a dir with homography.npy or a JSON with homography")


def _warp_norm_to_px(pt_norm, H_inv):
    x, y = float(pt_norm[0]), float(pt_norm[1])
    p = np.array([x, y, 1.0], dtype=float)
    p = H_inv @ p
    p = p / max(1e-9, p[2])
    return [float(p[0]), float(p[1])]


def _draw_polygon(frame, pts, label, color):
    if not pts:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2)
    lx, ly = int(pts[0][0]), int(pts[0][1])
    cv2.putText(frame, label, (lx + 6, ly + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _overlay_regions(frame, reg: RegionRegistry, H_inv):
    colors = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
        (128, 255, 0),
        (0, 255, 128),
        (255, 0, 128),
    ]
    for i, (rid, region) in enumerate(reg.regions.items()):
        color = colors[i % len(colors)]
        if region.coordinate_frame == "image_px":
            pts = region.polygon
            _draw_polygon(frame, pts, rid, color)
        elif region.coordinate_frame == "workspace_norm":
            if H_inv is None:
                continue
            pts = [_warp_norm_to_px(p, H_inv) for p in region.polygon]
            _draw_polygon(frame, pts, rid, color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay all regions on a frame for visual verification.")
    parser.add_argument("--video", default=None, help="Path to video file (uses first frame unless --frame-index set).")
    parser.add_argument("--image", default=None, help="Path to image file.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index for --video (start frame if --play).")
    parser.add_argument("--regions-config", required=True, help="Path to region registry JSON (use configs/regions_v2.json)")
    parser.add_argument("--calibration", default=None, help="Calibration dir/json for workspace_norm regions.")
    parser.add_argument("--output", default=None, help="Optional output image path.")
    parser.add_argument("--show", action="store_true", help="Show overlay window.")
    parser.add_argument("--play", action="store_true", help="Overlay and play video (requires --video).")
    parser.add_argument("--output-video", default=None, help="Optional output video path (use with --play).")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to render in --play (0 = all).")
    parser.add_argument("--play-fps", type=float, default=0.0, help="Override playback FPS (0 = use video FPS).")
    args = parser.parse_args()

    reg = RegionRegistry.from_file(args.regions_config)

    H = _load_homography(args.calibration)
    H_inv = None
    if H is not None:
        H_inv = np.linalg.inv(H)

    if args.play or args.output_video:
        if not args.video:
            raise SystemExit("--play/--output-video require --video")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video: {args.video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, args.frame_index)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        play_fps = float(args.play_fps) if args.play_fps and args.play_fps > 0 else fps
        delay_ms = int(1000 / max(1.0, play_fps)) if play_fps > 0 else 1

        writer = None
        if args.output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                raise SystemExit("Failed to read video resolution for output writer.")
            out_path = Path(args.output_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), fourcc, max(play_fps, 1.0), (width, height))

        frames_rendered = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _overlay_regions(frame, reg, H_inv)
            if writer is not None:
                writer.write(frame)
            if args.play or args.show:
                cv2.imshow("Region Overlay (Video)", frame)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break
            frames_rendered += 1
            if args.max_frames and frames_rendered >= args.max_frames:
                break
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved overlay video: {args.output_video}")
        if args.play or args.show:
            cv2.destroyAllWindows()
        return

    frame = _load_frame(video=args.video, image=args.image, frame_index=args.frame_index)
    _overlay_regions(frame, reg, H_inv)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"Saved overlay: {out}")

    if args.show:
        cv2.imshow("Region Overlay", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
