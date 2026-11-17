#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.pipeline.region_registry import RegionRegistry
from assembly_assistant.pipeline.virtual_projection import VirtualProjectionGuide
from assembly_assistant.trajectory.frame_processor import FramePreprocessor


VIEW_CHOICES = [
    ["z"],
    ["x"],
    ["y"],
    ["auto_top"],
    ["x", "y"],
    ["x", "z"],
    ["y", "z"],
    ["-z"],
    ["-x"],
    ["-y"],
]


def _load_frame(*, video: str | None, image: str | None, frame_index: int):
    if bool(video) == bool(image):
        raise SystemExit("Provide exactly one input: --video or --image")
    if image:
        frame = cv2.imread(str(image))
        if frame is None:
            raise SystemExit(f"Failed to read image: {image}")
        return frame
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Failed to read frame {frame_index} from video: {video}")
    return frame


def _find_item(config: dict, stage_name: str, part_label: str | None, item_index: int | None):
    stage = (config.get("stages") or {}).get(stage_name)
    if not isinstance(stage, dict):
        raise SystemExit(f"Stage not found in projection config: {stage_name}")
    items = list(stage.get("items") or [])
    if not items:
        raise SystemExit(f"Stage has no projection items: {stage_name}")
    if item_index is not None:
        if item_index < 0 or item_index >= len(items):
            raise SystemExit(f"--item-index out of range [0,{len(items) - 1}]")
        return stage, items[item_index], item_index
    if part_label:
        for idx, item in enumerate(items):
            if str(item.get("part_label", "")).strip() == str(part_label).strip():
                return stage, item, idx
        raise SystemExit(f"No item with part_label={part_label} under stage={stage_name}")
    return stage, items[0], 0


def _apply_defaults(config: dict, item: dict) -> None:
    defaults = dict(config.get("cad_defaults") or {})
    for key in ("projection_axes", "rotate_deg", "flip_x", "flip_y", "fit_scale"):
        if key not in item and key in defaults:
            item[key] = defaults[key]
    item.setdefault("projection_axes", ["z"])
    item.setdefault("rotate_deg", 0)
    item.setdefault("flip_x", False)
    item.setdefault("flip_y", False)
    if str(item.get("shape", "")) == "cad_corner_instances":
        item.setdefault("instance_size_ratio", 0.22)
    else:
        item.setdefault("fit_scale", 1.0)


def _render(frame, *, config, stage_name, region_registry, pre, operation_mode):
    vis = frame.copy()
    guide = VirtualProjectionGuide(config, base_dir=Path("."))
    guide.draw(
        vis,
        pre=pre,
        region_registry=region_registry,
        stage_descriptor={"stage_name": stage_name},
        operation_mode=operation_mode,
    )
    return vis


def _draw_hud(frame, item: dict, stage_name: str, item_index: int) -> None:
    lines = [
        f"{stage_name} item={item_index} part={item.get('part_label')}",
        f"view={item.get('projection_axes')} rot={item.get('rotate_deg')} flip_x={item.get('flip_x')} flip_y={item.get('flip_y')}",
        f"fit_scale={item.get('fit_scale', '-')} instance_size_ratio={item.get('instance_size_ratio', '-') } alpha={item.get('alpha', '-')}",
        "keys: v view | r/R rotate | x/y flip | +/- scale | [/ ] alpha | enter/s save | q quit",
    ]
    x, y = 16, 26
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 120), 1, cv2.LINE_AA)
        y += 24


def _cycle_view(item: dict) -> None:
    current = list(item.get("projection_axes") or ["z"])
    try:
        idx = VIEW_CHOICES.index(current)
    except ValueError:
        idx = -1
    item["projection_axes"] = VIEW_CHOICES[(idx + 1) % len(VIEW_CHOICES)]


def _adjust_scale(item: dict, delta: float) -> None:
    if str(item.get("shape", "")) == "cad_corner_instances":
        key = "instance_size_ratio"
    else:
        key = "fit_scale"
    item[key] = round(max(0.05, float(item.get(key, 1.0)) + float(delta)), 3)


def _adjust_alpha(item: dict, delta: float) -> None:
    item["alpha"] = round(max(0.02, min(1.0, float(item.get("alpha", 0.42)) + float(delta))), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively tune one virtual CAD projection item and save it to JSON.")
    parser.add_argument("--projection-config", default="configs/virtual_projections_v2.json")
    parser.add_argument("--regions-config", default="configs/regions_v2.json")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--part-label", default=None)
    parser.add_argument("--item-index", type=int, default=None)
    parser.add_argument("--operation-mode", default="assembly", choices=["assembly", "dismantle"])
    parser.add_argument("--video", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--frame-index", type=int, default=100)
    parser.add_argument("--output", default=None, help="Defaults to overwriting --projection-config.")
    args = parser.parse_args()

    config_path = Path(args.projection_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage, item, item_index = _find_item(config, args.stage_name, args.part_label, args.item_index)
    _apply_defaults(config, item)
    stage["items"][item_index] = item

    frame = _load_frame(video=args.video, image=args.image, frame_index=args.frame_index)
    region_registry = RegionRegistry.from_file(args.regions_config)
    pre = FramePreprocessor(args.calibration)

    window = "Tune Virtual Projection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        single_item_config = dict(config)
        single_item_config["stages"] = {args.stage_name: {"items": [item]}}
        vis = _render(
            frame,
            config=single_item_config,
            stage_name=args.stage_name,
            region_registry=region_registry,
            pre=pre,
            operation_mode=args.operation_mode,
        )
        _draw_hud(vis, item, args.stage_name, item_index)
        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, ord("s")):
            out = Path(args.output) if args.output else config_path
            out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"Saved: {out}")
            break
        if key in (27, ord("q")):
            print("Quit without saving.")
            break
        if key == ord("v"):
            _cycle_view(item)
        elif key == ord("r"):
            item["rotate_deg"] = round(float(item.get("rotate_deg", 0.0)) + 90.0, 3)
        elif key == ord("R"):
            item["rotate_deg"] = round(float(item.get("rotate_deg", 0.0)) - 90.0, 3)
        elif key == ord("x"):
            item["flip_x"] = not bool(item.get("flip_x", False))
        elif key == ord("y"):
            item["flip_y"] = not bool(item.get("flip_y", False))
        elif key in (ord("+"), ord("=")):
            _adjust_scale(item, 0.05)
        elif key in (ord("-"), ord("_")):
            _adjust_scale(item, -0.05)
        elif key == ord("]"):
            _adjust_alpha(item, 0.04)
        elif key == ord("["):
            _adjust_alpha(item, -0.04)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
