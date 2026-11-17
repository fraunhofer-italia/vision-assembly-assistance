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

from assembly_assistant.trajectory.frame_processor import FramePreprocessor


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


def _annotate_one_region(frame, *, region_id: str, index: int, total: int) -> list[list[int]]:
    points: list[list[int]] = []
    window = "Annotate Polygon (left-click add, u undo, r reset, enter finish, q quit)"

    def redraw() -> None:
        vis = frame.copy()
        title = f"{region_id} [{index}/{total}]"
        cv2.putText(vis, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(vis, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(vis, tuple(points[i]), tuple(points[i + 1]), (255, 255, 0), 2)
        if len(points) >= 3:
            cv2.line(vis, tuple(points[-1]), tuple(points[0]), (80, 255, 80), 1)
        cv2.imshow(window, vis)

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter
            if len(points) < 3:
                print("Need at least 3 points.")
                continue
            break
        if key == ord("u") and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            raise SystemExit(0)

    cv2.destroyAllWindows()
    return points


def _to_norm(points_px: list[list[int]], pre: FramePreprocessor | None):
    if pre is None:
        return None
    out = []
    for x, y in points_px:
        wx, wy = pre.warp_point((float(x), float(y)))
        out.append([float(wx), float(wy)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Click polygon vertices on a video/image frame to get polygon coordinates. "
            "Supports image_px output and workspace_norm output via homography."
        )
    )
    parser.add_argument("--video", default=None, help="Path to video file (uses first frame unless --frame-index set).")
    parser.add_argument("--image", default=None, help="Path to image file.")
    parser.add_argument("--frame-index", type=int, default=100, help="Frame index for --video.")

    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument("--region-id", default=None, help="Single region identifier for output snippet.")
    region_group.add_argument(
        "--region-ids",
        default=None,
        help="Comma-separated region identifiers for bulk annotation in one session.",
    )

    parser.add_argument(
        "--coordinate-frame",
        choices=["image_px", "workspace_norm"],
        default="image_px",
        help="Output coordinate frame for polygon.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Calibration dir or JSON containing homography. "
            "Required for --coordinate-frame workspace_norm; optional for image_px to also print polygon_norm."
        ),
    )
    parser.add_argument(
        "--update-workspace-layout",
        default=None,
        help=(
            "Optional workspace layout JSON path to update captured polygon. "
            "workspace_norm -> workspace.zones.<region-id>; "
            "image_px -> part_boxes_image_px[region-id].polygon_px"
        ),
    )
    parser.add_argument(
        "--update-regions-config",
        default=None,
        help=(
            "Optional regions JSON path to update the matching region polygon. "
            "workspace_norm -> polygon_norm; image_px -> polygon_px."
        ),
    )
    parser.add_argument(
        "--create-missing-region",
        action="store_true",
        help="When --update-regions-config is used, create a missing region with role=projection_target.",
    )
    parser.add_argument("--output", default=None, help="Optional output .json path for snippet.")
    args = parser.parse_args()

    if args.coordinate_frame == "workspace_norm" and not args.calibration:
        raise SystemExit("--calibration is required when --coordinate-frame workspace_norm")

    region_ids = []
    if args.region_id:
        region_ids = [str(args.region_id).strip()]
    else:
        region_ids = [x.strip() for x in str(args.region_ids or "").split(",") if x.strip()]
    if not region_ids:
        raise SystemExit("No valid region ids provided.")

    frame = _load_frame(video=args.video, image=args.image, frame_index=args.frame_index)
    pre = FramePreprocessor(args.calibration) if args.calibration else None

    workspace_layout_path = Path(args.update_workspace_layout) if args.update_workspace_layout else None
    regions_config_path = Path(args.update_regions_config) if args.update_regions_config else None
    workspace_layout = (
        json.loads(workspace_layout_path.read_text(encoding="utf-8"))
        if workspace_layout_path is not None
        else None
    )
    regions_config = (
        json.loads(regions_config_path.read_text(encoding="utf-8"))
        if regions_config_path is not None
        else None
    )

    snippets = []
    total = len(region_ids)
    for idx, rid in enumerate(region_ids, start=1):
        points_px = _annotate_one_region(frame, region_id=rid, index=idx, total=total)
        points_norm = _to_norm(points_px, pre)

        if args.coordinate_frame == "workspace_norm":
            if points_norm is None:
                raise SystemExit("Internal error: expected normalized points when coordinate-frame is workspace_norm")
            snippet = {
                "region_id": rid,
                "coordinate_frame": "workspace_norm",
                "polygon_norm": points_norm,
            }
        else:
            snippet = {
                "region_id": rid,
                "coordinate_frame": "image_px",
                "polygon_px": points_px,
            }
            if points_norm is not None:
                snippet["polygon_norm_preview"] = points_norm

        snippets.append(snippet)
        print(json.dumps(snippet, indent=2))

        if workspace_layout is not None:
            if args.coordinate_frame == "workspace_norm":
                zones = (((workspace_layout.get("workspace") or {}).get("zones")) or {})
                zones[str(rid)] = points_norm
                workspace_layout.setdefault("workspace", {})["zones"] = zones
                print(f"Staged workspace zone update: workspace.zones.{rid}")
            else:
                boxes = list(workspace_layout.get("part_boxes_image_px") or [])
                updated = False
                for box in boxes:
                    if str(box.get("region_id", "")) == str(rid):
                        box["polygon_px"] = points_px
                        updated = True
                        break
                if not updated:
                    boxes.append({"region_id": str(rid), "polygon_px": points_px})
                workspace_layout["part_boxes_image_px"] = boxes
                print(f"Staged workspace part-box update: part_boxes_image_px[{rid}]")

        if regions_config is not None:
            regions = list(regions_config.get("regions") or [])
            target = None
            for r in regions:
                if str(r.get("region_id", "")) == str(rid):
                    target = r
                    break
            if target is None:
                if not args.create_missing_region:
                    raise SystemExit(
                        f"Region '{rid}' not found in {regions_config_path}. "
                        "Create the region first or pass --create-missing-region."
                    )
                target = {
                    "region_id": str(rid),
                    "roles": ["projection_target"],
                    "notes": "Created by annotate_region_polygons.py.",
                }
                regions.append(target)
            if args.coordinate_frame == "workspace_norm":
                target["coordinate_frame"] = "workspace_norm"
                target["polygon_norm"] = points_norm
                target.pop("polygon_px", None)
            else:
                target["coordinate_frame"] = "image_px"
                target["polygon_px"] = points_px
            regions_config["regions"] = regions
            print(f"Staged regions config update: regions[{rid}]")

    if workspace_layout_path is not None:
        workspace_layout_path.write_text(json.dumps(workspace_layout, indent=2), encoding="utf-8")
        print(f"Updated workspace layout: {workspace_layout_path}")

    if regions_config_path is not None:
        regions_config_path.write_text(json.dumps(regions_config, indent=2), encoding="utf-8")
        print(f"Updated regions config: {regions_config_path}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = snippets[0] if len(snippets) == 1 else {"regions": snippets}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
