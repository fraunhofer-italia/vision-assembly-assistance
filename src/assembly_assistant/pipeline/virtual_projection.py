from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from assembly_assistant.trajectory.frame_processor import FramePreprocessor


_CAD_SPRITE_CACHE: dict[tuple, np.ndarray | None] = {}
_WARPED_SPRITE_CACHE: dict[tuple, np.ndarray] = {}
_COMPOSITE_SPRITE_CACHE: dict[tuple, np.ndarray] = {}
_MAX_WARPED_SPRITE_CACHE_ITEMS = 256


@dataclass
class VirtualProjectionGuide:
    config: dict
    base_dir: Path = Path(".")
    _sprite_cache: dict | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "VirtualProjectionGuide":
        p = Path(path)
        with p.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data, base_dir=p.parent, _sprite_cache={})

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def draw(
        self,
        frame: np.ndarray,
        *,
        pre: FramePreprocessor,
        region_registry,
        stage_descriptor: dict | None,
        operation_mode: str,
    ) -> None:
        if not self.enabled or region_registry is None:
            return
        stage_name = str((stage_descriptor or {}).get("stage_name", "")).strip()
        if not stage_name:
            return

        stage_cfg = ((self.config.get("stages") or {}).get(stage_name)) or {}
        items = list(stage_cfg.get("items") or [])
        if not items:
            items = self._fallback_items(stage_descriptor, region_registry, operation_mode)
        if not items:
            return

        try:
            h_inv = np.linalg.inv(pre.H)
        except np.linalg.LinAlgError:
            h_inv = None

        default_alpha = float(self.config.get("default_alpha", 0.28))
        default_color = _as_bgr(self.config.get("default_color_bgr", [80, 255, 80]))
        cad_defaults = dict(self.config.get("cad_defaults") or {})
        for item in items:
            rid = str(item.get("target_region", "")).strip()
            region = region_registry.regions.get(rid)
            if region is None:
                continue
            pts = _region_points_px(region, h_inv)
            if not pts:
                continue
            color = _as_bgr(item.get("color_bgr", default_color))
            alpha = float(item.get("alpha", default_alpha))
            shape = str(item.get("shape", "region_fill")).strip().lower()
            label = str(item.get("label", item.get("part_label", ""))).strip()
            item_with_context = dict(cad_defaults)
            item_with_context.update(item)
            item_with_context["_config_base_dir"] = str(self.base_dir)
            _draw_item(frame, pts, shape=shape, color=color, alpha=alpha, label=label, item=item_with_context)

    def _fallback_items(self, stage_descriptor: dict | None, region_registry, operation_mode: str) -> list[dict]:
        stage_name = str((stage_descriptor or {}).get("stage_name", "")).strip()
        mode_rules = ((region_registry.stage_bindings.get(stage_name) or {}).get(str(operation_mode).lower()) or {})
        target_regions = list(mode_rules.get("valid_target_regions") or [])
        expected_parts = list((stage_descriptor or {}).get("expected_parts") or [])
        if not target_regions or not expected_parts:
            return []
        return [
            {
                "part_label": part,
                "target_region": target_regions[0],
                "shape": _default_shape_for_part(str(part)),
                "label": str(part),
            }
            for part in expected_parts
            if str(part).strip() and str(part).strip() != "NONE"
        ]


def _as_bgr(value: Any) -> tuple[int, int, int]:
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(max(0, min(255, v))) for v in value)
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(max(0, min(255, v))) for v in value)
    return (80, 255, 80)


def _project_norm_to_image(h_inv: np.ndarray | None, x: float, y: float):
    if h_inv is None:
        return None
    p = h_inv @ np.array([float(x), float(y), 1.0], dtype=float)
    if abs(float(p[2])) < 1e-9:
        return None
    return [int(round(float(p[0] / p[2]))), int(round(float(p[1] / p[2])))]


def _region_points_px(region, h_inv: np.ndarray | None) -> list[list[int]] | None:
    if region.coordinate_frame == "image_px":
        return [[int(round(float(x))), int(round(float(y)))] for x, y in region.polygon]
    if region.coordinate_frame == "workspace_norm":
        pts = []
        for x, y in region.polygon:
            pt = _project_norm_to_image(h_inv, float(x), float(y))
            if pt is None:
                return None
            pts.append(pt)
        return pts
    return None


def _default_shape_for_part(part_label: str) -> str:
    label = part_label.strip().lower()
    if "tie_rod_nut" in label or label == "nut":
        return "corner_circles"
    if "tie_rod" in label:
        return "four_rods"
    if "rod" in label:
        return "capsule"
    if "tube" in label:
        return "capsule"
    if label in {"piston", "screw"}:
        return "center_circle"
    return "region_fill"


def _draw_item(
    frame: np.ndarray,
    pts: list[list[int]],
    *,
    shape: str,
    color: tuple[int, int, int],
    alpha: float,
    label: str,
    item: dict,
) -> None:
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    alpha = max(0.0, min(1.0, float(alpha)))

    if shape == "corner_circles":
        _draw_corner_circles(overlay, pts, color, item)
    elif shape == "cad_top_view":
        if _draw_cad_top_view(frame, pts, item=item, alpha=alpha):
            if label:
                _draw_label(frame, pts, label, color)
            return
        _draw_primitive_fallback(overlay, pts, color, item)
    elif shape == "cad_corner_instances":
        if _draw_cad_corner_instances(frame, pts, item=item, alpha=alpha):
            if label:
                _draw_label(frame, pts, label, color)
            return
        _draw_corner_circles(overlay, pts, color, item)
    elif shape == "center_circle":
        _draw_center_circle(overlay, pts, color, item)
    elif shape == "capsule":
        _draw_capsule(overlay, pts, color, item)
    elif shape == "four_rods":
        _draw_four_rods(overlay, pts, color, item)
    elif shape == "outline":
        cv2.polylines(overlay, [arr], isClosed=True, color=color, thickness=int(item.get("thickness", 3)), lineType=cv2.LINE_AA)
    else:
        cv2.fillPoly(overlay, [arr], color)
        cv2.polylines(overlay, [arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        _draw_label(frame, pts, label, color)


def _draw_label(frame, pts: list[list[int]], label: str, color: tuple[int, int, int]) -> None:
    cx, cy = _centroid(pts)
    cv2.putText(
        frame,
        label,
        (int(cx) + 6, int(cy) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_primitive_fallback(overlay, pts: list[list[int]], color: tuple[int, int, int], item: dict) -> None:
    fallback = str(item.get("fallback_shape") or _default_shape_for_part(str(item.get("part_label", "")))).lower()
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    if fallback == "corner_circles":
        _draw_corner_circles(overlay, pts, color, item)
    elif fallback == "center_circle":
        _draw_center_circle(overlay, pts, color, item)
    elif fallback == "capsule":
        _draw_capsule(overlay, pts, color, item)
    elif fallback == "four_rods":
        _draw_four_rods(overlay, pts, color, item)
    elif fallback == "outline":
        cv2.polylines(overlay, [arr], isClosed=True, color=color, thickness=int(item.get("thickness", 3)), lineType=cv2.LINE_AA)
    else:
        cv2.fillPoly(overlay, [arr], color)
        cv2.polylines(overlay, [arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def _centroid(pts: list[list[int]]) -> tuple[float, float]:
    arr = np.array(pts, dtype=np.float32)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


def _min_rect_points(pts: list[list[int]]) -> np.ndarray:
    arr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(arr)
    return cv2.boxPoints(rect).astype(np.float32)


def _draw_corner_circles(overlay, pts, color, item):
    corners = _min_rect_points(pts)
    radius = int(item.get("radius_px", 10))
    for x, y in corners:
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius + 3, color, 2, cv2.LINE_AA)


def _draw_center_circle(overlay, pts, color, item):
    cx, cy = _centroid(pts)
    rect = cv2.minAreaRect(np.array(pts, dtype=np.float32))
    radius = int(item.get("radius_px", max(10.0, min(rect[1]) * 0.20)))
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), radius + 4, color, 2, cv2.LINE_AA)


def _draw_capsule(overlay, pts, color, item):
    rect = cv2.minAreaRect(np.array(pts, dtype=np.float32))
    (cx, cy), (w, h), angle = rect
    length = max(w, h) * float(item.get("length_ratio", 0.82))
    thickness = int(max(8.0, min(w, h) * float(item.get("thickness_ratio", 0.24))))
    theta = np.deg2rad(angle + (0.0 if w >= h else 90.0))
    dx = 0.5 * length * np.cos(theta)
    dy = 0.5 * length * np.sin(theta)
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    radius = max(3, thickness // 2)
    cv2.circle(overlay, p1, radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, p2, radius, color, -1, cv2.LINE_AA)


def _draw_four_rods(overlay, pts, color, item):
    corners = _min_rect_points(pts)
    center = np.mean(corners, axis=0)
    shrink = float(item.get("shrink", 0.16))
    c = corners + (center - corners) * shrink
    thickness = int(item.get("thickness", 8))
    for i in range(4):
        p1 = tuple(np.round(c[i]).astype(int))
        p2 = tuple(np.round(c[(i + 1) % 4]).astype(int))
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_cad_top_view(frame: np.ndarray, pts: list[list[int]], *, item: dict, alpha: float) -> bool:
    sprite = _cad_sprite_for_item(item)
    if sprite is None:
        return False
    fit_scale = float(item.get("fit_scale", item.get("projection_scale", item.get("instance_size_ratio", 1.0))))
    quad = _scale_quad(_order_quad(_min_rect_points(pts)), fit_scale)
    return _warp_rgba_onto_frame(frame, sprite, quad, alpha=float(alpha))


def _draw_cad_corner_instances(frame: np.ndarray, pts: list[list[int]], *, item: dict, alpha: float) -> bool:
    sprite = _cad_sprite_for_item(item)
    if sprite is None:
        return False
    corners = _order_quad(_min_rect_points(pts))
    center = np.mean(corners, axis=0)
    inset = float(item.get("corner_inset", 0.08))
    corners = corners + (center - corners) * inset
    rect = cv2.minAreaRect(np.array(pts, dtype=np.float32))
    size = max(8.0, min(rect[1]) * float(item.get("instance_size_ratio", 0.16)))
    quads = []
    for corner in corners:
        half = 0.5 * size
        quads.append(
            np.array(
                [
                    [corner[0] - half, corner[1] - half],
                    [corner[0] + half, corner[1] - half],
                    [corner[0] + half, corner[1] + half],
                    [corner[0] - half, corner[1] + half],
                ],
                dtype=np.float32,
            )
        )
    composite = _cached_composite_sprite(sprite, quads, frame_size=(frame.shape[1], frame.shape[0]))
    return _blend_warped_rgba_onto_frame(frame, composite, alpha=float(alpha))


def _cad_sprite_for_item(item: dict) -> np.ndarray | None:
    mesh_path = _resolve_mesh_path(item.get("mesh_path"), item.get("_config_base_dir"))
    if mesh_path is None:
        return None
    axes = tuple(item.get("projection_axes", ["auto_top"]))
    view = str(item.get("view", "top")).strip().lower()
    rotate_deg = float(item.get("rotate_deg", 0.0))
    flip_x = bool(item.get("flip_x", False))
    flip_y = bool(item.get("flip_y", False))
    view_axis = item.get("view_axis")
    sprite_size = int(item.get("sprite_size", 512))
    color = _as_bgr(item.get("cad_color_bgr", [160, 210, 220]))
    edge_color = _as_bgr(item.get("cad_edge_bgr", [40, 255, 80]))
    cache_key = (str(mesh_path), axes, view, str(view_axis), rotate_deg, flip_x, flip_y, sprite_size, color, edge_color)
    if cache_key in _CAD_SPRITE_CACHE:
        return _CAD_SPRITE_CACHE[cache_key]
    vertices, faces = _load_obj_mesh(mesh_path)
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        _CAD_SPRITE_CACHE[cache_key] = None
        return None
    axes = _resolve_projection_axes(vertices, axes, view_axis)
    sprite = _rasterize_obj_top_view(
        vertices,
        faces,
        axes=axes,
        rotate_deg=rotate_deg,
        flip_x=flip_x,
        flip_y=flip_y,
        size=sprite_size,
        color_bgr=color,
        edge_bgr=edge_color,
    )
    _CAD_SPRITE_CACHE[cache_key] = sprite
    return sprite


def _resolve_mesh_path(raw_path, config_base_dir) -> Path | None:
    if not raw_path:
        return None
    p = Path(str(raw_path)).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        if config_base_dir:
            candidates.append(Path(str(config_base_dir)) / p)
            candidates.append(Path(str(config_base_dir)).parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_obj_mesh(path: Path):
    vertices = []
    faces = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    raw = line.strip().split()[1:]
                    face = []
                    for token in raw:
                        if not token:
                            continue
                        idx = token.split("/")[0]
                        if not idx:
                            continue
                        vi = int(idx)
                        if vi < 0:
                            vi = len(vertices) + vi
                        else:
                            vi = vi - 1
                        face.append(vi)
                    if len(face) >= 3:
                        faces.append(face)
    except Exception:
        return None, None
    return np.array(vertices, dtype=np.float32), faces


def _rasterize_obj_top_view(
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    axes: tuple,
    rotate_deg: float,
    flip_x: bool,
    flip_y: bool,
    size: int,
    color_bgr: tuple[int, int, int],
    edge_bgr: tuple[int, int, int],
) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
    ax0 = axis_map.get(str(axes[0]).lower(), 0) if len(axes) >= 1 else 0
    ax1 = axis_map.get(str(axes[1]).lower(), 1) if len(axes) >= 2 else 1
    pts2 = vertices[:, [ax0, ax1]].astype(np.float32)
    if str(axes[0]).lower().startswith("-"):
        pts2[:, 0] *= -1.0
    if len(axes) >= 2 and str(axes[1]).lower().startswith("-"):
        pts2[:, 1] *= -1.0
    if flip_x:
        pts2[:, 0] *= -1.0
    if flip_y:
        pts2[:, 1] *= -1.0
    if abs(float(rotate_deg)) > 1e-6:
        theta = np.deg2rad(float(rotate_deg))
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        pts2 = (pts2 - np.mean(pts2, axis=0)) @ rot.T

    lo = np.min(pts2, axis=0)
    hi = np.max(pts2, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    margin = max(8.0, float(size) * 0.08)
    scale = min((float(size) - 2.0 * margin) / span[0], (float(size) - 2.0 * margin) / span[1])
    pix = (pts2 - lo) * scale
    pix[:, 1] = (span[1] * scale) - pix[:, 1]
    used = np.max(pix, axis=0) - np.min(pix, axis=0)
    offset = np.array([(size - used[0]) * 0.5, (size - used[1]) * 0.5], dtype=np.float32) - np.min(pix, axis=0)
    pix = pix + offset

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    depth_axis = next((i for i in [2, 1, 0] if i not in {ax0, ax1}), 2)
    ordered_faces = sorted(faces, key=lambda f: float(np.mean(vertices[np.array(f, dtype=int), depth_axis])))
    base = np.array(color_bgr, dtype=np.float32)
    for face in ordered_faces:
        idx = np.array(face, dtype=int)
        if np.any(idx < 0) or np.any(idx >= len(pix)):
            continue
        poly = np.round(pix[idx]).astype(np.int32)
        shade = _face_shade(vertices[idx], depth_axis)
        bgr = np.clip(base * shade, 0, 255).astype(np.uint8).tolist()
        cv2.fillConvexPoly(rgba, poly, [int(bgr[0]), int(bgr[1]), int(bgr[2]), 230], cv2.LINE_AA)
        cv2.polylines(rgba, [poly.reshape((-1, 1, 2))], True, [edge_bgr[0], edge_bgr[1], edge_bgr[2], 255], 1, cv2.LINE_AA)
    return rgba


def _resolve_projection_axes(vertices: np.ndarray, axes: tuple, view_axis=None) -> tuple[str, str]:
    axis_names = ["x", "y", "z"]
    if view_axis:
        raw = str(view_axis).strip().lower().replace("+", "")
        sign = "-" if raw.startswith("-") else ""
        axis = raw[1:] if raw.startswith("-") else raw
        if axis in axis_names:
            keep = [name for name in axis_names if name != axis]
            return tuple(keep) if sign != "-" else (keep[1], keep[0])

    raw_axes = [str(a).strip().lower() for a in axes]
    if raw_axes and raw_axes[0] in {"auto", "auto_top", "top"}:
        ext = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        order = np.argsort(ext)
        # Top-view heuristic: look along the thinnest dimension, keep the two widest axes.
        keep = [axis_names[int(i)] for i in order[1:]]
        return keep[0], keep[1]

    if len(raw_axes) >= 2:
        return raw_axes[0], raw_axes[1]
    if len(raw_axes) == 1:
        axis = raw_axes[0].replace("+", "")
        sign = "-" if axis.startswith("-") else ""
        axis = axis[1:] if axis.startswith("-") else axis
        if axis in axis_names:
            keep = [name for name in axis_names if name != axis]
            return (keep[0], keep[1]) if sign != "-" else (keep[1], keep[0])
    return "x", "y"


def _face_shade(face_vertices: np.ndarray, depth_axis: int) -> float:
    if len(face_vertices) < 3:
        return 1.0
    v0, v1, v2 = face_vertices[0], face_vertices[1], face_vertices[2]
    n = np.cross(v1 - v0, v2 - v0)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-9:
        return 0.9
    n = n / norm
    return float(0.72 + 0.28 * abs(n[depth_axis]))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _scale_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    scale = max(0.05, float(scale))
    center = np.mean(quad, axis=0, keepdims=True)
    return center + (quad - center) * scale


def _warp_rgba_onto_frame(frame: np.ndarray, sprite_rgba: np.ndarray, dst_quad: np.ndarray, alpha: float) -> bool:
    if sprite_rgba is None or sprite_rgba.size == 0:
        return False
    h, w = frame.shape[:2]
    sh, sw = sprite_rgba.shape[:2]
    warped = _cached_warped_sprite(sprite_rgba, dst_quad, frame_size=(w, h), sprite_size=(sw, sh))
    return _blend_warped_rgba_onto_frame(frame, warped, alpha=float(alpha))


def _blend_warped_rgba_onto_frame(frame: np.ndarray, warped: np.ndarray, alpha: float) -> bool:
    mask = warped[:, :, 3:4].astype(np.float32) / 255.0
    mask *= max(0.0, min(1.0, float(alpha)))
    if float(np.max(mask)) <= 0.0:
        return False
    rgb = warped[:, :, :3].astype(np.float32)
    base = frame.astype(np.float32)
    out = rgb * mask + base * (1.0 - mask)
    frame[:] = np.clip(out, 0, 255).astype(np.uint8)
    return True


def _cached_warped_sprite(
    sprite_rgba: np.ndarray,
    dst_quad: np.ndarray,
    *,
    frame_size: tuple[int, int],
    sprite_size: tuple[int, int],
) -> np.ndarray:
    sw, sh = sprite_size
    w, h = frame_size
    quad_key = tuple(int(round(float(v))) for v in np.array(dst_quad, dtype=np.float32).reshape(-1))
    cache_key = (id(sprite_rgba), sw, sh, w, h, quad_key)
    cached = _WARPED_SPRITE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    src_quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_quad, np.array(dst_quad, dtype=np.float32))
    warped = cv2.warpPerspective(sprite_rgba, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if len(_WARPED_SPRITE_CACHE) >= _MAX_WARPED_SPRITE_CACHE_ITEMS:
        _WARPED_SPRITE_CACHE.clear()
    _WARPED_SPRITE_CACHE[cache_key] = warped
    return warped


def _cached_composite_sprite(
    sprite_rgba: np.ndarray,
    dst_quads: list[np.ndarray],
    *,
    frame_size: tuple[int, int],
) -> np.ndarray:
    h, w = sprite_rgba.shape[:2]
    frame_w, frame_h = frame_size
    quads_key = tuple(
        tuple(int(round(float(v))) for v in np.array(quad, dtype=np.float32).reshape(-1))
        for quad in dst_quads
    )
    cache_key = (id(sprite_rgba), w, h, frame_w, frame_h, quads_key)
    cached = _COMPOSITE_SPRITE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    composite = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    for quad in dst_quads:
        warped = _cached_warped_sprite(sprite_rgba, quad, frame_size=frame_size, sprite_size=(w, h))
        _alpha_composite_rgba(composite, warped)

    if len(_COMPOSITE_SPRITE_CACHE) >= _MAX_WARPED_SPRITE_CACHE_ITEMS:
        _COMPOSITE_SPRITE_CACHE.clear()
    _COMPOSITE_SPRITE_CACHE[cache_key] = composite
    return composite


def _alpha_composite_rgba(base: np.ndarray, overlay: np.ndarray) -> None:
    overlay_alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    if float(np.max(overlay_alpha)) <= 0.0:
        return
    base_alpha = base[:, :, 3:4].astype(np.float32) / 255.0
    out_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)
    out_rgb = (
        overlay[:, :, :3].astype(np.float32) * overlay_alpha
        + base[:, :, :3].astype(np.float32) * base_alpha * (1.0 - overlay_alpha)
    ) / np.maximum(out_alpha, 1e-6)
    base[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    base[:, :, 3:4] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8)
