from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Region:
    region_id: str
    coordinate_frame: str
    polygon: List[List[float]]
    roles: List[str]
    mode_semantics: Dict[str, Dict[str, bool]]
    detector: Dict[str, object]


class RegionRegistry:
    """
    Config-driven region registry.
    Core validation is rigid; coordinates/semantics are configurable.
    """

    def __init__(self, registry_id: str, modes: List[str], regions: Dict[str, Region], stage_bindings: dict):
        self.registry_id = str(registry_id)
        self.modes = [str(m).lower() for m in modes]
        self.regions = regions
        self.stage_bindings = stage_bindings

    @classmethod
    def from_file(cls, path: str | Path) -> "RegionRegistry":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "RegionRegistry":
        registry_id = data.get("registry_id")
        if not registry_id:
            raise ValueError("regions registry requires 'registry_id'")

        modes = data.get("modes") or []
        if not isinstance(modes, list) or not modes:
            raise ValueError("regions registry requires non-empty 'modes' list")

        regions_raw = data.get("regions") or []
        if not isinstance(regions_raw, list) or not regions_raw:
            raise ValueError("regions registry requires non-empty 'regions' list")

        regions: Dict[str, Region] = {}
        for r in regions_raw:
            region_id = str(r.get("region_id", "")).strip()
            if not region_id:
                raise ValueError("each region requires non-empty 'region_id'")
            if region_id in regions:
                raise ValueError(f"duplicate region_id: {region_id}")

            coordinate_frame = str(r.get("coordinate_frame", "workspace_norm")).strip().lower()
            if coordinate_frame == "workspace_norm":
                polygon = r.get("polygon_norm") if r.get("polygon_norm") is not None else r.get("polygon")
            elif coordinate_frame == "image_px":
                polygon = r.get("polygon_px") if r.get("polygon_px") is not None else r.get("polygon")
            else:
                raise ValueError(
                    f"region '{region_id}' has unsupported coordinate_frame '{coordinate_frame}'. "
                    "Use 'workspace_norm' or 'image_px'."
                )
            _validate_polygon_with_frame(region_id, coordinate_frame, polygon)

            roles = [str(x) for x in (r.get("roles") or [])]
            if not roles:
                raise ValueError(f"region '{region_id}' requires non-empty roles")

            mode_semantics = r.get("mode_semantics") or {}
            detector = r.get("detector") or {}
            if detector and not isinstance(detector, dict):
                raise ValueError(f"region '{region_id}' field 'detector' must be a dict when provided")
            regions[region_id] = Region(
                region_id=region_id,
                coordinate_frame=coordinate_frame,
                polygon=polygon,
                roles=roles,
                mode_semantics=mode_semantics,
                detector=detector,
            )

        stage_bindings = data.get("stage_bindings") or {}
        if not isinstance(stage_bindings, dict):
            raise ValueError("'stage_bindings' must be a dict")

        reg = cls(
            registry_id=registry_id,
            modes=modes,
            regions=regions,
            stage_bindings=stage_bindings,
        )
        reg._validate_stage_bindings()
        return reg

    def _validate_stage_bindings(self) -> None:
        valid_regions = set(self.regions.keys())
        valid_modes = set(self.modes)
        for stage_name, mode_map in self.stage_bindings.items():
            if not isinstance(mode_map, dict):
                raise ValueError(f"stage_bindings['{stage_name}'] must map modes to rule dicts")
            for mode_name, rules in mode_map.items():
                mode_key = str(mode_name).lower()
                if mode_key not in valid_modes:
                    raise ValueError(
                        f"stage '{stage_name}' has unsupported mode '{mode_name}'. Expected one of {sorted(valid_modes)}"
                    )
                if not isinstance(rules, dict):
                    raise ValueError(f"stage '{stage_name}' mode '{mode_name}' rules must be a dict")
                for key in ("valid_source_regions", "valid_target_regions", "valid_tool_regions"):
                    ids = rules.get(key, [])
                    if not isinstance(ids, list):
                        raise ValueError(f"stage '{stage_name}' mode '{mode_name}' field '{key}' must be a list")
                    unknown = [x for x in ids if x not in valid_regions]
                    if unknown:
                        raise ValueError(
                            f"stage '{stage_name}' mode '{mode_name}' field '{key}' has unknown region ids: {unknown}"
                        )

    def is_region_allowed(self, stage_name: str, mode: str, role_bucket: str, region_id: str) -> bool:
        stage = self.stage_bindings.get(stage_name) or {}
        mode_rules = stage.get(str(mode).lower()) or stage.get(mode) or {}
        allowed = mode_rules.get(role_bucket, [])
        return region_id in allowed

    def resolve_role_enabled(self, region_id: str, mode: str, role: str) -> bool:
        """
        Returns whether a role is semantically enabled for a region in the given mode.
        If no mode_semantics is provided, falls back to static role presence.
        """
        region = self.regions.get(region_id)
        if region is None:
            return False
        if role not in region.roles:
            return False
        if not region.mode_semantics:
            return True
        mode_rules = region.mode_semantics.get(str(mode).lower()) or region.mode_semantics.get(mode) or {}
        if not mode_rules:
            return True
        return bool(mode_rules.get(role, False))


def _validate_polygon(region_id: str, polygon: Optional[list]) -> None:
    _validate_polygon_with_frame(region_id, "workspace_norm", polygon)


def _validate_polygon_with_frame(region_id: str, coordinate_frame: str, polygon: Optional[list]) -> None:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"region '{region_id}' polygon must contain at least 3 points")
    for idx, pt in enumerate(polygon):
        if not isinstance(pt, list) or len(pt) != 2:
            raise ValueError(f"region '{region_id}' polygon point[{idx}] must be [x,y]")
        x, y = pt
        if coordinate_frame == "workspace_norm":
            x = float(x)
            y = float(y)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"region '{region_id}' polygon point[{idx}] out of normalized bounds [0,1]: [{x},{y}]"
                )
        elif coordinate_frame == "image_px":
            x = float(x)
            y = float(y)
            if x < 0.0 or y < 0.0:
                raise ValueError(
                    f"region '{region_id}' polygon point[{idx}] must be non-negative in image_px frame: [{x},{y}]"
                )
        else:
            raise ValueError(
                f"region '{region_id}' has unsupported coordinate_frame '{coordinate_frame}'. "
                "Use 'workspace_norm' or 'image_px'."
            )
