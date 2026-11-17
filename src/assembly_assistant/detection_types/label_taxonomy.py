from __future__ import annotations

from typing import Dict, Iterable, Optional


PART_LABELS = {
    "front_cover",
    "rear_cover",
    "cylinder_tube",
    "piston",
    "piston_rod",
    "tie_rod_assembly",
    "bttm_tie_rod_nut",
    "nut",
    "screw",
}

TOOL_LABELS = {
    "screwdriver_small_l",
    "screwdriver_large_l",
}

LABEL_ALIASES = {
    "ScrewdriverS": "screwdriver_small_l",
    "ScrewdriverL": "screwdriver_large_l",
    "small_l_screwdriver": "screwdriver_small_l",
    "large_l_screwdriver": "screwdriver_large_l",
}


def canonical_label(label: Optional[str]) -> str:
    value = str(label or "").strip()
    if not value:
        return "NONE"
    return LABEL_ALIASES.get(value, value)


def label_category(label: Optional[str]) -> str:
    name = canonical_label(label)
    if name in TOOL_LABELS:
        return "tool"
    if name in PART_LABELS:
        return "part"
    return "unknown"


def is_tool_label(label: Optional[str]) -> bool:
    return label_category(label) == "tool"


def is_part_label(label: Optional[str]) -> bool:
    return label_category(label) == "part"


def build_inventory(items: Iterable[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        label = canonical_label(item.get("label", item.get("class", "NONE")))
        counts[label] = int(counts.get(label, 0)) + 1
    return counts
