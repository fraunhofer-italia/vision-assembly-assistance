from __future__ import annotations

from textwrap import wrap


POSE_MESSAGES = {
    "correct_pose": "Correct orientation",
    "flipped_pose": "Flipped: turn the part over",
    "side_pose": "Sideways: rotate the part so the correct face is up",
    "horizontal_left": "Place the rod horizontally with the tip pointing left",
    "horizontal_right": "Place the rod horizontally with the tip pointing right",
    "vertical_up": "Hold the rod upright",
    "vertical_down": "Hold the rod upside down",
}


OBJECT_MESSAGES = {
    "bttm_tie_rod_nut": "Bottom tie rod nuts",
    "rear_cover": "Rear cover",
    "cylinder_tube": "Cylinder tube",
    "piston_rod": "Piston rod",
    "piston": "Piston",
    "screw": "Mounting screw",
    "front_cover": "Front cover",
    "tie_rod_assembly": "Tie rod assemblies",
    "nut": "Top nuts",
    "NONE": "No tool or part required",
}


REGION_MESSAGES = {
    "box_bottom_tie_rod_nuts": "Pick the bottom tie rod nuts",
    "box_rear_cover": "Pick the rear cover",
    "box_cylinder_tube": "Pick the cylinder tube",
    "box_piston_rod": "Pick the piston rod",
    "box_piston": "Pick the piston",
    "box_mounting_screw": "Pick the mounting screw",
    "box_tools_pair": "Pick or return the required tool",
    "box_front_cover": "Pick the front cover",
    "box_tie_rod_assemblies": "Pick the tie rod assemblies",
    "box_piston_rod_nut": "Pick the top nuts",
    "square_holder_zone": "Place the part on the square holder",
    "rod_holder_zone": "Place the part on the rod holder",
    "assembly_target": "Work in the assembly target area",
    "square_base": "Square base guide",
    "rod_base": "Rod base guide",
}


STAGE_TITLE_OVERRIDES = {
    "A01_place_bottom_tie_rod_nuts": "Place bottom nuts",
    "A02_place_rear_cover": "Place rear cover",
    "A03_place_cylinder_tube": "Place cylinder tube",
    "A04_place_piston_rod_in_rod_holder": "Insert piston rod",
    "A05_place_piston_on_rod": "Place piston on rod",
    "A06_place_mounting_screw_on_piston": "Place mounting screw",
    "A07_small_l_tool_tighten_and_insert_subassembly": "Tighten and insert subassembly",
    "A08_place_front_cover": "Place front cover",
    "A09_place_long_tie_rod_assemblies": "Place tie rod assemblies",
    "A10_large_l_tool_tighten_tie_rods": "Tighten tie rods",
    "A11_tighten_top_bolt": "Tighten top bolt",
    "D01_remove_piston_rod_nut_drop": "Remove piston rod nut",
    "D02_large_l_tool_unscrew_and_drop": "Loosen tie rod nuts",
    "D03_remove_tie_rod_assemblies_drop": "Remove tie rod assemblies",
    "D04_remove_front_cover_drop": "Remove front cover",
    "D05_remove_rear_cover_drop": "Remove rear cover",
    "D06_remove_bottom_tie_rod_nuts_drop": "Remove bottom nuts",
    "D07_small_l_tool_unscrew_mounting_screw_drop": "Unscrew mounting screw",
    "D08a_drop_mounting_screw": "Drop mounting screw",
    "D08b_drop_piston": "Drop piston",
    "D09_remove_cylinder_tube_drop": "Remove cylinder tube",
    "D10_remove_piston_rod_drop": "Remove piston rod",
}


STAGE_INSTRUCTIONS = {
    "D01_remove_piston_rod_nut_drop": "Remove the piston rod nut from the piston rod and place it 1",
    "D02_large_l_tool_unscrew_and_drop": "Pick up the large L-shaped screwdriver from 3 and loosen the four tie rod nuts attached to the tie rods. Once all four have been loosened, return the screwdriver to 3.",
    "D03_remove_tie_rod_assemblies_drop": "Remove the four tie rod and tie rod nut assemblies as complete units and place each of them in 2.",
    "D04_remove_front_cover_drop": "Remove the front cylinder cover and place it in 8.",
    "D05_remove_rear_cover_drop": "Remove the cylinder tube housing together with the attached piston rod assembly, place the assembly in the rod holder, then remove the rear cylinder cover and place it in 9.",
    "D06_remove_bottom_tie_rod_nuts_drop": "Remove the remaining four bottom tie rod nuts from the square holder and place them in 10.",
    "D07_small_l_tool_unscrew_mounting_screw_drop": "Pick up the small L-shaped screwdriver 3 and unscrew the piston mounting screw from the piston rod assembly held in the rod holder. Return the screwdriver to 3.",
    "D08a_drop_mounting_screw": "Remove the piston mounting screw from the piston and place it in 7.",
    "D08b_drop_piston": "Remove the piston from the piston rod and place the piston in 6.",
    "D09_remove_cylinder_tube_drop": "Remove the cylinder tube housing from the rod holder and place it in 8.",
    "D10_remove_piston_rod_drop": "Remove the piston rod from the rod holder and place it in 5.",
    "A01_place_bottom_tie_rod_nuts": "Pick from 10 and place all four pieces into 11.",
    "A02_place_rear_cover": "Pick 9 and place it on 11 so the holes align.",
    "A03_place_cylinder_tube": "Pick 8 and place it on top of 11.",
    "A04_place_piston_rod_in_rod_holder": "Pick 5 and place it into 12.",
    "A05_place_piston_on_rod": "Pick 6 and mount it onto 12.",
    "A06_place_mounting_screw_on_piston": "Pick 7 and insert it into 12.",
    "A07_small_l_tool_tighten_and_insert_subassembly": "Pick Small screwdriver from 3, tighten, return the tool, then move the rod assembly from 12 into 11.",
    "A08_place_front_cover": "Pick 4 and place it onto the front of the assembly.",
    "A09_place_long_tie_rod_assemblies": "Pick the four pieces from 2 and insert them through the matching holes.",
    "A10_large_l_tool_tighten_tie_rods": "Pick Large Screwdriver from 3, tighten the four tie rods, then return the tool.",
    "A11_tighten_top_bolt": "Pick 1 and tighten it onto the end of 11 to complete the assembly.",
}


STAGE_ACTION_LINES = {
    "A01_place_bottom_tie_rod_nuts": [
        "First: pick ALL the four nuts from 10.",
        "Then: place them into 11.",
        "Next step: pick the rear cover from 9.",
    ],
    "A02_place_rear_cover": [
        "First: pick the rear cover from 9.(ignore if done)",
        "Then: place it onto 11.",
        "Next step: pick the cylinder tube from 8.",
    ],
    "A03_place_cylinder_tube": [
        "First: pick the cylinder tube from 8.(ignore if done)",
        "Then: place it onto the rear cover in 11.",
        "Next step: pick the piston rod from 5.",
    ],
    "A04_place_piston_rod_in_rod_holder": [
        "First: pick the piston rod from 5.(ignore if done)",
        "Then: place it into the hole in 12.(check the orientation in livefeed)",
        "Next step: pick the piston from 6.",
    ],
    "A05_place_piston_on_rod": [
        "First: pick the piston from 6.(ignore if done)",
        "Then: mount it onto the piston rod in 12.",
        "Next step: pick the mounting screw from 7.",
    ],
    "A06_place_mounting_screw_on_piston": [
        "First: pick the mounting screw from 7.(ignore if done)",
        "Then: insert it into the piston in 12.",
        "Next step: pick the small L-shaped screwdriver from 3.",
    ],
    "A07_small_l_tool_tighten_and_insert_subassembly": [
        "First: pick the small L-shaped screwdriver from 3.(ignore if done)",
        "Then: tighten the mounting screw and return the tool to 3.",
        "Then: insert the piston rod assembly inverted into ",
        "the cylinder tube and push it in.(check the orientation in livefeed)",
        "Next step: pick the front cover from 4.",
    ],
    "A08_place_front_cover": [
        "First: pick the front cover from 4.(ignore if done)",
        "Then: place it onto the assembly in 11.",
        "Next step: pick the tie rod assemblies from 2.",
    ],
    "A09_place_long_tie_rod_assemblies": [
        "First: pick ALL the tie rod assemblies from 2.(ignore if done)",
        "Then: insert all four through the front cover holes in 11.",
        "Next step: pick the large L-shaped screwdriver from 3.",
    ],
    "A10_large_l_tool_tighten_tie_rods": [
        "First: pick the large L-shaped screwdriver from 3.(ignore if done)",
        "Then: tighten the four tie rod nuts and return the tool to 3.",
        "Next step: pick the nut from 1.",
    ],
    "A11_tighten_top_bolt": [
        "First: pick the nut from 1.(ignore if done)",
        "Then: tighten it onto the end of the rod in 11.",
    ],
    "D01_remove_piston_rod_nut_drop": [
        "First: Drop the nut from 11 in 1.",
        "Next: Pick the large L-shaped screwdriver from 3."
    ],
    "D02_large_l_tool_unscrew_and_drop": [
        "First: Pick the large L-shaped "
        "screwdriver from 3.(ignore if done)",
        "Then: ONLY Loosen the four tie rod nuts in 11.",
        "Next: Return the tool to 3.",
    ],
    "D03_remove_tie_rod_assemblies_drop": [
        "Pick each tie rod assembly from 11 and ",
        "drop ALL together AT ONCE in 2.",
    ],
    "D04_remove_front_cover_drop": [
        "Drop the front cover from 11 in 4.",
    ],
    "D05_remove_rear_cover_drop": [
        "Remove the cylinder tube and rod assembly, ",
        "holding onto the base rear cover, ",
        "invert it and place in 12.",
        "Drop the rear cover from 11 in 9.",
    ],
    "D06_remove_bottom_tie_rod_nuts_drop": [
        "Pick ALL 4 nuts and drop ",
        "them all together AT ONCE in 10.",
    ],
    "D07_small_l_tool_unscrew_mounting_screw_drop": [
        "Pick the small L-shaped screwdriver from 3.",
        "Push the cylinder tube down.",
        "ONLY Loosen the screw in 12.",
        "Return the tool to 3.",
    ],
    "D08a_drop_mounting_screw": [
        "Drop the screw inside the piston 7.",
    ],
    "D08b_drop_piston": [
        "Drop the piston in 6.",
    ],
    "D09_remove_cylinder_tube_drop": [
        "Drop the cylinder tube in 8.",
    ],
    "D10_remove_piston_rod_drop": [
        "Drop the piston rod in 5.",
    ],
}


def object_name(label: str | None) -> str:
    key = str(label or "NONE").strip()
    return OBJECT_MESSAGES.get(key, key.replace("_", " ").title())


def pose_message(label: str | None) -> str:
    key = str(label or "").strip()
    return POSE_MESSAGES.get(key, key.replace("_", " ").title())


def region_message(region_id: str | None) -> str:
    key = str(region_id or "").strip()
    return REGION_MESSAGES.get(key, key.replace("_", " ").title())


def stage_title(stage_name: str | None) -> str:
    key = str(stage_name or "").strip()
    if key in STAGE_TITLE_OVERRIDES:
        return STAGE_TITLE_OVERRIDES[key]
    return key.replace("_", " ").title() if key else "Current stage"


def wrap_text(text: str, width: int = 44) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    return wrap(text, width=max(12, int(width))) or [text]


def stage_instruction(stage_desc: dict | None) -> list[str]:
    desc = stage_desc or {}
    stage_name = str(desc.get("stage_name") or "").strip()
    if " / " in stage_name:
        lines = []
        for part in [p.strip() for p in stage_name.split("/") if p.strip()]:
            lines.extend(wrap_text(STAGE_INSTRUCTIONS.get(part, stage_title(part)), width=46))
        if lines:
            return lines
    if stage_name in STAGE_INSTRUCTIONS:
        return wrap_text(STAGE_INSTRUCTIONS[stage_name], width=46)
    custom = (
        desc.get("operator_guidance")
        or desc.get("user_guidance")
        or desc.get("detailed_explanation")
        or desc.get("description")
    )
    if custom:
        return wrap_text(str(custom), width=46)

    expected_parts = [object_name(x) for x in (desc.get("expected_parts") or []) if str(x).strip() != "NONE"]
    target_regions = list(
        desc.get("completion_regions")
        or desc.get("next_stage_source_regions")
        or desc.get("start_source_regions")
        or []
    )
    if expected_parts and target_regions:
        return wrap_text(f"Handle {', '.join(expected_parts)}. {region_message(target_regions[0])}.", width=46)
    if expected_parts:
        return wrap_text(f"Handle {', '.join(expected_parts)}.", width=46)
    return ["Follow the highlighted workspace guide."]


def expected_pose_lines(stage_desc: dict | None) -> list[str]:
    lines = []
    for item in (stage_desc or {}).get("expected_poses", []) or []:
        part = object_name(item.get("part_label"))
        pose = pose_message(item.get("expected_pose"))
        window = str(item.get("temporal_window", "late")).replace("_", " ")
        lines.append(f"{part}: {pose} ({window})")
    return lines


def action_region_lines(stage_desc: dict | None) -> list[str]:
    desc = stage_desc or {}
    stage_name = str(desc.get("stage_name") or "").strip()
    if " / " in stage_name:
        lines = []
        for part in [p.strip() for p in stage_name.split("/") if p.strip()]:
            for line in STAGE_ACTION_LINES.get(part, []):
                if line not in lines:
                    lines.append(line)
        if lines:
            return lines
    if stage_name in STAGE_ACTION_LINES:
        return list(STAGE_ACTION_LINES[stage_name])

    lines = []
    for rid in desc.get("start_source_regions", []) or []:
        lines.append(region_message(rid))
    for rid in desc.get("completion_regions", []) or []:
        msg = region_message(rid)
        if msg not in lines:
            lines.append(msg)
    for rid in desc.get("next_stage_source_regions", []) or []:
        msg = region_message(rid)
        if msg not in lines:
            lines.append(msg)
    return lines
