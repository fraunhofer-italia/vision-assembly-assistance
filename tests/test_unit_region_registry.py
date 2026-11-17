from assembly_assistant.pipeline.region_registry import RegionRegistry


def test_region_registry_load_and_mode_semantics():
    reg = RegionRegistry.from_file("configs/regions_v2.json")
    assert reg.registry_id == "regions_v2"
    assert reg.resolve_role_enabled("box_piston_rod", "assembly", "part_pick") is True
    assert reg.resolve_role_enabled("box_piston_rod", "assembly", "part_drop") is False
    assert reg.resolve_role_enabled("box_piston_rod", "dismantle", "part_drop") is True


def test_region_registry_stage_binding_validation_usage():
    reg = RegionRegistry.from_file("configs/regions_v2.json")
    assert reg.is_region_allowed("A10_large_l_tool_tighten_tie_rods", "assembly", "valid_tool_regions", "box_tools_pair") is True
    assert reg.is_region_allowed("A10_large_l_tool_tighten_tie_rods", "assembly", "valid_source_regions", "box_front_cover") is False
