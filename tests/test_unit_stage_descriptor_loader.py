from assembly_assistant.pipeline.stage_descriptor_loader import load_stage_descriptors


def test_load_stage_descriptors_for_mode_and_order():
    out = load_stage_descriptors(
        config_path="configs/stage_descriptors_v1.json",
        operation_mode="dismantle",
        stage_names=["D01_remove_piston_rod_nut_drop"],
    )
    assert [x["stage_name"] for x in out] == ["D01_remove_piston_rod_nut_drop"]
    assert "completion_regions" in out[0]


def test_load_stage_descriptors_missing_stage_raises():
    try:
        load_stage_descriptors(
            config_path="configs/stage_descriptors_v1.json",
            operation_mode="assembly",
            stage_names=["StageZ"],
        )
    except ValueError as e:
        assert "missing descriptors" in str(e)
        return
    assert False, "Expected ValueError for missing stage descriptor"
