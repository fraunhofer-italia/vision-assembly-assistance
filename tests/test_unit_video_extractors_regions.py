from assembly_assistant.pipeline.region_detector import RegionDetection
from assembly_assistant.pipeline.region_registry import RegionRegistry
from assembly_assistant.pipeline.video_extractors import _annotate_region_hits, _resolve_motion_allowed_regions


def test_annotate_region_hits_prefers_detector_polygons_over_static_registry():
    reg = RegionRegistry.from_dict(
        {
            "registry_id": "vx",
            "modes": ["assembly", "dismantle"],
            "regions": [
                {
                    "region_id": "box_a",
                    "coordinate_frame": "image_px",
                    "polygon_px": [[10, 10], [20, 10], [20, 20], [10, 20]],
                    "roles": ["part_pick"],
                }
            ],
            "stage_bindings": {},
        }
    )

    frame_data = {
        "hands": {
            "left": {"visible": True, "wrist": (0.5, 0.5), "keypoints": None},
            "right": {"visible": False, "wrist": None, "keypoints": None},
        }
    }
    keypoints_px_by_label = {
        "LEFT": [[105.0, 105.0]],  # inside detected polygon, not static polygon
        "RIGHT": None,
    }
    detected_regions = {
        "box_a": RegionDetection(
            region_id="box_a",
            source="detected",
            confidence=0.9,
            coordinate_frame="image_px",
            polygon_px=[[100.0, 100.0], [120.0, 100.0], [120.0, 120.0], [100.0, 120.0]],
            polygon_norm=None,
            detector_debug={},
        )
    }
    _annotate_region_hits(
        frame_data=frame_data,
        keypoints_px_by_label=keypoints_px_by_label,
        selected_label="LEFT",
        region_registry=reg,
        detected_regions=detected_regions,
        operation_mode="assembly",
        trigger_point_mode="wrist",
    )
    assert "box_a" in frame_data["region_hits"]["active"]["regions"]
    assert "part_pick" in frame_data["region_hits"]["active"]["roles"]


def test_resolve_motion_allowed_regions_unions_remaining_unordered_group_members():
    class _FakeMotion:
        current_stage_index = 0

        def _unordered_remaining_descs(self, group_id):
            assert group_id == "g1"
            return [
                {"stage_name": "D08a_drop_mounting_screw", "completion_regions": ["box_mounting_screw"]},
                {"stage_name": "D08b_drop_piston", "completion_regions": ["box_piston"]},
            ]

    stage_descs = [
        {
            "stage_name": "D08a_drop_mounting_screw",
            "unordered_group_id": "g1",
            "unordered_group_members": ["D08a_drop_mounting_screw", "D08b_drop_piston"],
            "completion_regions": ["box_mounting_screw"],
        },
        {
            "stage_name": "D08b_drop_piston",
            "unordered_group_id": "g1",
            "unordered_group_members": ["D08a_drop_mounting_screw", "D08b_drop_piston"],
            "completion_regions": ["box_piston"],
        },
    ]

    allowed = _resolve_motion_allowed_regions(_FakeMotion(), stage_descs, "dismantle", "completion")

    assert allowed == {"box_mounting_screw", "box_piston"}
