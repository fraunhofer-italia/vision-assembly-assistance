from assembly_assistant.synchronization.sync_manager import SyncManager

def test_resolve_tool_none():
    sm = SyncManager(max_time_diff=0.2)

    result = sm._resolve_tool(
        hand_data={},
        tool_data=None,
        hand_timestamp=10.0
    )

    assert result["class"] == "NONE"
    assert result["confidence"] == 0.0

def test_resolve_tool_temporal_gating():
    sm = SyncManager(max_time_diff=0.1)

    tool_data = {
        "class" : "ScrewdriverL",
        "timestamp" : 9.0,
        "confidence" : 0.9
    }

    result = sm._resolve_tool(
        hand_data={},
        tool_data=tool_data,
        hand_timestamp=10.0
    )

    assert result["class"] == "NONE"

def test_resolve_tool_valid():
    sm = SyncManager(max_time_diff=0.5)

    tool_data = {
        "class" : "ScrewdriverL",
        "timestamp" : 10.05,
        "bbox" : (10, 10, 50, 50),
        "confidence" : 0.87
    }

    result = sm._resolve_tool(
        hand_data={},
        tool_data=tool_data,
        hand_timestamp=10.0
    )

    assert result["class"] == "screwdriver_large_l"
    assert result["confidence"] == 0.87
    assert result["bbox"] == (10, 10, 50, 50)

def test_resolve_tool_missing_timestamp():
    sm = SyncManager(max_time_diff=0.1)

    tool_data = {
        "class" : "ScrewdriverL",
        "timestamp" : 0.6
    }

    result = sm._resolve_tool(
        hand_data={},
        tool_data=tool_data,
        hand_timestamp=5.0
    )

    assert result["class"] == "NONE"
