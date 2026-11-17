from assembly_assistant.utils.metadata_validator import validate_metadata_payload


def test_validate_metadata_payload_valid_minimal() -> None:
    payload = {
        "experiment_name": "exp_v1",
        "created_at": "2026-02-23T00:00:00",
        "session": {
            "user_id": "user1",
            "session_id": "sess1",
            "operator_id": "op1",
            "trial_id": "trial1",
        },
        "references": ["ref1.mp4"],
        "system_config": {},
        "protocol": {
            "version": "1.0",
            "run_manifest": {},
            "data_sources": {},
            "camera_calibration_checklist": {},
        },
        "fsm_decisions": [],
        "finalized_at": "2026-02-23T00:10:00",
    }
    assert validate_metadata_payload(payload) == []


def test_validate_metadata_payload_missing_fields() -> None:
    payload = {"experiment_name": "exp_v1"}
    errors = validate_metadata_payload(payload)
    assert errors
    assert any("session" in e for e in errors)
    assert any("protocol" in e for e in errors)

