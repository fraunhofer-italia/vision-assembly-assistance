from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_metadata_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(payload, dict):
        return ["metadata payload must be a JSON object"]

    if not _is_non_empty_str(payload.get("experiment_name")):
        errors.append("missing/invalid 'experiment_name'")
    if not _is_non_empty_str(payload.get("created_at")):
        errors.append("missing/invalid 'created_at'")

    session = payload.get("session")
    if not isinstance(session, dict):
        errors.append("missing/invalid 'session' object")
    else:
        for key in ("user_id", "session_id", "operator_id", "trial_id"):
            if not _is_non_empty_str(session.get(key)):
                errors.append(f"missing/invalid 'session.{key}'")

    refs = payload.get("references")
    if not isinstance(refs, list):
        errors.append("missing/invalid 'references' list")

    if not isinstance(payload.get("system_config"), dict):
        errors.append("missing/invalid 'system_config' object")

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("missing/invalid 'protocol' object")
    else:
        if not _is_non_empty_str(protocol.get("version")):
            errors.append("missing/invalid 'protocol.version'")
        run_manifest = protocol.get("run_manifest")
        if not isinstance(run_manifest, dict):
            errors.append("missing/invalid 'protocol.run_manifest'")
        data_sources = protocol.get("data_sources")
        if not isinstance(data_sources, dict):
            errors.append("missing/invalid 'protocol.data_sources'")
        checklist = protocol.get("camera_calibration_checklist")
        if not isinstance(checklist, dict):
            errors.append("missing/invalid 'protocol.camera_calibration_checklist'")

    fsm_decisions = payload.get("fsm_decisions", [])
    if not isinstance(fsm_decisions, list):
        errors.append("invalid 'fsm_decisions' (must be list)")
    else:
        for idx, dec in enumerate(fsm_decisions):
            if not isinstance(dec, dict):
                errors.append(f"fsm_decisions[{idx}] must be object")
                continue
            if not _is_non_empty_str(dec.get("stage")):
                errors.append(f"fsm_decisions[{idx}] missing/invalid 'stage'")
            fb = dec.get("feedback")
            if not isinstance(fb, dict):
                errors.append(f"fsm_decisions[{idx}] missing/invalid 'feedback' object")

    if "finalized_at" in payload and not _is_non_empty_str(payload.get("finalized_at")):
        errors.append("invalid 'finalized_at'")

    return errors


def validate_metadata_file(path: Path) -> List[str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"failed to read/parse metadata JSON: {exc}"]
    return validate_metadata_payload(payload)

