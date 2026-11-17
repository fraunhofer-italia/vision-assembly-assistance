from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def append_fatigue_metrics(experiment_path: Path, entry: Dict) -> Path:
    out_path = Path(experiment_path) / "fatigue_metrics.json"

    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = {"generated_at": datetime.utcnow().isoformat(), "entries": []}

    payload.setdefault("entries", []).append(entry)
    payload["updated_at"] = datetime.utcnow().isoformat()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

