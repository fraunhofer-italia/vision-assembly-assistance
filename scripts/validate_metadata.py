from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.utils.metadata_validator import validate_metadata_file


def _collect_targets(args) -> list[Path]:
    if args.metadata:
        return [Path(args.metadata)]
    if args.experiment_dir:
        return [Path(args.experiment_dir) / "metadata.json"]
    root = Path(args.experiments_root)
    return sorted(root.glob("*/metadata.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate metadata.json schema for experiment runs.")
    parser.add_argument("--metadata", default=None, help="Path to metadata.json")
    parser.add_argument("--experiment-dir", default=None, help="Path to experiment folder containing metadata.json")
    parser.add_argument("--experiments-root", default="experiments", help="Validate all experiments under this root")
    args = parser.parse_args()

    targets = _collect_targets(args)
    if not targets:
        print("No metadata files found.")
        return 1

    total_errors = 0
    for path in targets:
        if not path.exists():
            print(f"[FAIL] {path}: file not found")
            total_errors += 1
            continue
        errors = validate_metadata_file(path)
        if errors:
            print(f"[FAIL] {path}")
            for e in errors:
                print(f"  - {e}")
            total_errors += len(errors)
        else:
            print(f"[OK] {path}")

    if total_errors > 0:
        print(f"\nValidation failed: {total_errors} issue(s).")
        return 1
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

