import os
from pathlib import Path


def get_experiments_root() -> Path:
    return Path(os.environ.get("SLOA_EXPERIMENTS_ROOT", "experiments"))

