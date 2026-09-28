import json
import time
from pathlib import Path
from typing import Any

import yaml


def make_run_dir(root: str | Path, name: str) -> Path:
    """Create ``root/<name>_<timestamp>`` and return it."""
    run_dir = Path(root) / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class JsonlLogger:
    """Append one JSON object per line; also echo to stdout."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, record: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
