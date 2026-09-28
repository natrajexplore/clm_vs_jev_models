from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply dotted overrides (see :func:`apply_overrides`)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply overrides like ``task.test_size=50`` in place.

    Values are parsed as YAML, so ``1`` -> int, ``0.5`` -> float, ``null`` -> None.
    """
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Unknown config key {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return cfg
