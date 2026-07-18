from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(config: str | Path | dict) -> tuple[dict, Path | None]:
    if isinstance(config, dict):
        return deepcopy(config), None
    path = Path(config).expanduser().resolve()
    with path.open() as stream:
        return yaml.safe_load(stream), path


def output_paths(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["project"]["output_root"]).expanduser()
    return {
        "root": root,
        "raw": root / "spectra" / "raw",
        "binned": root / "spectra" / "binned",
        "workspaces": root / "namaster_workspaces",
        "theory": root / "theory",
        "covariance": root / "covariance",
        "chains": root / "chains",
        "plots": root / "plots",
        "results": root / "results",
    }


def ensure_output_paths(cfg: dict) -> dict[str, Path]:
    paths = output_paths(cfg)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
