"""Config loading + logging setup shared by every pipeline stage."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve every relative path against the project root so the CLI works from anywhere
    cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
