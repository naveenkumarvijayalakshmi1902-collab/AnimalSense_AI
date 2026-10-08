from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike[str] = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str | os.PathLike[str]) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def select_device(config_device: str = "auto") -> str | int:
    if config_device != "auto":
        return config_device
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dirs() -> None:
    for p in (ROOT / "outputs/videos", ROOT / "outputs/logs", ROOT / "outputs/graphs", ROOT / "models/baseline", ROOT / "models/enhanced"):
        p.mkdir(parents=True, exist_ok=True)
