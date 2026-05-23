from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    title: str
    image_size: tuple[int, int]
    model_path: Path


def load_config(path: str | Path = "configs/app.yaml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    return AppConfig(
        title=raw_config["app"]["title"],
        image_size=tuple(raw_config["app"]["image_size"]),
        model_path=Path(raw_config["model"]["path"]),
    )
