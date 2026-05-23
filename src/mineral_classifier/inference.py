from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    class_names: list[str]
    probabilities: list[float]


def load_model(model_path: str | Path) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Put best.pt in models/ or update configs/app.yaml."
        )
    return YOLO(str(path))


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(image_size)


def predict(model: YOLO, image_path: str | Path) -> PredictionResult:
    results: list[Any] = model(str(image_path))
    if not results or results[0].probs is None:
        raise ValueError("Model did not return classification probabilities.")

    probabilities = results[0].probs.data.cpu().numpy().astype(float)
    class_names = [model.names[index] for index in range(len(probabilities))]
    best_index = int(np.argmax(probabilities))

    return PredictionResult(
        predicted_class=class_names[best_index],
        confidence=float(probabilities[best_index]),
        class_names=class_names,
        probabilities=probabilities.tolist(),
    )
