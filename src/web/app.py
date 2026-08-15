"""Веб-интерфейс для просмотра образцов, карт признаков и детекции зон."""
from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from src.detection.zones import detect_zones, zone_profile
from src.preprocessing.dataset import load_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Путь задаётся от корня проекта, а не от рабочей директории процесса:
# сервер запускается из разных мест (CLI, launch.json, тесты).
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Honeycomb Thermal Inspection")


def _sample_paths() -> list[Path]:
    return sorted(PROCESSED_DIR.glob("*.npz"))


def _find_sample(name: str) -> Path:
    path = PROCESSED_DIR / f"{name}.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Образец '{name}' не найден")
    return path


def render_map_png(values: np.ndarray, cmap: str = "inferno") -> bytes:
    """Рендерит карту признака в PNG с перцентильным контрастированием."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    from PIL import Image

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.zeros(1, dtype=np.float32)
    low, high = np.percentile(finite, [2, 98])
    if high - low < 1e-9:
        high = low + 1e-9

    normalized = np.clip((values - low) / (high - low), 0, 1)
    normalized = np.nan_to_num(normalized)
    rgba = (colormaps[cmap](normalized) * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/api/samples")
def list_samples() -> list[dict]:
    """Список обработанных образцов с метаданными и числом найденных зон."""
    samples = []
    for path in _sample_paths():
        features, meta = load_features(path)
        zones = detect_zones(features)
        samples.append(
            {
                **asdict(meta),
                "n_zones": len(zones),
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "features": list(features.names),
            }
        )
    return samples


@app.get("/api/samples/{name}/zones")
def get_zones(name: str, feature: str = "half_decay_time", sigma: float = 1.0) -> dict:
    """Детекция зон с настраиваемыми параметрами."""
    features, meta = load_features(_find_sample(name))
    if feature not in features.names:
        raise HTTPException(status_code=400, detail=f"Неизвестный признак '{feature}'")

    zones = detect_zones(features, feature_name=feature, threshold_sigma=sigma)
    return {
        "sample": meta.name,
        "height": meta.height,
        "width": meta.width,
        "feature": feature,
        "sigma": sigma,
        "zones": [
            {**zone.as_dict(), "profile": zone_profile(features, zone)} for zone in zones
        ],
    }


@app.get("/api/samples/{name}/map/{feature}")
def get_map(name: str, feature: str) -> Response:
    """PNG-визуализация карты признака."""
    features, _ = load_features(_find_sample(name))
    if feature not in features.names:
        raise HTTPException(status_code=400, detail=f"Неизвестный признак '{feature}'")
    return Response(content=render_map_png(features[feature]), media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
