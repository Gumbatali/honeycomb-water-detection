"""Веб-интерфейс для просмотра образцов, карт признаков и детекции зон."""
from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from src.detection.zones import detect_zones, order_zones_by_grid, zone_profile
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
    # Имя приходит из URL, поэтому путь проверяется после разрешения:
    # иначе "../.." выводит чтение за пределы каталога образцов.
    path = (PROCESSED_DIR / f"{name}.npz").resolve()
    if not path.is_relative_to(PROCESSED_DIR.resolve()) or not path.exists():
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


# Детекция по всем образцам занимает сотни миллисекунд и растёт линейно
# с размером датасета, а карточка образца меняется только при перезаписи
# .npz — поэтому результат кэшируется по (путь, mtime, размер).
_SUMMARY_CACHE: dict[tuple[str, int, int], dict] = {}


def _sample_summary(path: Path) -> dict:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached

    features, meta = load_features(path)
    summary = {
        **asdict(meta),
        "n_zones": len(detect_zones(features)),
        "size_mb": round(stat.st_size / 1e6, 2),
        "features": list(features.names),
    }
    # Чистим только устаревшие записи ЭТОГО же файла (старый mtime/size) —
    # полный clear() на каждой вставке обнулял кэш внутри одного прохода
    # по списку образцов: список из N файлов пересчитывал детекцию N раз
    # вместо одного, и /api/samples не ускорялся между запросами.
    stale = [k for k in _SUMMARY_CACHE if k[0] == key[0] and k != key]
    for k in stale:
        del _SUMMARY_CACHE[k]
    _SUMMARY_CACHE[key] = summary
    return summary


@app.get("/api/samples")
def list_samples() -> list[dict]:
    """Список обработанных образцов с метаданными и числом найденных зон."""
    return [_sample_summary(path) for path in _sample_paths()]


@app.get("/api/samples/{name}/zones")
def get_zones(name: str) -> dict:
    """Детекция зон консенсусом четырёх независимых методов.

    Порог и признак больше не задаются вызывающей стороной: детектор
    голосует по всем картам признаков сразу (см. src/detection/zones.py),
    и настройка одного порога на один признак не отражает, как он
    в действительности принимает решение.
    """
    features, meta = load_features(_find_sample(name))
    zones = detect_zones(features)
    return {
        "sample": meta.name,
        "height": meta.height,
        "width": meta.width,
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


@app.get("/api/samples/{name}/compare")
def compare_features(name: str) -> dict:
    """Сравнение профилей всех зон образца — основа для классификации.

    Возвращает зоны, упорядоченные как читается сетка, чтобы их можно
    было сопоставить с протоколом эксперимента.
    """
    features, meta = load_features(_find_sample(name))
    zones = order_zones_by_grid(detect_zones(features))

    return {
        "sample": meta.name,
        "orientation": meta.orientation,
        "features": list(features.names),
        "zones": [
            {
                "index": index + 1,
                "bbox": [zone.row0, zone.col0, zone.row1, zone.col1],
                "score": round(zone.score, 3),
                "profile": zone_profile(features, zone),
            }
            for index, zone in enumerate(zones)
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
