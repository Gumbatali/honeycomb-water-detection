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


def render_map_jpeg(values: np.ndarray, cmap: str = "inferno") -> bytes:
    """Рендерит карту признака в JPEG с перцентильным контрастированием.

    PNG того же кадра весит ~768 КБ и на канале до ВМ (не локальная сеть)
    качается дольше, чем сама детекция зон: 4.4 с одной скачки против
    0.6 с построения PNG. JPEG q90 даёт ~155 КБ при визуально неотличимом
    результате — карта уже прогнана через цветовую схему и перцентильное
    контрастирование, поэтому артефакты сжатия на ней не заметны так, как
    были бы на исходных числовых данных.
    """
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
    Image.fromarray(rgba[:, :, :3]).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# Консенсусная детекция (4 голоса: порог, GMM, PCA, решётка) занимает
# секунды на образец, а не миллисекунды — GMM/PCA обучаются заново на
# каждый вызов. Без кэша это происходило трижды на клик по одному и тому
# же образцу: при построении списка (n_zones), при открытии карты (/zones)
# и при переключении на таблицу (/compare). Кэшируются сами объекты Zone —
# ключ учитывает mtime/размер файла, поэтому запись сбрасывается сама
# при обновлении .npz, без явной инвалидации.
_ZONES_CACHE: dict[tuple[str, int, int], list] = {}


def _cached_zones(path: Path, features) -> list:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _ZONES_CACHE.get(key)
    if cached is not None:
        return cached

    zones = detect_zones(features)
    # Чистим только устаревшие записи ЭТОГО же файла (старый mtime/size):
    # полный clear() на каждой вставке обнулял бы кэш внутри одного
    # прохода по списку образцов вместо инвалидации одного файла.
    stale = [k for k in _ZONES_CACHE if k[0] == key[0] and k != key]
    for k in stale:
        del _ZONES_CACHE[k]
    _ZONES_CACHE[key] = zones
    return zones


def _sample_summary(path: Path) -> dict:
    features, meta = load_features(path)
    stat = path.stat()
    return {
        **asdict(meta),
        "n_zones": len(_cached_zones(path, features)),
        "size_mb": round(stat.st_size / 1e6, 2),
        "features": list(features.names),
    }


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
    path = _find_sample(name)
    features, meta = load_features(path)
    zones = _cached_zones(path, features)
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
    """JPEG-визуализация карты признака."""
    features, _ = load_features(_find_sample(name))
    if feature not in features.names:
        raise HTTPException(status_code=400, detail=f"Неизвестный признак '{feature}'")
    return Response(content=render_map_jpeg(features[feature]), media_type="image/jpeg")


@app.get("/api/samples/{name}/compare")
def compare_features(name: str) -> dict:
    """Сравнение профилей всех зон образца — основа для классификации.

    Возвращает зоны, упорядоченные как читается сетка, чтобы их можно
    было сопоставить с протоколом эксперимента.
    """
    path = _find_sample(name)
    features, meta = load_features(path)
    zones = order_zones_by_grid(_cached_zones(path, features))

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
