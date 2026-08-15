"""Конвейер: сырые .mat/.7z -> компактные карты признаков (.npz).

Позволяет держать рабочий датасет в мегабайтах вместо гигабайтов и
не хранить исходные кубы локально дольше, чем нужно для одного прохода.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.preprocessing.features import FeatureMaps, extract_features
from src.utils.io import extracted_from_7z, load_thermogram


@dataclass(frozen=True)
class SampleMeta:
    """Метаданные образца, восстановленные из имени файла и каталога."""

    name: str
    source_file: str
    orientation: str
    fps: float
    n_frames: int
    height: int
    width: int


def infer_orientation(filename: str) -> str:
    """Определяет ориентацию образца по имени файла эксперимента."""
    lowered = filename.lower()
    if "переворот" in lowered:
        return "flipped"
    if "подвеш" in lowered:
        return "suspended"
    if "полке" in lowered:
        return "shelf"
    if "напечатанный" in lowered:
        return "printed"
    if "calib" in lowered:
        return "calibration"
    return "unknown"


def save_features(features: FeatureMaps, meta: SampleMeta, out_path: str | Path) -> Path:
    """Сохраняет карты признаков и метаданные в сжатый .npz."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        maps=features.maps,
        names=np.array(features.names),
        meta=np.array(json.dumps(asdict(meta))),
    )
    return out_path


def load_features(path: str | Path) -> tuple[FeatureMaps, SampleMeta]:
    """Читает карты признаков и метаданные из .npz."""
    with np.load(path, allow_pickle=False) as bundle:
        meta = SampleMeta(**json.loads(str(bundle["meta"])))
        features = FeatureMaps(
            maps=bundle["maps"],
            names=tuple(str(n) for n in bundle["names"]),
            fps=meta.fps,
            source=meta.source_file,
        )
    return features, meta


def process_mat(
    path: str | Path,
    out_dir: str | Path,
    fps: float | None = None,
    label_name: str | None = None,
) -> Path:
    """Обрабатывает один .mat: считает признаки и сохраняет .npz.

    Args:
        label_name: имя образца для метаданных. Нужно, когда .mat лежит
            во временной папке, а условия эксперимента закодированы
            в имени исходного архива.
    """
    path = Path(path)
    display_name = label_name or path.stem
    thermogram = load_thermogram(path, fps=fps)
    features = extract_features(thermogram.data, thermogram.fps, source=display_name)
    height, width, n_frames = thermogram.shape
    meta = SampleMeta(
        name=display_name,
        source_file=f"{display_name}{path.suffix}",
        orientation=infer_orientation(display_name),
        fps=thermogram.fps,
        n_frames=n_frames,
        height=height,
        width=width,
    )
    return save_features(features, meta, Path(out_dir) / f"{display_name}.npz")


def process_7z(archive: str | Path, out_dir: str | Path, fps: float | None = None) -> Path:
    """Обрабатывает .7z: распаковывает во временную папку, считает признаки, удаляет сырьё.

    Исходный .mat не остаётся на диске — освобождается сразу после прохода.
    """
    from src.utils.io import fps_from_filename

    archive = Path(archive)
    resolved_fps = fps if fps else fps_from_filename(archive.name)
    with tempfile.TemporaryDirectory(prefix="thermo_") as tmp:
        with extracted_from_7z(archive, tmp) as mat_path:
            # Имя берём от архива — оно несёт условия эксперимента,
            # тогда как распакованный .mat лежит во временной папке.
            return process_mat(
                mat_path, out_dir, fps=resolved_fps, label_name=archive.stem
            )
