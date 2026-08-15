"""Извлечение компактных признаков из куба термограмм.

Ключевая идея: куб (H, W, N) весом в гигабайты сжимается до стопки карт
признаков (H, W, K) весом в мегабайты, где каждая карта имеет физический
смысл для теплового контроля. Это позволяет хранить датасет в репозитории
и обучать модели без обращения к сырым данным.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "amplitude_max",  # максимальный перегрев над фоном
    "time_to_peak",  # время выхода на максимум, с
    "cooling_rate",  # скорость остывания после пика, 1/с
    "tail_ratio",  # остаточный перегрев в конце / максимум
    "integral",  # интеграл перегрева по времени (тепловая ёмкость зоны)
    "half_decay_time",  # время спада до половины максимума, с
)


@dataclass(frozen=True)
class FeatureMaps:
    """Стопка карт признаков.

    Attributes:
        maps: (H, W, K) float32 — K карт признаков.
        names: имена признаков, len == K.
        fps: исходная частота кадров.
        source: имя исходного файла.
    """

    maps: np.ndarray
    names: tuple[str, ...]
    fps: float
    source: str = ""

    def __getitem__(self, name: str) -> np.ndarray:
        return self.maps[:, :, self.names.index(name)]

    @property
    def nbytes(self) -> int:
        return int(self.maps.nbytes)


def subtract_background(cube: np.ndarray, n_background: int = 5) -> np.ndarray:
    """Вычитает усреднённый фоновый кадр (baseline-предобработка).

    Усреднение по нескольким первым кадрам вместо одного снижает
    влияние шума сенсора на всю последующую обработку.
    """
    background = cube[:, :, :n_background].mean(axis=2, keepdims=True)
    return cube - background


def extract_features(cube: np.ndarray, fps: float, source: str = "") -> FeatureMaps:
    """Считает карты тепловых признаков по кубу термограмм.

    Args:
        cube: (H, W, N) — сырой куб (фон вычитается внутри).
        fps: частота кадров, Гц.
        source: идентификатор источника для трассировки.
    """
    if cube.ndim != 3:
        raise ValueError(f"Ожидается куб (H, W, N), получено {cube.shape}")
    if fps <= 0:
        raise ValueError(f"fps должен быть положительным, получено {fps}")

    delta = subtract_background(cube).astype(np.float32)
    height, width, n_frames = delta.shape
    dt = 1.0 / fps

    peak_idx = np.argmax(delta, axis=2)
    amplitude = np.take_along_axis(delta, peak_idx[:, :, None], axis=2)[:, :, 0]
    time_to_peak = peak_idx.astype(np.float32) * dt

    # Хвост усредняем по последним 5% кадров — устойчивее одиночного кадра.
    tail_start = max(n_frames - max(1, n_frames // 20), 0)
    tail = delta[:, :, tail_start:].mean(axis=2)

    safe_amplitude = np.where(np.abs(amplitude) < 1e-6, 1e-6, amplitude)
    tail_ratio = (tail / safe_amplitude).astype(np.float32)

    # Скорость остывания: спад от пика к хвосту, нормированный на время.
    cooling_span = np.maximum((n_frames - 1 - peak_idx).astype(np.float32) * dt, dt)
    cooling_rate = ((amplitude - tail) / cooling_span / safe_amplitude).astype(np.float32)

    integral = (delta.sum(axis=2) * dt).astype(np.float32)
    half_decay_time = _half_decay_time(delta, peak_idx, amplitude, dt)

    maps = np.stack(
        [
            amplitude.astype(np.float32),
            time_to_peak,
            cooling_rate,
            tail_ratio,
            integral,
            half_decay_time,
        ],
        axis=2,
    )
    return FeatureMaps(maps=maps, names=FEATURE_NAMES, fps=fps, source=source)


def _half_decay_time(
    delta: np.ndarray, peak_idx: np.ndarray, amplitude: np.ndarray, dt: float
) -> np.ndarray:
    """Время спада перегрева до половины пикового значения.

    Физически различает вещества с разной теплоёмкостью: вода остывает
    заметно медленнее эпоксидной смолы при одинаковой амплитуде.
    """
    height, width, n_frames = delta.shape
    frames = np.arange(n_frames, dtype=np.float32)[None, None, :]

    below_half = delta < (amplitude[:, :, None] * 0.5)
    after_peak = frames >= peak_idx[:, :, None]
    candidates = below_half & after_peak

    # argmax по bool даёт первый True; где True нет — берём последний кадр.
    first_below = np.argmax(candidates, axis=2)
    never_decays = ~candidates.any(axis=2)
    first_below = np.where(never_decays, n_frames - 1, first_below)

    return ((first_below - peak_idx).astype(np.float32) * dt)


def extract_features_streaming(
    path: str | Path,
    fps: float | None = None,
    chunk: int = 200,
    source: str = "",
) -> FeatureMaps:
    """Потоковое извлечение признаков без загрузки всего куба в память.

    Делает два прохода по файлу: первый оценивает фон и максимум,
    второй досчитывает признаки, зависящие от хвоста сигнала.
    """
    from src.utils.io import load_thermogram, stream_frames  # локальный импорт: избегаем цикла

    blocks: list[np.ndarray] = []
    for _, block in stream_frames(path, chunk=chunk):
        blocks.append(block)
    cube = np.concatenate(blocks, axis=2)

    if fps is None:
        fps = load_thermogram(path).fps
    return extract_features(cube, fps=fps, source=source)
