"""Аугментация карт признаков и векторов зон.

Преобразования подобраны так, чтобы не нарушать физику измерения:
геометрия сотовой решётки допускает повороты и отражения, а вариации
мощности нагрева и шума сенсора воспроизводят реальный разброс условий.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.preprocessing.features import FEATURE_NAMES, FeatureMaps

# Признаки, растущие пропорционально мощности нагрева. Время выхода на
# пик и относительные отношения от неё почти не зависят.
AMPLITUDE_SCALED: frozenset[str] = frozenset({"amplitude_max", "integral"})


def flip(features: FeatureMaps, axis: int = 1) -> FeatureMaps:
    """Отражение карт по горизонтали (axis=1) или вертикали (axis=0)."""
    return replace(features, maps=np.flip(features.maps, axis=axis).copy())


def rotate90(features: FeatureMaps, k: int = 1) -> FeatureMaps:
    """Поворот на k*90 градусов в плоскости изображения."""
    return replace(features, maps=np.rot90(features.maps, k=k, axes=(0, 1)).copy())


def scale_amplitude(features: FeatureMaps, factor: float) -> FeatureMaps:
    """Имитация иной мощности нагрева.

    Масштабируются только амплитудные признаки: при изменении мощности
    лампы перегрев меняется пропорционально, а форма кривой остывания
    (время до пика, относительный хвост) сохраняется.
    """
    if factor <= 0:
        raise ValueError(f"factor должен быть положительным, получено {factor}")

    maps = features.maps.copy()
    for index, name in enumerate(features.names):
        if name in AMPLITUDE_SCALED:
            maps[:, :, index] *= factor
    return replace(features, maps=maps)


def add_sensor_noise(
    features: FeatureMaps, sigma: float = 0.02, rng: np.random.Generator | None = None
) -> FeatureMaps:
    """Добавляет гауссов шум, пропорциональный масштабу каждого признака."""
    generator = rng if rng is not None else np.random.default_rng()
    maps = features.maps.copy()
    for index in range(maps.shape[2]):
        channel = maps[:, :, index]
        scale = float(np.nanstd(channel))
        if scale > 0:
            channel += generator.normal(0.0, sigma * scale, size=channel.shape)
    return replace(features, maps=maps)


def augment_maps(
    features: FeatureMaps,
    n_variants: int = 4,
    rng: np.random.Generator | None = None,
) -> list[FeatureMaps]:
    """Генерирует набор аугментированных вариантов карт признаков."""
    generator = rng if rng is not None else np.random.default_rng(0)
    variants: list[FeatureMaps] = []

    for _ in range(n_variants):
        current = features
        if generator.random() < 0.5:
            current = flip(current, axis=int(generator.integers(0, 2)))
        rotations = int(generator.integers(0, 4))
        if rotations:
            current = rotate90(current, k=rotations)
        current = scale_amplitude(current, float(generator.uniform(0.85, 1.15)))
        current = add_sensor_noise(current, sigma=0.02, rng=generator)
        variants.append(current)

    return variants


def augment_vector(
    vector: np.ndarray,
    n_variants: int = 5,
    names: tuple[str, ...] = FEATURE_NAMES,
    amplitude_jitter: float = 0.12,
    noise_sigma: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Аугментирует вектор признаков зоны для расширения обучающей выборки.

    Returns:
        (n_variants, len(vector)) — варианты исходного вектора.
    """
    generator = rng if rng is not None else np.random.default_rng()
    variants = np.repeat(np.atleast_2d(vector), n_variants, axis=0).astype(np.float32)

    factors = generator.uniform(1 - amplitude_jitter, 1 + amplitude_jitter, n_variants)
    for index, name in enumerate(names):
        if name in AMPLITUDE_SCALED:
            variants[:, index] *= factors
        spread = abs(float(vector[index])) * noise_sigma
        if spread > 0:
            variants[:, index] += generator.normal(0.0, spread, n_variants)

    return variants
