"""Локальная вариация контраста дефекта относительно фона зоны.

Перенос идеи vooterr (`gru-experiments`, эксперимент 042, "local +/-10% gain
cell patching"): вместо того чтобы масштабировать весь кадр целиком (как
`SIM_TO_REAL_SCALE` в `src/synthesis/fd_solver.py` — поправка к синтетике),
здесь варьируется КОНТРАСТ ОТДЕЛЬНОЙ ЗОНЫ относительно её локального фона —
аугментация поверх уже реальных записей, а не калибровка синтетики.

Формула vooterr: ``output = neutral + gain * (source_defect - source_neutral)``,
то есть смещение зоны от нейтрального уровня масштабируется, а не сам
абсолютный сигнал — так контраст меняется, а физически реалистичный уровень
фона не завышается. Здесь адаптировано под наш формат: вместо отдельного
"нейтрального" кадра (у vooterr — синтезированного заменой дефектных
пикселей донорами из фона, `donor_maps`) используется среднее значение
пикселей маски зоны на первом кадре записи — тот же ноль отсчёта, что
использует `normalized_contrast` для всего кадра (``T - T(0)``), но взятый
локально по зоне, а не по опорной области.

Диапазон ±10% по умолчанию — измеренная у vooterr разница в среднем
5-секундном отклике между water1 и water2 (7.9%), с запасом.
"""
from __future__ import annotations

import numpy as np

#: Диапазон случайного коэффициента усиления по умолчанию — покрывает
#: измеренную 7.9%-ю разницу отклика между записями с запасом.
DEFAULT_GAIN_RANGE: tuple[float, float] = (0.90, 1.10)


def apply_local_gain(
    cube: np.ndarray, zone_mask: np.ndarray, gain: float
) -> np.ndarray:
    """Масштабирует отклонение зоны от её собственного начального уровня.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, N_frames) — сырой или предобработанный температурный куб.
    zone_mask : np.ndarray
        (H, W) bool — маска одной зоны (например, срез `cell_index == id`).
    gain : float
        Коэффициент масштабирования отклонения от начального кадра.
        1.0 — без изменений.

    Returns
    -------
    np.ndarray
        Копия куба; пиксели вне ``zone_mask`` не изменяются.

    Raises
    ------
    ValueError
        Если формы не совпадают, маска пуста или ``gain`` не положителен.
    """
    if cube.ndim != 3:
        raise ValueError(f"Ожидается куб (H, W, N), получено {cube.shape}")
    if zone_mask.shape != cube.shape[:2]:
        raise ValueError(f"Маска {zone_mask.shape} не совпадает с кадром {cube.shape[:2]}")
    mask = np.asarray(zone_mask, dtype=bool)
    if not mask.any():
        raise ValueError("Маска зоны пуста")
    if gain <= 0.0:
        raise ValueError(f"gain должен быть положительным, получено {gain}")

    result = cube.astype(np.float32, copy=True)
    baseline = result[mask, :1]  # (n_pixels, 1) — уровень первого кадра.
    result[mask] = baseline + gain * (result[mask] - baseline)
    return result


def random_local_gain(
    cube: np.ndarray,
    zone_mask: np.ndarray,
    rng: np.random.Generator,
    gain_range: tuple[float, float] = DEFAULT_GAIN_RANGE,
) -> np.ndarray:
    """`apply_local_gain` со случайным коэффициентом из ``gain_range``."""
    gain = float(rng.uniform(*gain_range))
    return apply_local_gain(cube, zone_mask, gain)
