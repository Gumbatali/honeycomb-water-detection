"""Препроцессинг термограмм: нормированный контраст и лог-временное представление.

Модуль готовит трёхканальный вход модели из сырого куба термограмм.

Два принципиальных решения, подтверждённых численной проверкой на модельных
кривых остывания:

1. Контраст нормируется на СКАЛЯР — пиковый подъём температуры опорной зоны,
   а не на её мгновенный подъём ``T_ref(t) - T_ref(0)``. Мгновенный подъём сам
   затухает к нулю по мере остывания панели, проходит через ноль и меняет знак,
   поэтому контраст на хвосте записи расходится — ровно там, где живёт
   информация о верхних уровнях заполнения ячеек. Кроме того, в момент t=0
   такой знаменатель строго равен нулю.

2. Логарифмическая временная сетка начинается ПОСЛЕ импульса нагрева
   (``t_min = 5.5 с`` при длительности импульса 5 с). При ``t_min = 0.1 с``
   почти половина точек сетки попадает в фазу нагрева, и временной энкодер
   получает на вход переходный процесс лампы вместо кривой остывания.
"""
from __future__ import annotations

import numpy as np

# Минимальный пиковый подъём опорной зоны (К), ниже которого запись считается
# браком: нагрев не состоялся, нормировать не на что.
MIN_PEAK_RISE_K = 1e-3

# Длительность импульса нагрева по методике контроля, с.
DEFAULT_PULSE_S = 5.0

# Запас после импульса перед началом лог-сетки, с. Отсекает остаточное
# послесвечение лампы и переходный процесс сенсора.
POST_PULSE_MARGIN_S = 0.5

# Число каналов входного тензора модели.
N_CHANNELS = 3


def normalized_contrast(
    cube: np.ndarray, ref_mask: np.ndarray, pulse_end_idx: int
) -> np.ndarray:
    """Нормированный тепловой контраст относительно бездефектной зоны.

    ``C(t) = (T_ref(t) - T(t)) / max_t(T_ref(t) - T_ref(0))``

    Знаменатель — СКАЛЯР: пиковый подъём опорной зоны, взятый на интервале
    импульса. Нормировка делает контраст инвариантным к мощности нагрева
    и не разваливается на хвосте записи (см. докстринг модуля).

    Parameters
    ----------
    cube : np.ndarray
        (H, W, N_frames) float32 — температурное поле во времени.
    ref_mask : np.ndarray
        (H, W) bool — маска бездефектного (опорного) участка панели.
    pulse_end_idx : int
        Индекс кадра окончания импульса нагрева (при fps=10 и импульсе 5 с — 50).
        Пик подъёма ищется на ``[0, pulse_end_idx]`` включительно.

    Returns
    -------
    np.ndarray
        (H, W, N_frames) float32 — контраст.

    Raises
    ------
    ValueError
        Если формы входов несогласованы, маска пуста, ``pulse_end_idx``
        вне диапазона, или пиковый подъём опорной зоны меньше
        ``MIN_PEAK_RISE_K`` (нагрев не состоялся, запись брак).
    """
    ref_curve = reference_curve(cube, ref_mask)
    n_frames = ref_curve.shape[0]

    if not 0 <= pulse_end_idx < n_frames:
        raise ValueError(
            f"pulse_end_idx={pulse_end_idx} вне диапазона [0, {n_frames - 1}]"
        )

    rise = ref_curve - ref_curve[0]
    peak_rise = float(rise[: pulse_end_idx + 1].max())
    if peak_rise < MIN_PEAK_RISE_K:
        raise ValueError(
            f"Пиковый подъём опорной зоны {peak_rise:.3e} К < {MIN_PEAK_RISE_K:.1e} К: "
            "нагрев не состоялся, запись бракованная"
        )

    contrast = (ref_curve[None, None, :] - cube) / peak_rise
    return contrast.astype(np.float32)


def reference_curve(cube: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Средняя температурная кривая опорной зоны ``T_ref(t)``.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, N_frames).
    ref_mask : np.ndarray
        (H, W) bool — непустая маска опорного участка.

    Returns
    -------
    np.ndarray
        (N_frames,) float64 — усреднение по пикселям маски.
    """
    if cube.ndim != 3:
        raise ValueError(f"Ожидается куб (H, W, N), получено {cube.shape}")
    if ref_mask.shape != cube.shape[:2]:
        raise ValueError(
            f"Маска {ref_mask.shape} не совпадает с кадром {cube.shape[:2]}"
        )

    mask = np.asarray(ref_mask, dtype=bool)
    if not mask.any():
        raise ValueError("Опорная маска пуста: нечего брать за бездефектную зону")

    return cube[mask].mean(axis=0).astype(np.float64)


#: Ограничение итоговой производной. У полубесконечного тела наклон -0.5;
#: даже с большим запасом на подповерхностные неоднородности значения за
#: пределами [-20, 20] — не физический сигнал, а всплеск на переходе
#: контраста через ноль (см. ниже). Диапазон согласован с тем, что даёт
#: гладкая физическая синтетика (SyntheticCurveDataset): без ограничения
#: реальные данные, где контраст на 20-30% пикселей проходит рядом с нулём
#: из-за боковой диффузии и шума, давали производную вплоть до ±120 —
#: сеть, обученная на синтетике с диапазоном ±17, получала на fine-tune
#: вход далеко за пределами того, что видела на этапе A.
SLOPE_CLIP: float = 20.0


def log_derivative(
    contrast: np.ndarray, t: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Логарифмическая производная ``d ln(dT) / d ln(t)``.

    Классический признак импульсной термографии: форма кривой остывания
    независимо от её амплитуды. Для полубесконечного тела наклон равен -0.5,
    отклонения указывают на подповерхностную неоднородность.

    Производная считается центральными разностями по ``ln t``; ``eps``
    защищает логарифм от нулевых и отрицательных значений контраста.
    Результат обрезается до ``[-SLOPE_CLIP, SLOPE_CLIP]``: сигнал вблизи
    перехода через ноль (на реальных данных — 20-30% пикселей, боковая
    диффузия и шум) даёт не физическую особенность, а численный артефакт
    производной, а не саму форму кривой.

    Parameters
    ----------
    contrast : np.ndarray
        (H, W, N_frames) — контраст или любой перегрев.
    t : np.ndarray
        (N_frames,) — моменты времени, с. Должны быть положительными.
    eps : float
        Порог отсечки под логарифмом.

    Returns
    -------
    np.ndarray
        (H, W, N_frames) float32, значения в [-SLOPE_CLIP, SLOPE_CLIP].
    """
    if contrast.ndim != 3:
        raise ValueError(f"Ожидается куб (H, W, N), получено {contrast.shape}")
    if t.ndim != 1 or t.shape[0] != contrast.shape[2]:
        raise ValueError(
            f"Ось времени {t.shape} не совпадает с кубом {contrast.shape}"
        )
    if t.shape[0] < 2:
        raise ValueError("Для производной нужно минимум 2 кадра")
    if np.any(t <= 0.0):
        raise ValueError("Все моменты времени должны быть положительными для ln(t)")

    log_t = np.log(np.asarray(t, dtype=np.float64))
    log_signal = np.log(np.maximum(np.abs(contrast).astype(np.float64), eps))

    derivative = np.gradient(log_signal, log_t, axis=2)
    derivative = np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)
    derivative = np.clip(derivative, -SLOPE_CLIP, SLOPE_CLIP)
    return derivative.astype(np.float32)


def resample_log_grid(
    cube: np.ndarray,
    t: np.ndarray,
    n_points: int = 64,
    t_min: float = 5.5,
    t_max: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ресемплинг куба на равномерную по ``log(t)`` сетку ПОСЛЕ импульса.

    Равномерность по логарифму времени соответствует физике диффузии тепла:
    ранние отсчёты несут информацию о приповерхностных дефектах, поздние —
    о глубоких, и в лог-масштабе они равнозначны.

    Сетка стартует с ``t_min = 5.5 с``, то есть после импульса нагрева
    длительностью 5 с. При ``t_min = 0.1 с`` 31 точка из 64 попала бы в фазу
    нагрева (см. докстринг модуля). При ``[5.5, 300]`` и ``n_points = 64``
    минимальный шаг сетки ~0.36 с — крупнее периода кадра 0.1 с,
    передискретизации нет.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, N_frames) — куб произвольной величины (контраст, перегрев).
    t : np.ndarray
        (N_frames,) — исходные моменты времени, с.
    n_points : int
        Число точек выходной сетки.
    t_min, t_max : float
        Границы сетки, с. ``t_max`` подрезается по фактической длине записи.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(resampled (H, W, n_points) float32, t_grid (n_points,) float64)``.
    """
    if cube.ndim != 3:
        raise ValueError(f"Ожидается куб (H, W, N), получено {cube.shape}")
    if t.ndim != 1 or t.shape[0] != cube.shape[2]:
        raise ValueError(f"Ось времени {t.shape} не совпадает с кубом {cube.shape}")
    if n_points < 2:
        raise ValueError(f"n_points должен быть >= 2, получено {n_points}")
    if t_min <= 0.0:
        raise ValueError(f"t_min должен быть положительным, получено {t_min}")

    t = np.asarray(t, dtype=np.float64)
    # Не экстраполируем за пределы записи: короткая запись укорачивает сетку.
    upper = min(float(t_max), float(t[-1]))
    if upper <= t_min:
        raise ValueError(
            f"Запись слишком короткая: доступно до {t[-1]:.2f} с, требуется > {t_min} с"
        )

    t_grid = np.logspace(np.log10(t_min), np.log10(upper), n_points)

    height, width, _ = cube.shape
    flat = cube.reshape(-1, cube.shape[2]).astype(np.float64)
    resampled = np.empty((flat.shape[0], n_points), dtype=np.float64)
    for i in range(flat.shape[0]):
        resampled[i] = np.interp(t_grid, t, flat[i])

    return resampled.reshape(height, width, n_points).astype(np.float32), t_grid


def build_input_tensor(
    cube: np.ndarray,
    ref_mask: np.ndarray,
    fps: float,
    pulse_end_s: float = DEFAULT_PULSE_S,
    n_points: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Полный пайплайн: сырой куб термограмм -> трёхканальный вход модели.

    Каналы:
        0. ``C(t)`` — нормированный контраст (инвариантен к мощности нагрева);
        1. ``d ln dT / d ln t`` — форма кривой остывания без амплитуды;
        2. ``dT``, нормированный покадрово на свой максимум по модулю —
           пространственный рисунок перегрева, устойчивый к дрейфу амплитуды.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, N_frames) — сырой куб температур.
    ref_mask : np.ndarray
        (H, W) bool — маска бездефектной опорной зоны.
    fps : float
        Частота регистрации кадров, Гц.
    pulse_end_s : float
        Момент окончания импульса нагрева, с.
    n_points : int
        Число точек логарифмической временной сетки.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(tensor (3, n_points, H, W) float32, t_grid (n_points,) float64)``.
    """
    if cube.ndim != 3:
        raise ValueError(f"Ожидается куб (H, W, N), получено {cube.shape}")
    if fps <= 0.0:
        raise ValueError(f"fps должен быть положительным, получено {fps}")
    if pulse_end_s < 0.0:
        raise ValueError(f"pulse_end_s должен быть неотрицательным, получено {pulse_end_s}")

    n_frames = cube.shape[2]
    t = np.arange(n_frames, dtype=np.float64) / fps
    pulse_end_idx = min(int(round(pulse_end_s * fps)), n_frames - 1)

    contrast = normalized_contrast(cube, ref_mask, pulse_end_idx)
    delta = (cube - cube[:, :, :1]).astype(np.float32)

    t_min = pulse_end_s + POST_PULSE_MARGIN_S
    contrast_grid, t_grid = resample_log_grid(
        contrast, t, n_points=n_points, t_min=t_min
    )
    delta_grid, _ = resample_log_grid(delta, t, n_points=n_points, t_min=t_min)

    # ВАЖНО: производная берётся от CONTRAST, а не от delta (сырого перегрева
    # в кельвинах). delta = cube - cube[:,:,:1] на реальных записях меняет
    # знак у ~2/3 пикселей (пиксель может стать холоднее опорного кадра из-за
    # дрейфа фона), и log(|delta|) взрывается на переходе через ноль — на
    # water2 это давало диапазон производной [-120, 98] против [-17, 13.6] у
    # синтетики (SyntheticCurveDataset.build_channels), которая всегда берёт
    # производную от нормированного контраста. Рассогласование источника
    # обесценивало перенос весов с этапа A: сеть видела на fine-tune вход
    # вне диапазона, на котором обучалась, и коллапсировала в константу.
    slope_grid = log_derivative(contrast_grid, t_grid)
    delta_norm = _normalize_per_frame(delta_grid)

    # (H, W, n_points) -> (n_points, H, W) для каждого канала.
    channels = [contrast_grid, slope_grid, delta_norm]
    tensor = np.stack([np.moveaxis(ch, 2, 0) for ch in channels], axis=0)
    return tensor.astype(np.float32), t_grid


def _normalize_per_frame(cube: np.ndarray) -> np.ndarray:
    """Покадровая нормировка на максимум модуля: каждый кадр приводится к [-1, 1].

    Убирает общий спад амплитуды со временем и оставляет пространственный
    рисунок — то, что модели нужно от третьего канала.
    """
    scale = np.abs(cube).max(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (cube / scale).astype(np.float32)
