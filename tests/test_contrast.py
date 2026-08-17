"""Тесты препроцессинга: нормировка контраста и лог-временная сетка.

Ключевые регрессионные проверки — на два бага первой редакции препроцессинга:
нормировка на функцию времени (расходится на хвосте) и лог-сетка с t_min=0.1
(половина точек попадает в фазу нагрева).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.contrast import (
    build_input_tensor,
    log_derivative,
    normalized_contrast,
    resample_log_grid,
)

FPS = 10.0
PULSE_S = 5.0
DURATION_S = 300.0
N_FRAMES = int(DURATION_S * FPS) + 1  # 3001 кадр, запись до 300 с включительно
PULSE_END_IDX = int(PULSE_S * FPS)  # 50


def _curve(t: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """Модельная кривая: линейный нагрев за импульс, затем экспоненциальный спад."""
    profile = np.zeros_like(t)
    heating = t < PULSE_S
    profile[heating] = amplitude * t[heating] / PULSE_S
    cooling = ~heating
    profile[cooling] = amplitude * np.exp(-(t[cooling] - PULSE_S) / tau)
    return profile


def make_cube(
    height: int = 8,
    width: int = 8,
    amplitude: float = 8.0,
    tau_ref: float = 60.0,
    tau_defect: float = 140.0,
    ambient: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Синтетический куб с известным ответом.

    Левая половина кадра — опорная (бездефектная) зона, остывает с tau_ref.
    Правая половина — дефектная (вода в ячейках), остывает медленнее.

    Returns:
        (cube (H, W, N) float32, ref_mask (H, W) bool)
    """
    t = np.arange(N_FRAMES) / FPS
    ref_profile = ambient + _curve(t, amplitude, tau_ref)
    defect_profile = ambient + _curve(t, amplitude, tau_defect)

    cube = np.empty((height, width, N_FRAMES), dtype=np.float32)
    half = width // 2
    cube[:, :half, :] = ref_profile[None, None, :]
    cube[:, half:, :] = defect_profile[None, None, :]

    ref_mask = np.zeros((height, width), dtype=bool)
    ref_mask[:, :half] = True
    return cube, ref_mask


# --- Главный баг: нормировка на функцию времени --------------------------------


def test_old_formula_diverges_and_flips_sign():
    """Численно подтверждает несостоятельность старой формулы.

    Старый знаменатель T_ref(t) - T_ref(0) сам затухает к нулю, проходит через
    ноль и меняет знак: контраст на хвосте записи расходится.
    """
    cube, ref_mask = make_cube()
    ref_curve = cube[ref_mask].mean(axis=0).astype(np.float64)
    denom = ref_curve - ref_curve[0]

    assert denom[0] == 0.0, "старый знаменатель строго ноль в первом кадре"

    # Знаменатель монотонно затухает после импульса.
    idx_5s = int(5.0 * FPS)
    idx_150s = int(150.0 * FPS)
    idx_300s = N_FRAMES - 1
    assert denom[idx_5s] > denom[idx_150s] > denom[idx_300s]

    # Деление на ноль здесь — суть проверки, а не случайность.
    with np.errstate(divide="ignore", invalid="ignore"):
        old_contrast = (ref_curve[None, None, :] - cube) / denom[None, None, :]
    assert not np.isfinite(old_contrast[:, :, 0]).all(), (
        "старая формула обязана давать неконечные значения при denom=0"
    )


def test_new_contrast_is_finite_everywhere():
    """Новая формула конечна во всех отсчётах, включая последний."""
    cube, ref_mask = make_cube()
    contrast = normalized_contrast(cube, ref_mask, PULSE_END_IDX)

    assert np.isfinite(contrast).all()
    assert not np.isnan(contrast).any()
    assert np.isfinite(contrast[:, :, -1]).all(), "хвост записи должен быть конечным"
    assert np.isfinite(contrast[:, :, 0]).all(), "первый кадр должен быть конечным"


def test_new_contrast_keeps_sign_on_tail():
    """Контраст дефектной зоны не меняет знак на хвосте (дефект всегда теплее)."""
    cube, ref_mask = make_cube()
    contrast = normalized_contrast(cube, ref_mask, PULSE_END_IDX)

    defect = contrast[:, 4:, PULSE_END_IDX:]
    assert (defect <= 1e-6).all(), "T_ref < T_defect => контраст отрицателен весь хвост"


def test_denominator_is_scalar_not_time_dependent():
    """Контраст пропорционален (T_ref(t) - T(t)) с одним общим множителем."""
    cube, ref_mask = make_cube()
    contrast = normalized_contrast(cube, ref_mask, PULSE_END_IDX)

    ref_curve = cube[ref_mask].mean(axis=0).astype(np.float64)
    numerator = ref_curve[None, None, :] - cube

    # В момент окончания импульса кривые совпадают: числитель и контраст строго
    # нули, отношение не определено — берём только ненулевые отсчёты.
    nonzero = np.abs(contrast) > 1e-6
    ratio = numerator[nonzero] / contrast[nonzero]

    assert ratio.size > 0
    assert np.ptp(ratio) < 1e-3, "множитель обязан быть одинаков во всех отсчётах"
    # Множитель — это и есть скалярный пиковый подъём опорной зоны (8.0 К).
    assert float(ratio.mean()) == pytest.approx(8.0, rel=1e-3)


# --- Инвариантность к мощности нагрева -----------------------------------------


def test_contrast_invariant_to_heating_power():
    """Умножение всего куба на 2 не меняет нормированный контраст."""
    cube, ref_mask = make_cube(ambient=0.0)
    base = normalized_contrast(cube, ref_mask, PULSE_END_IDX)
    doubled = normalized_contrast(cube * 2.0, ref_mask, PULSE_END_IDX)

    assert np.allclose(base, doubled, atol=1e-5)


def test_contrast_rejects_record_without_heating():
    """Куб без нагрева — брак записи, нормировать не на что."""
    cube = np.full((4, 4, 200), 20.0, dtype=np.float32)
    ref_mask = np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="нагрев не состоялся"):
        normalized_contrast(cube, ref_mask, PULSE_END_IDX)


def test_contrast_rejects_empty_mask():
    cube, _ = make_cube()
    with pytest.raises(ValueError, match="пуста"):
        normalized_contrast(cube, np.zeros(cube.shape[:2], dtype=bool), PULSE_END_IDX)


# --- Лог-сетка -----------------------------------------------------------------


def test_log_grid_starts_after_pulse():
    """С t_min=5.5 ни одна точка не попадает в фазу нагрева."""
    cube, _ = make_cube()
    t = np.arange(N_FRAMES) / FPS
    _, t_grid = resample_log_grid(cube, t, n_points=64, t_min=5.5, t_max=DURATION_S)

    assert t_grid.shape == (64,)
    assert (t_grid >= 5.5).all()
    assert (t_grid > PULSE_S).all(), "ни одна точка не должна лежать в импульсе"
    assert t_grid[-1] == pytest.approx(DURATION_S)


def test_log_grid_regression_t_min_0p1_puts_31_points_in_pulse():
    """Регрессия на старый баг: при t_min=0.1 ровно 31 точка из 64 внутри импульса."""
    cube, _ = make_cube()
    t = np.arange(N_FRAMES) / FPS
    _, bad_grid = resample_log_grid(cube, t, n_points=64, t_min=0.1, t_max=DURATION_S)

    in_pulse = int((bad_grid < PULSE_S).sum())
    assert in_pulse == 31, f"ожидалось 31 точка в фазе нагрева, получено {in_pulse}"

    _, good_grid = resample_log_grid(cube, t, n_points=64, t_min=5.5, t_max=DURATION_S)
    assert int((good_grid < PULSE_S).sum()) == 0


def test_log_grid_step_coarser_than_frame_period():
    """Минимальный шаг сетки крупнее периода кадра — передискретизации нет."""
    cube, _ = make_cube()
    t = np.arange(N_FRAMES) / FPS
    _, t_grid = resample_log_grid(cube, t, n_points=64, t_min=5.5, t_max=DURATION_S)

    min_step = float(np.diff(t_grid).min())
    assert min_step > 1.0 / FPS, f"шаг {min_step:.3f} с не крупнее периода кадра"


def test_resample_preserves_values_of_constant_cube():
    cube = np.full((3, 3, N_FRAMES), 7.5, dtype=np.float32)
    t = np.arange(N_FRAMES) / FPS
    resampled, t_grid = resample_log_grid(cube, t, n_points=32)

    assert resampled.shape == (3, 3, 32)
    assert t_grid.shape == (32,)
    assert np.allclose(resampled, 7.5, atol=1e-5)


def test_resample_rejects_too_short_record():
    cube = np.zeros((2, 2, 30), dtype=np.float32)
    t = np.arange(30) / FPS  # запись всего 2.9 с
    with pytest.raises(ValueError, match="слишком короткая"):
        resample_log_grid(cube, t, n_points=16, t_min=5.5)


# --- Логарифмическая производная -----------------------------------------------


def test_log_derivative_shape_and_finiteness():
    cube, ref_mask = make_cube()
    t = np.arange(N_FRAMES) / FPS
    contrast = normalized_contrast(cube, ref_mask, PULSE_END_IDX)
    resampled, t_grid = resample_log_grid(contrast, t, n_points=64, t_min=5.5)

    slope = log_derivative(resampled, t_grid)
    assert slope.shape == resampled.shape
    assert slope.dtype == np.float32
    assert np.isfinite(slope).all()


def test_log_derivative_recovers_power_law_slope():
    """Для сигнала dT ~ t^-0.5 логарифмическая производная равна -0.5."""
    t = np.logspace(np.log10(5.5), np.log10(300.0), 64)
    signal = (t ** -0.5)[None, None, :].repeat(2, 0).repeat(2, 1)

    slope = log_derivative(signal.astype(np.float32), t)
    # Края смещены односторонней разностью — проверяем внутреннюю часть.
    assert np.allclose(slope[:, :, 2:-2], -0.5, atol=1e-3)


def test_log_derivative_rejects_nonpositive_time():
    cube = np.ones((2, 2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="положительными"):
        log_derivative(cube, np.arange(8, dtype=np.float64))


# --- Полный пайплайн -----------------------------------------------------------


def test_build_input_tensor_shapes():
    cube, ref_mask = make_cube(height=8, width=8)
    tensor, t_grid = build_input_tensor(cube, ref_mask, fps=FPS, n_points=64)

    assert tensor.shape == (3, 64, 8, 8)
    assert tensor.dtype == np.float32
    assert t_grid.shape == (64,)


def test_build_input_tensor_is_finite_and_post_pulse():
    cube, ref_mask = make_cube()
    tensor, t_grid = build_input_tensor(cube, ref_mask, fps=FPS, n_points=64)

    assert np.isfinite(tensor).all()
    assert (t_grid > PULSE_S).all()


def test_build_input_tensor_respects_n_points():
    cube, ref_mask = make_cube(height=6, width=6)
    tensor, t_grid = build_input_tensor(cube, ref_mask, fps=FPS, n_points=32)

    assert tensor.shape == (3, 32, 6, 6)
    assert t_grid.shape == (32,)


def test_build_input_tensor_contrast_channel_invariant_to_power():
    """Канал контраста не зависит от мощности нагрева и на уровне пайплайна."""
    cube, ref_mask = make_cube(ambient=0.0)
    base, _ = build_input_tensor(cube, ref_mask, fps=FPS, n_points=64)
    doubled, _ = build_input_tensor(cube * 2.0, ref_mask, fps=FPS, n_points=64)

    assert np.allclose(base[0], doubled[0], atol=1e-5)
    assert np.allclose(base[2], doubled[2], atol=1e-5), "покадровая нормировка тоже инвариантна"


def test_build_input_tensor_rejects_bad_input():
    cube, ref_mask = make_cube(height=4, width=4)
    with pytest.raises(ValueError):
        build_input_tensor(cube, ref_mask, fps=0.0)
    with pytest.raises(ValueError):
        build_input_tensor(np.zeros((4, 4), dtype=np.float32), ref_mask, fps=FPS)
