"""Валидация неявного FD-солвера теплопроводности.

Проверяется не только «код не падает», а физика: устойчивость схемы,
время выхода тепловой волны на дно столба вещества, вырождение кривых
при большой степени заполнения и совпадение с аналитикой t^(-1/2)
для однородного полубесконечного тела.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.synthesis.fd_solver import (
    LAYER_AIR,
    LAYER_SKIN,
    LAYER_SUBSTANCE,
    SKIN_THICKNESS_M,
    WATER,
    build_1d_stack,
    generate_synthetic_dataset,
    log_time_grid,
    simulate_cooling_curve,
)

FILL_FRACTIONS = (0.0, 0.2, 0.5, 0.8, 1.0)
SUBSTANCES = ("empty", "water", "epoxy")

# Температуропроводность воды, м^2/с — задаёт скорость тепловой волны.
WATER_DIFFUSIVITY = 1.4376e-7
PULSE_S = 5.0
NETD_K = 0.04  # шумовой порог тепловизора, К


@pytest.fixture(scope="module")
def dense_times() -> np.ndarray:
    """Плотная сетка моментов наблюдения, включая фазу нагрева."""
    return np.arange(0.0, 300.001, 0.5)


def _front_arrival_time(depth_m: float) -> float:
    """Аналитическое время выхода тепловой волны на глубину d: d^2/(pi*a)."""
    return depth_m**2 / (np.pi * WATER_DIFFUSIVITY)


# --- Геометрия стека -------------------------------------------------------


def test_stack_layers_follow_geometry():
    _, _, layers = build_1d_stack(0.5, "water", cell_height_m=0.010, n_nodes=120)

    assert layers[0] == LAYER_SKIN
    assert (layers == LAYER_SUBSTANCE).sum() > 0
    assert layers[-1] == LAYER_AIR
    # Слои идут строго по порядку: обшивка -> вещество -> воздух.
    assert np.all(np.diff(layers) >= 0)


def test_stack_properties_match_materials():
    k, rho_cp, layers = build_1d_stack(1.0, "water")

    water_nodes = layers == LAYER_SUBSTANCE
    assert np.allclose(k[water_nodes], WATER.conductivity)
    assert np.allclose(rho_cp[water_nodes], WATER.volumetric_capacity)


def test_empty_cell_has_no_substance_layer():
    _, _, layers = build_1d_stack(0.0, "empty")
    assert (layers == LAYER_SUBSTANCE).sum() == 0


@pytest.mark.parametrize("bad_fill", [-0.1, 1.5])
def test_invalid_fill_fraction_rejected(bad_fill: float):
    with pytest.raises(ValueError):
        build_1d_stack(bad_fill, "water")


def test_unknown_substance_rejected():
    with pytest.raises(ValueError):
        build_1d_stack(0.5, "mercury")


# --- Устойчивость схемы ----------------------------------------------------


@pytest.mark.parametrize("substance", SUBSTANCES)
@pytest.mark.parametrize("fill", FILL_FRACTIONS)
def test_solution_is_finite_and_nonnegative(
    substance: str, fill: float, dense_times: np.ndarray
):
    """Неявная схема не расходится ни для одного сочетания параметров."""
    curve = simulate_cooling_curve(fill, substance, t_eval_s=dense_times)

    assert curve.shape == dense_times.shape
    assert np.isfinite(curve).all()
    assert (curve >= -1e-9).all()
    assert curve.max() > 0.1


@pytest.mark.parametrize("substance", SUBSTANCES)
@pytest.mark.parametrize("fill", FILL_FRACTIONS)
def test_curve_rises_during_pulse_then_decays(
    substance: str, fill: float, dense_times: np.ndarray
):
    """Монотонный рост во время импульса и монотонный спад после него."""
    curve = simulate_cooling_curve(fill, substance, t_eval_s=dense_times)
    peak_index = int(np.argmax(curve))

    # Допуск на округление float64: пустая ячейка выходит на адиабатическое
    # равновесие, где спад по величине сравним с машинной точностью.
    tolerance = 1e-9 * max(curve.max(), 1.0)

    rising = np.diff(curve[: peak_index + 1])
    falling = np.diff(curve[peak_index:])
    assert (rising >= -tolerance).all()
    assert (falling <= tolerance).all()

    # Пик достигается на импульсе, а не в произвольный момент.
    assert dense_times[peak_index] <= PULSE_S + 1.0


def test_implicit_scheme_stable_at_large_timestep(dense_times: np.ndarray):
    """Явная схема при таком dt разошлась бы (Fo >> 0.5), неявная — нет."""
    curve = simulate_cooling_curve(0.5, "water", t_eval_s=dense_times, dt_s=0.1)

    assert np.isfinite(curve).all()
    assert curve.max() < 1e3


def test_coarse_and_fine_timestep_agree(dense_times: np.ndarray):
    """Решение сошлось по шагу времени: dt=1e-2 и dt=2e-3 дают то же самое."""
    coarse = simulate_cooling_curve(0.5, "water", t_eval_s=dense_times, dt_s=1e-2)
    fine = simulate_cooling_curve(0.5, "water", t_eval_s=dense_times, dt_s=2e-3)

    relative_error = np.abs(coarse - fine).max() / coarse.max()
    assert relative_error < 0.05


# --- Физика: время прихода тепловой волны ----------------------------------


@pytest.mark.parametrize(
    ("fill", "expected_s"),
    [(0.2, _front_arrival_time(0.002)), (0.6, _front_arrival_time(0.006))],
)
def test_front_arrival_matches_diffusion_time(fill: float, expected_s: float):
    """Кривая отклоняется от «полной» ячейки в момент t = d^2/(pi*a).

    Пока тепловая волна не дошла до дна столба воды, поверхность не может
    знать о степени заполнения — кривые обязаны совпадать. Отклонение
    должно появиться около аналитического времени, а не сразу.
    """
    times = np.arange(PULSE_S, 300.001, 0.25)
    full = simulate_cooling_curve(1.0, "water", t_eval_s=times)
    partial = simulate_cooling_curve(fill, "water", t_eval_s=times)

    divergence = np.abs(partial - full) / full.max()
    arrival_s = float(times[int(np.argmax(divergence > 0.01))])

    # Сразу после импульса кривые ещё неразличимы. Окно берётся от начала
    # сетки, но не шире половины аналитического времени прихода.
    early = times <= max(times[0] + 0.5, expected_s * 0.5)
    assert divergence[early].max() < 0.01
    # А в районе аналитического времени расхождение уже есть.
    assert 0.5 * expected_s < arrival_s < 2.0 * expected_s


def test_deeper_fill_diverges_later():
    """Чем глубже столб воды, тем позже поверхность о нём «узнаёт»."""
    times = np.arange(PULSE_S, 300.001, 0.25)
    full = simulate_cooling_curve(1.0, "water", t_eval_s=times)

    arrivals = []
    for fill in (0.2, 0.4, 0.6):
        curve = simulate_cooling_curve(fill, "water", t_eval_s=times)
        divergence = np.abs(curve - full) / full.max()
        arrivals.append(float(times[int(np.argmax(divergence > 0.01))]))

    assert arrivals[0] < arrivals[1] < arrivals[2]


# --- Физика: вырождение больших заполнений ---------------------------------


def test_high_fills_are_indistinguishable_at_30s():
    """80% и 100% неразличимы на 30-й секунде — ожидаемое вырождение.

    Тепловая волна за 30 с проходит sqrt(pi*a*t) ~ 3.7 мм и не успевает
    дойти до дна ни 8-мм, ни 10-мм столба воды, поэтому кривые совпадают
    заведомо точнее шумового порога тепловизора NETD.
    """
    times = np.array([30.0])
    high = simulate_cooling_curve(0.8, "water", t_eval_s=times)[0]
    full = simulate_cooling_curve(1.0, "water", t_eval_s=times)[0]

    peak = simulate_cooling_curve(1.0, "water", t_eval_s=np.arange(0.0, 31.0, 0.5)).max()
    normalized_difference = abs(high - full) / peak

    # Много строже шумового порога — различить 80% и 100% физически нельзя.
    assert normalized_difference < NETD_K / 10.0


def test_small_fills_remain_distinguishable_at_30s():
    """Контроль на осмысленность: 20% и 100% на 30-й секунде различимы."""
    times = np.array([30.0])
    low = simulate_cooling_curve(0.2, "water", t_eval_s=times)[0]
    full = simulate_cooling_curve(1.0, "water", t_eval_s=times)[0]

    peak = simulate_cooling_curve(1.0, "water", t_eval_s=np.arange(0.0, 31.0, 0.5)).max()
    assert abs(low - full) / peak > NETD_K


def test_water_and_empty_are_well_separated():
    """Пустая ячейка почти не отводит тепло — держит перегрев много дольше."""
    times = np.array([60.0])
    empty = simulate_cooling_curve(0.0, "empty", t_eval_s=times)[0]
    water = simulate_cooling_curve(1.0, "water", t_eval_s=times)[0]

    assert empty > 10.0 * water


# --- Сравнение с аналитикой ------------------------------------------------


@pytest.mark.parametrize("substance", ["water", "epoxy"])
def test_semi_infinite_body_decays_as_sqrt_t(substance: str):
    """Однородное полубесконечное тело: T_surface ~ t^(-1/2) после импульса.

    Классическое решение для мгновенного поверхностного источника.
    Проверяется наклон в log-log координатах с допуском +-20%.

    Теплоотдача с поверхности отключена: закон t^(-1/2) выведен для тела
    без потерь в среду, и с ненулевым `surface_loss_w_per_m2_k` наклон
    закономерно круче (-0.66 при h = 10 Вт/(м^2*К)).
    """
    times = np.logspace(np.log10(6.0), np.log10(300.0), 80)
    # Столб 0.5 м с мелкой сеткой: за 300 c волна до дна не доходит,
    # поэтому тело фактически полубесконечное. Обшивка убрана —
    # нужен ОДНОРОДНЫЙ материал во всех узлах.
    curve = simulate_cooling_curve(
        1.0,
        substance,
        t_eval_s=times,
        cell_height_m=0.5,
        n_nodes=1500,
        skin_thickness_m=0.0,
        surface_loss_w_per_m2_k=0.0,
    )

    window = (times >= 20.0) & (times <= 200.0)
    slope = float(np.polyfit(np.log(times[window]), np.log(curve[window]), 1)[0])

    assert slope == pytest.approx(-0.5, abs=0.1)


def test_empty_cell_reaches_adiabatic_equilibrium():
    """Пустая ячейка приходит к равновесию Q/(sum C_i * dx) — проверка энергии.

    Баланс энергии проверяется в ЗАМКНУТОЙ постановке: теплоотдача с
    поверхности отключена, иначе вся закачанная энергия уходит в среду и
    равновесная температура стремится к нулю, а не к Q/(sum C_i * dx).
    """
    _, rho_cp, _ = build_1d_stack(0.0, "empty", n_nodes=120)
    dx = 0.010 / 120
    deposited_j = 5000.0 * PULSE_S
    expected_k = deposited_j / (rho_cp * dx).sum()

    final = simulate_cooling_curve(
        0.0, "empty", t_eval_s=np.array([300.0]), surface_loss_w_per_m2_k=0.0
    )[0]

    assert final == pytest.approx(expected_k, rel=1e-3)


def test_surface_loss_drains_energy_to_ambient():
    """С теплоотдачей ячейка с воздухом остывает, без неё — держит плато.

    Это и есть причина переписывания генератора: при адиабатических границах
    воздушная ячейка (эффузивность 5.6 против 735 у обшивки) не остывает
    вовсе, контраст к ней определяется фактом «воздух против вещества», и
    синтетика остаётся тривиально разделимой по амплитуде.
    """
    times = np.array([5.5, 300.0])

    adiabatic = simulate_cooling_curve(
        0.0, "empty", t_eval_s=times, surface_loss_w_per_m2_k=0.0
    )
    with_loss = simulate_cooling_curve(0.0, "empty", t_eval_s=times)

    # Без потерь плато: температура на 300 c практически та же, что на 5.5 c.
    assert adiabatic[1] == pytest.approx(adiabatic[0], rel=0.02)
    # С потерями энергия уходит в среду.
    assert with_loss[1] < with_loss[0] * 0.01


def test_harmonic_averaging_used_at_water_air_interface():
    """На скачке вода/воздух поток определяется гармоническим средним.

    Арифметическое среднее (k~0.3) завысило бы проводимость границы
    примерно в 6 раз против гармонического (k~0.05).
    """
    from src.synthesis.fd_solver import _harmonic_face_conductivity

    k = np.array([0.6, 0.026])
    face = _harmonic_face_conductivity(k)[0]

    assert face == pytest.approx(2 * 0.6 * 0.026 / 0.626, rel=1e-9)
    assert face < np.mean(k) / 5.0


# --- Датасет ---------------------------------------------------------------


def test_log_time_grid_starts_after_pulse():
    """Сетка начинается после импульса — иначе точки описывают нагрев."""
    grid = log_time_grid(64)

    assert grid.size == 64
    assert grid.min() > PULSE_S
    assert grid.max() == pytest.approx(300.0)
    assert np.all(np.diff(grid) > 0)


def test_dataset_timestep_stays_accurate():
    """Ускоренный шаг генерации не портит кривую заметно для сенсора."""
    from src.synthesis.fd_solver import DATASET_DT_S

    times = log_time_grid(64)
    for fill, substance in ((0.2, "water"), (1.0, "water"), (0.0, "empty")):
        reference = simulate_cooling_curve(fill, substance, t_eval_s=times, dt_s=1e-2)
        fast = simulate_cooling_curve(fill, substance, t_eval_s=times, dt_s=DATASET_DT_S)
        assert np.abs(fast - reference).max() / reference.max() < 0.01


def test_generated_dataset_shapes_and_labels():
    curves, substances, fills = generate_synthetic_dataset(
        n_samples=30, seed=0, n_time_points=32
    )

    assert curves.shape == (30, 32)
    assert curves.dtype == np.float32
    assert substances.shape == (30,)
    assert fills.shape == (30,)
    assert fills.dtype == np.float32
    assert set(np.unique(substances)) <= {0, 1, 2}
    assert np.isfinite(curves).all()


def test_fill_labels_defined_only_for_water():
    _, substances, fills = generate_synthetic_dataset(
        n_samples=60, seed=1, n_time_points=16
    )

    water = substances == 1
    assert np.isfinite(fills[water]).all()
    assert np.isnan(fills[~water]).all()
    assert ((fills[water] > 0.0) & (fills[water] <= 1.0)).all()


def test_dataset_is_reproducible():
    first = generate_synthetic_dataset(n_samples=20, seed=7, n_time_points=16)
    second = generate_synthetic_dataset(n_samples=20, seed=7, n_time_points=16)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_dataset_has_amplitude_spread_from_jitter():
    """Разброс мощности и толщины обшивки даёт неодинаковые амплитуды."""
    curves, substances, _ = generate_synthetic_dataset(
        n_samples=40, seed=3, n_time_points=16
    )

    empty_curves = curves[substances == 0]
    if empty_curves.shape[0] >= 2:
        peaks = empty_curves.max(axis=1)
        assert peaks.std() > 0
