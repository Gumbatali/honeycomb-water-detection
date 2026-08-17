"""Неявный 1D конечно-разностный солвер теплопроводности по глубине ячейки.

В отличие от модели сосредоточенных параметров (`heat_transfer_model`), здесь
столб «обшивка + заполнитель + воздух» разбивается на узлы, и уравнение
теплопроводности решается по глубине. Это воспроизводит главный физический
эффект метода: тепловая волна доходит до дна столба вещества за время
t ~ d^2 / (pi * a), и только после этого кривая остывания поверхности
«узнаёт» о степени заполнения ячейки.

Схема ОБЯЗАТЕЛЬНО неявная (backward Euler). Явная схема расходится: для
воздуха a = 2.16e-5 м^2/с, и при dt = 1e-3 c, dx = H/120 число Фурье
Fo = a*dt/dx^2 = 3.10 >> 0.5.

Уравнение берётся в консервативной форме, а проводимость на гранях узлов
усредняется ГАРМОНИЧЕСКИ — на скачке свойств вода/воздух (k отличается в
20 раз) арифметическое среднее даёт качественно неверный поток.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from src.synthesis.heat_transfer_model import (
    AIR,
    EPOXY,
    GEL,
    WATER,
    Material,
)

# Обшивка (стеклопластик) — отдельный «материал» стека по глубине.
SKIN = Material("skin", density=1800.0, heat_capacity=1000.0, conductivity=0.3)
SKIN_THICKNESS_M = 0.2e-3

# Коды слоёв в маске, возвращаемой `build_1d_stack`.
LAYER_SKIN = 0
LAYER_SUBSTANCE = 1
LAYER_AIR = 2

# Вещества-заполнители, доступные по имени.
SUBSTANCES: dict[str, Material] = {
    "empty": AIR,
    "air": AIR,
    "water": WATER,
    "gel": GEL,
    "epoxy": EPOXY,
}

# Кодировка классов в синтетическом датасете.
SUBSTANCE_CLASSES: tuple[str, ...] = ("empty", "water", "epoxy")

# Окно ресемплинга: начинается ПОСЛЕ импульса нагрева (5.0 c), иначе треть
# точек лог-сетки попадает внутрь фазы нагрева и несёт не остывание, а форму импульса.
T_RESAMPLE_MIN_S = 5.5
T_RESAMPLE_MAX_S = 300.0

# Шаг интегрирования для массовой генерации. Схема безусловно устойчива, и
# при dt = 0.05 c отклонение от dt = 0.01 c составляет 0.54% от пика —
# на порядок ниже шума тепловизора, зато генерация быстрее в ~14 раз.
DATASET_DT_S = 0.05


def build_1d_stack(
    fill_fraction: float,
    substance: str,
    cell_height_m: float = 0.010,
    n_nodes: int = 120,
    skin_thickness_m: float = SKIN_THICKNESS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Собирает профили теплофизических свойств по глубине ячейки.

    Стек: обшивка 0.2 мм -> вещество на глубину `fill_fraction * H` ->
    остаток заполнен воздухом.

    Args:
        fill_fraction: доля заполнения ячейки веществом, 0..1.
        substance: имя вещества (`empty`/`water`/`gel`/`epoxy`).
        cell_height_m: полная глубина ячейки H, м.
        n_nodes: число узлов сетки по глубине.
        skin_thickness_m: толщина обшивки, м.

    Returns:
        (k, rho_cp, layers) — теплопроводность Вт/(м*К), объёмная теплоёмкость
        Дж/(м^3*К) и маска слоёв (`LAYER_*`). Каждый массив формы (n_nodes,).

    Raises:
        ValueError: если доля заполнения вне [0, 1] или вещество неизвестно.
    """
    if not 0.0 <= fill_fraction <= 1.0:
        raise ValueError(f"fill_fraction должен быть в [0, 1], получено {fill_fraction}")
    if substance not in SUBSTANCES:
        raise ValueError(f"Неизвестное вещество {substance!r}, ожидается {sorted(SUBSTANCES)}")
    if n_nodes < 4:
        raise ValueError(f"n_nodes должен быть >= 4, получено {n_nodes}")

    filler = SUBSTANCES[substance]
    dx = cell_height_m / n_nodes
    # Координаты центров узлов — по ним и определяется принадлежность слою.
    depth = (np.arange(n_nodes, dtype=np.float64) + 0.5) * dx

    substance_end = skin_thickness_m + fill_fraction * cell_height_m
    layers = np.full(n_nodes, LAYER_AIR, dtype=np.int8)
    layers[depth < substance_end] = LAYER_SUBSTANCE
    layers[depth < skin_thickness_m] = LAYER_SKIN

    k = np.empty(n_nodes, dtype=np.float64)
    rho_cp = np.empty(n_nodes, dtype=np.float64)
    for code, material in ((LAYER_SKIN, SKIN), (LAYER_SUBSTANCE, filler), (LAYER_AIR, AIR)):
        mask = layers == code
        k[mask] = material.conductivity
        rho_cp[mask] = material.volumetric_capacity

    return k, rho_cp, layers


def _harmonic_face_conductivity(k: np.ndarray) -> np.ndarray:
    """Гармоническое среднее k на гранях между соседними узлами.

    Гармоническое среднее — точное для последовательно соединённых
    сопротивлений. На скачке вода/воздух арифметическое среднее завышает
    поток почти в 10 раз.

    Returns:
        (n_nodes - 1,) — проводимость на гранях i+1/2.
    """
    left, right = k[:-1], k[1:]
    return 2.0 * left * right / (left + right)


def _build_banded_matrix(
    k: np.ndarray, rho_cp: np.ndarray, dx: float, dt: float
) -> np.ndarray:
    """Собирает трёхдиагональную матрицу неявной схемы в banded-формате.

    Дискретизация консервативная:
        C_i * (T_i^{n+1} - T_i^n) / dt =
            [k_{i+1/2}(T_{i+1} - T_i) - k_{i-1/2}(T_i - T_{i-1})] / dx^2

    Матрица не зависит от времени, поэтому собирается один раз до цикла.

    Returns:
        (3, n_nodes) — строки [верхняя диагональ, главная, нижняя] для
        `scipy.linalg.solve_banded` с (l, u) = (1, 1).
    """
    n = k.size
    k_face = _harmonic_face_conductivity(k)
    # Коэффициенты обмена узла i с соседями, приведённые к 1/с.
    alpha = np.zeros(n)  # связь с i-1
    beta = np.zeros(n)  # связь с i+1
    alpha[1:] = dt * k_face / (rho_cp[1:] * dx * dx)
    beta[:-1] = dt * k_face / (rho_cp[:-1] * dx * dx)

    banded = np.zeros((3, n), dtype=np.float64)
    banded[0, 1:] = -beta[:-1]  # верхняя диагональ, сдвиг вправо
    banded[1, :] = 1.0 + alpha + beta  # главная диагональ
    banded[2, :-1] = -alpha[1:]  # нижняя диагональ, сдвиг влево
    return banded


def simulate_cooling_curve(
    fill_fraction: float,
    substance: str,
    power_w_per_m2: float = 5000.0,
    pulse_duration_s: float = 5.0,
    t_eval_s: np.ndarray | None = None,
    dt_s: float = 1e-2,
    cell_height_m: float = 0.010,
    n_nodes: int = 120,
    skin_thickness_m: float = SKIN_THICKNESS_M,
) -> np.ndarray:
    """Считает кривую перегрева поверхности неявной FD-схемой.

    Начальная температура нулевая всюду — работаем в контрасте к фону.
    Импульс мощности подаётся на поверхностный узел первые
    `pulse_duration_s` секунд, дальше поток нулевой. Обе границы столба
    адиабатические (потери в среду в контрастной постановке малы на фоне
    перераспределения тепла по глубине).

    Args:
        fill_fraction: доля заполнения ячейки веществом, 0..1.
        substance: имя вещества (`empty`/`water`/`gel`/`epoxy`).
        power_w_per_m2: поглощённая плотность потока, Вт/м^2.
        pulse_duration_s: длительность импульса нагрева, с.
        t_eval_s: моменты, на которые нужен ответ; по умолчанию 0..300 c с шагом 0.1 c.
        dt_s: шаг интегрирования, с.
        cell_height_m: глубина ячейки H, м.
        n_nodes: число узлов сетки.
        skin_thickness_m: толщина обшивки, м.

    Returns:
        (len(t_eval_s),) float64 — перегрев поверхностного узла, К.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s должен быть положительным, получено {dt_s}")

    times = np.arange(0.0, 300.0 + 1e-9, 0.1) if t_eval_s is None else np.asarray(t_eval_s, dtype=np.float64)
    if times.size == 0:
        return np.zeros(0, dtype=np.float64)
    if times.min() < 0:
        raise ValueError("t_eval_s не может содержать отрицательные моменты")

    k, rho_cp, _ = build_1d_stack(
        fill_fraction, substance, cell_height_m, n_nodes, skin_thickness_m
    )
    dx = cell_height_m / n_nodes
    banded = _build_banded_matrix(k, rho_cp, dx, dt_s)

    # Импульс греет поверхностный узел: dT_0 += dt * q / (C_0 * dx).
    pulse_increment = dt_s * power_w_per_m2 / (rho_cp[0] * dx)

    n_steps = int(np.ceil(times.max() / dt_s))
    history = np.empty(n_steps + 1, dtype=np.float64)
    temperature = np.zeros(n_nodes, dtype=np.float64)
    history[0] = 0.0

    for step in range(1, n_steps + 1):
        rhs = temperature
        if (step - 1) * dt_s < pulse_duration_s:
            rhs = rhs + np.concatenate(([pulse_increment], np.zeros(n_nodes - 1)))
        temperature = solve_banded((1, 1), banded, rhs)
        history[step] = temperature[0]

    grid = np.arange(n_steps + 1, dtype=np.float64) * dt_s
    return np.interp(times, grid, history)


def log_time_grid(n_time_points: int = 64) -> np.ndarray:
    """Логарифмическая сетка моментов наблюдения после импульса нагрева."""
    return np.logspace(
        np.log10(T_RESAMPLE_MIN_S), np.log10(T_RESAMPLE_MAX_S), n_time_points
    )


def generate_synthetic_dataset(
    n_samples: int = 20000,
    seed: int = 0,
    n_time_points: int = 64,
    dt_s: float = DATASET_DT_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Генерирует синтетический датасет кривых остывания.

    Разброс для реализма: толщина обшивки +-10%, мощность нагрева +-15%.
    Кривые считаются на логарифмической сетке времени в
    [`T_RESAMPLE_MIN_S`, `T_RESAMPLE_MAX_S`] — то есть уже после импульса.

    Args:
        n_samples: число примеров.
        seed: зерно генератора.
        n_time_points: число отсчётов по времени в одной кривой.
        dt_s: шаг интегрирования, см. `DATASET_DT_S`.

    Returns:
        (curves, substance_labels, fill_labels) — кривые (n_samples, n_time_points)
        float32, метки вещества int {0: empty, 1: water, 2: epoxy} и доля
        заполнения float32 (NaN там, где вещество не вода).
    """
    rng = np.random.default_rng(seed)
    times = log_time_grid(n_time_points)

    curves = np.empty((n_samples, n_time_points), dtype=np.float32)
    substance_labels = rng.integers(0, len(SUBSTANCE_CLASSES), size=n_samples).astype(np.int64)
    fill_labels = np.full(n_samples, np.nan, dtype=np.float32)

    for index in range(n_samples):
        name = SUBSTANCE_CLASSES[substance_labels[index]]
        fill = 0.0 if name == "empty" else float(rng.uniform(0.05, 1.0))
        power = 5000.0 * float(rng.uniform(0.85, 1.15))
        skin = SKIN_THICKNESS_M * float(rng.uniform(0.9, 1.1))

        curves[index] = simulate_cooling_curve(
            fill,
            name,
            power_w_per_m2=power,
            t_eval_s=times,
            dt_s=dt_s,
            skin_thickness_m=skin,
        ).astype(np.float32)
        if name == "water":
            fill_labels[index] = np.float32(fill)

    return curves, substance_labels, fill_labels
