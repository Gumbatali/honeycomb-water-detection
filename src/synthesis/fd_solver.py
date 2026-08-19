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

Поверхность отдаёт тепло в среду (`SURFACE_LOSS_W_PER_M2_K`). Это не деталь
второго порядка: при адиабатических границах ячейка с воздухом не остывает
вовсе, и синтетика вырождается в тривиальный порог по амплитуде.

`generate_synthetic_dataset` отдаёт НОРМИРОВАННЫЙ КОНТРАСТ, а не абсолютные
кельвины::

    C(t) = (T_ref(t) - T(t)) / max_t(T_ref(t) - T_ref(0))

— то же представление, что `preprocessing.contrast.build_input_tensor` даёт на
реальных данных. Единство представления обязательно: иначе этап A
(предобучение на синтетике) и этап B (fine-tune на реальных данных) видят
входы разного масштаба и перенос весов не работает. Подробнее —
ARCHITECTURE.md раздел 3.
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

# Номинальная мощность нагрева, Вт/м^2 (галогенная лампа 500 Вт, раздел 0).
NOMINAL_POWER_W_PER_M2 = 5000.0

#: Калибровка sim-to-real по амплитуде контраста. Измерено на 12 зонах
#: water1+water4: отношение реального контраста к синтетическому равно
#: 0.300 для воды и 0.290 для смолы — систематическое расхождение, а не
#: разброс, поэтому применяется единый множитель. Причина в том, что
#: 1D-модель не воспроизводит боковую диффузию тепла, потери на стенках
#: сот и разброс излучательной способности (ARCHITECTURE.md раздел 10).
#: Без калибровки сеть принимает более слабый реальный сигнал за признак
#: пустой ячейки: замер до правки — все зоны воды на water1 предсказаны
#: как «пусто» с логитами 0.3-3.3.
#:
#: Значение привязано к конкретной установке (лампа 500 Вт, покрытие
#: панели, геометрия сот). При смене условий съёмки требуется пересчёт —
#: см. `scripts/calibrate_sim_to_real.py`.
SIM_TO_REAL_SCALE = 0.295

# Длительность импульса нагрева по методике контроля, с.
PULSE_DURATION_S = 5.0

# NETD камеры Optris PI 450, К (ARCHITECTURE.md раздел 0). Пиксельный шум
# этого уровня — то, что делает пару 80%/100% неразличимой: без него
# синтетика не воспроизводит главное ограничение задачи.
NETD_K = 0.04

# Джиттер условий съёмки, общий для ячейки и её опоры. Дискретизирован по
# сетке, чтобы опорные симуляции кэшировались по ключу
# (power, skin_thickness): без дискретизации каждая опорная кривая уникальна
# и кэш не срабатывает ни разу.
#
# Внимание: этот джиттер в контрасте почти полностью СОКРАЩАЕТСЯ — числитель и
# знаменатель считаются при одних условиях, в чём и смысл нормировки. Поэтому
# межъячеечный разброс он не создаёт, и полагаться на него как на источник
# реалистичного перекрытия классов нельзя (замерено: вклад мощности ровно 0).
POWER_JITTER = 0.15
SKIN_JITTER = 0.10
JITTER_LEVELS = 16

# ЛОКАЛЬНЫЙ разброс на уровне отдельной ячейки, который нормировка на опору НЕ
# убирает, потому что опора — это усреднённая бездефектная зона панели, а не
# двойник конкретной ячейки.
#
# Физические источники: неравномерность освещённости пятна лампы по панели,
# разброс излучательной способности и загрязнения поверхности, отклонение
# фактического пролива от номинала, боковой отток тепла в стенки соты (1D-модель
# его не воспроизводит вовсе).
#
# Без этого члена синтетика остаётся тривиально разделимой: единственным
# источником разброса остаются дискретные скачки толщины обшивки, классы
# ложатся в непересекающиеся полосы, и логистическая регрессия даёт 1.0000 —
# тот самый дефект, ради которого переписан генератор. На реальных данных
# классы перекрываются (0.291 / 0.295 / 0.323 — раздел 11.2).
LOCAL_GAIN_SIGMA = 0.12
LOCAL_FILL_SIGMA = 0.08

# Разброс САМОЙ опорной оценки. На реальной панели опора — одна усреднённая
# бездефектная зона на всю запись, а локальный бездефектный фон под каждой
# ячейкой от неё отличается: подложка, прижим обшивки и освещённость меняются
# по полю кадра. В синтетике опора считалась бы идеально «своей» для каждой
# ячейки, чего на реальных данных не бывает.
#
# Моделируется как ошибка масштаба опорной кривой: она входит и в числитель, и
# в знаменатель контраста, поэтому сдвигает всю кривую целиком, а не добавляет
# независимый шум в каждую точку. Именно такой разброс наблюдается на реальных
# данных, где соседние уровни заполнения расходятся на 0.004..0.03 при
# собственном разбросе того же порядка.
REFERENCE_MISMATCH_SIGMA = 0.10

# Минимальный пиковый подъём опорной кривой, ниже которого нормировать не на
# что. Совпадает по смыслу с `contrast.MIN_PEAK_RISE_K`.
MIN_PEAK_RISE_K = 1e-3

# Диапазоны заполнения. Нижняя граница воды намеренно уходит НИЖЕ реальных
# 20%: контраст воды выходит на насыщение уже к fill ~0.15 (0.387 при 0.413
# для полной ячейки), и без мелких заполнений класс «вода» вырождается в узкую
# полосу около 0.41, отделимую от смолы одним порогом.
#
# Замеренная физика (C(30 с), номинальные условия):
#   вода  fill 0.04 / 0.06 / 0.10 / 0.20 -> 0.294 / 0.337 / 0.366 / 0.398
#   смола fill 0.10 / 0.20 / 0.30 / 0.50 -> 0.262 / 0.334 / 0.363 / 0.375
# То есть на мелких проливах вода и смола перекрываются, а на глубоких смола
# насыщается на 0.375 против 0.413 у воды. Именно это перекрытие и даёт
# реальную картину раздела 11.2, где смола (0.295) лежит между водой 20%
# (0.291) и водой 40% (0.323).
WATER_FILL_RANGE = (0.03, 1.0)
EPOXY_FILL_RANGE = (0.08, 1.0)

# Суммарный коэффициент теплоотдачи с поверхности в среду, Вт/(м^2*К):
# свободная конвекция (~5) плюс линеаризованное излучение
# 4*eps*sigma*T^3 при eps=0.9, T=300 К (~5.5).
#
# Этот член ОБЯЗАТЕЛЕН, а не косметика. Без него обе границы столба
# адиабатические, и ячейка с воздухом (эффузивность 5.6 против 735 у обшивки)
# оказывается идеальным изолятором: тепло некуда уходить, и поверхность
# держит ~80 К от 5.5 с до 300 с без остывания. Тогда контраст любой
# заполненной ячейки к такой опоре равен почти единице и определяется не
# формой кривой, а самим фактом «воздух против вещества» — то есть ровно тот
# же тривиальный порог по амплитуде, что и в абсолютных кельвинах.
# С теплоотдачей опора остывает физично, и контрасты выходят на масштаб
# реальных данных (раздел 11.2).
SURFACE_LOSS_W_PER_M2_K = 10.0


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
    k: np.ndarray,
    rho_cp: np.ndarray,
    dx: float,
    dt: float,
    surface_loss_w_per_m2_k: float = SURFACE_LOSS_W_PER_M2_K,
) -> np.ndarray:
    """Собирает трёхдиагональную матрицу неявной схемы в banded-формате.

    Дискретизация консервативная:
        C_i * (T_i^{n+1} - T_i^n) / dt =
            [k_{i+1/2}(T_{i+1} - T_i) - k_{i-1/2}(T_i - T_{i-1})] / dx^2

    Поверхностный узел дополнительно теряет тепло в среду: член
    ``-h * T_0 / dx``, взятый НЕЯВНО (уходит на главную диагональ), поэтому
    безусловная устойчивость схемы сохраняется. См.
    `SURFACE_LOSS_W_PER_M2_K` — без этого члена ячейка с воздухом не остывает
    вовсе.

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

    # Теплоотдача с поверхности в среду (T_ambient = 0 в контрастной постановке).
    surface_loss = np.zeros(n)
    surface_loss[0] = dt * surface_loss_w_per_m2_k / (rho_cp[0] * dx)

    banded = np.zeros((3, n), dtype=np.float64)
    banded[0, 1:] = -beta[:-1]  # верхняя диагональ, сдвиг вправо
    banded[1, :] = 1.0 + alpha + beta + surface_loss  # главная диагональ
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
    surface_loss_w_per_m2_k: float = SURFACE_LOSS_W_PER_M2_K,
) -> np.ndarray:
    """Считает кривую перегрева поверхности неявной FD-схемой.

    Начальная температура нулевая всюду — работаем в контрасте к фону.
    Импульс мощности подаётся на поверхностный узел первые
    `pulse_duration_s` секунд, дальше поток нулевой. Дно столба
    адиабатическое, а поверхность отдаёт тепло в среду с коэффициентом
    `surface_loss_w_per_m2_k` (см. константу: без этого члена ячейка с
    воздухом вообще не остывает и синтетика остаётся тривиальной).

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
        surface_loss_w_per_m2_k: коэффициент теплоотдачи с поверхности,
            Вт/(м^2*К). Ноль возвращает полностью адиабатическую постановку.

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
    banded = _build_banded_matrix(k, rho_cp, dx, dt_s, surface_loss_w_per_m2_k)

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


def reference_curve_and_peak(
    power_w_per_m2: float,
    skin_thickness_m: float,
    t_eval_s: np.ndarray,
    dt_s: float = DATASET_DT_S,
    pulse_duration_s: float = PULSE_DURATION_S,
    cell_height_m: float = 0.010,
    n_nodes: int = 120,
) -> tuple[np.ndarray, float]:
    """Опорная кривая бездефектного участка и её пиковый подъём.

    Физически бездефектный участок панели — ячейка, заполненная воздухом
    (`fill_fraction=0`, вещество `air`). Именно она играет роль опорной зоны
    из `contrast.normalized_contrast` на реальных данных.

    Пиковый подъём ищется на интервале импульса, то есть ВНЕ логарифмической
    сетки наблюдения (та стартует уже после импульса). Поэтому опорная кривая
    считается на объединённой сетке: пульсовая часть нужна только ради
    знаменателя, наружу отдаются значения на `t_eval_s`.

    Args:
        power_w_per_m2: поглощённая плотность потока, Вт/м^2.
        skin_thickness_m: толщина обшивки, м.
        t_eval_s: моменты наблюдения (лог-сетка после импульса), с.
        dt_s: шаг интегрирования, с.
        pulse_duration_s: длительность импульса нагрева, с.
        cell_height_m: глубина ячейки H, м.
        n_nodes: число узлов сетки по глубине.

    Returns:
        ``(T_ref на t_eval_s (len(t_eval_s),) float64, пиковый подъём К)``.

    Raises:
        ValueError: если пиковый подъём опорной кривой ниже `MIN_PEAK_RISE_K`
            (нагрев не состоялся, нормировать не на что).
    """
    t_eval_s = np.asarray(t_eval_s, dtype=np.float64)
    # Сетка импульса нужна для знаменателя: пик подъёма приходится на момент
    # выключения лампы, а лог-сетка наблюдения начинается уже после него.
    pulse_grid = np.arange(0.0, pulse_duration_s + dt_s, dt_s)
    combined = np.concatenate([pulse_grid, t_eval_s])

    curve = simulate_cooling_curve(
        0.0,
        "air",
        power_w_per_m2=power_w_per_m2,
        pulse_duration_s=pulse_duration_s,
        t_eval_s=combined,
        dt_s=dt_s,
        cell_height_m=cell_height_m,
        n_nodes=n_nodes,
        skin_thickness_m=skin_thickness_m,
        surface_loss_w_per_m2_k=SURFACE_LOSS_W_PER_M2_K,
    )

    pulse_part = curve[: pulse_grid.size]
    # Начальная температура нулевая по построению солвера, поэтому подъём
    # относительно t=0 совпадает с самой кривой.
    peak_rise = float(pulse_part.max() - pulse_part[0])
    if peak_rise < MIN_PEAK_RISE_K:
        raise ValueError(
            f"Пиковый подъём опорной кривой {peak_rise:.3e} К < {MIN_PEAK_RISE_K:.1e} К: "
            "нагрев не состоялся"
        )

    return curve[pulse_grid.size :], peak_rise


def _quantize_jitter(value: float, levels: int = JITTER_LEVELS) -> float:
    """Округляет множитель джиттера до сетки из `levels` уровней.

    Дискретизация нужна ради кэша опорных симуляций: ключ
    ``(power, skin_thickness)`` из непрерывного джиттера никогда не повторяется,
    и на каждый сэмпл приходилось бы считать вторую симуляцию с нуля.
    Шаг сетки (например, 2%/16 уровней по мощности) на порядок мельче самого
    разброса, поэтому межобразцовая вариативность сохраняется.
    """
    return float(np.round(value * levels) / levels)


def generate_synthetic_dataset(
    n_samples: int = 20000,
    seed: int = 0,
    n_time_points: int = 64,
    dt_s: float = DATASET_DT_S,
    netd_k: float = NETD_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Генерирует синтетический датасет НОРМИРОВАННОГО КОНТРАСТА.

    Генератор отдаёт кривые в том же представлении, что
    `contrast.build_input_tensor` даёт на реальных данных::

        C(t) = (T_ref(t) - T(t)) / max_t(T_ref(t) - T_ref(0))

    где знаменатель — СКАЛЯР (пиковый подъём опорной кривой), а опорная
    кривая `T_ref` считается для бездефектной ячейки (воздух, fill=0) при ТОМ
    ЖЕ джиттере мощности и толщины обшивки. Совпадение джиттера принципиально:
    нормировка на опору с другими условиями съёмки не убирает зависимость от
    условий, а добавляет шум.

    Отдача абсолютных кельвинов (первая версия генератора) обесценивала
    предобучение: классы разделялись одним порогом по амплитуде, логистическая
    регрессия на сырых кривых давала accuracy 1.0000, а этапы A и B видели
    входы разного масштаба (ARCHITECTURE.md раздел 3).

    Класс «пусто» генерируется как реальная бездефектная ячейка с воздухом, у
    которой СВОЙ джиттер, отличный от джиттера опоры. Иначе он был бы
    вырожден: сам себе опора даёт C ≡ 0 по построению. Так класс получает
    малый, но ненулевой контраст — как бездефектные зоны реальной панели.

    Поверх контраста накладывается пиксельный шум уровня NETD камеры,
    отнормированный тем же скаляром: без него вырождение пары 80%/100% ниже
    порога чувствительности не воспроизводится.

    Межъячеечный разброс задают три независимых источника, которые нормировка
    на опору НЕ сокращает (в отличие от общего джиттера мощности, который
    сокращается точно — это и есть смысл нормировки):

    * `LOCAL_GAIN_SIGMA` — локальная освещённость и излучательная способность;
    * `LOCAL_FILL_SIGMA` — отклонение фактического пролива от номинала;
    * `REFERENCE_MISMATCH_SIGMA` — расхождение локального бездефектного фона с
      усреднённой по панели опорной зоной.

    Без них классы ложатся в непересекающиеся полосы по амплитуде и
    логистическая регрессия снова берёт ~1.0, несмотря на верную нормировку.

    Args:
        n_samples: число примеров.
        seed: зерно генератора.
        n_time_points: число отсчётов по времени в одной кривой.
        dt_s: шаг интегрирования, см. `DATASET_DT_S`.
        netd_k: СКО пиксельного шума в кельвинах до нормировки.

    Returns:
        (curves, substance_labels, fill_labels) — нормированный контраст
        (n_samples, n_time_points) float32, метки вещества int
        {0: empty, 1: water, 2: epoxy} и доля заполнения float32 (NaN там,
        где вещество не вода).
    """
    if netd_k < 0.0:
        raise ValueError(f"netd_k должен быть >= 0, получено {netd_k}")

    rng = np.random.default_rng(seed)
    times = log_time_grid(n_time_points)

    curves = np.empty((n_samples, n_time_points), dtype=np.float32)
    substance_labels = rng.integers(0, len(SUBSTANCE_CLASSES), size=n_samples).astype(np.int64)
    fill_labels = np.full(n_samples, np.nan, dtype=np.float32)

    # Кэш опорных симуляций по дискретизированному ключу условий съёмки.
    reference_cache: dict[tuple[float, float], tuple[np.ndarray, float]] = {}

    for index in range(n_samples):
        name = SUBSTANCE_CLASSES[substance_labels[index]]
        fill = _sample_fill(rng, name)

        power_factor = _quantize_jitter(
            float(rng.uniform(1.0 - POWER_JITTER, 1.0 + POWER_JITTER))
        )
        skin_factor = _quantize_jitter(
            float(rng.uniform(1.0 - SKIN_JITTER, 1.0 + SKIN_JITTER))
        )
        power = NOMINAL_POWER_W_PER_M2 * power_factor
        skin = SKIN_THICKNESS_M * skin_factor

        key = (power_factor, skin_factor)
        if key not in reference_cache:
            reference_cache[key] = reference_curve_and_peak(
                power, skin, times, dt_s=dt_s
            )
        ref_curve, peak_rise = reference_cache[key]

        # Локальная освещённость ЭТОЙ ячейки отличается от усреднённой по
        # опорной зоне: пятно лампы неравномерно, излучательная способность
        # разнится. Множитель входит в саму симуляцию (до нормировки), поэтому
        # нормировка на опору его не сокращает — в отличие от общего джиттера
        # мощности выше.
        local_gain = float(
            np.clip(rng.normal(1.0, LOCAL_GAIN_SIGMA), 0.5, 1.5)
        )

        if name == "empty":
            # Бездефектная ячейка отличается от опоры своими условиями съёмки:
            # тот же воздух, но собственный джиттер и собственная локальная
            # освещённость. Отсюда малый ненулевой контраст — как у
            # бездефектных зон реальной панели, где C около нуля с шумом.
            sample_curve = _empty_cell_curve(
                rng, times, dt_s, reference_cache, local_gain
            )
        else:
            # Фактический пролив отклоняется от номинала: реальная заливка
            # неравномерна по ячейкам зоны.
            actual_fill = float(
                np.clip(fill + rng.normal(0.0, LOCAL_FILL_SIGMA), 0.02, 1.0)
            )
            sample_curve = simulate_cooling_curve(
                actual_fill,
                name,
                power_w_per_m2=power * local_gain,
                pulse_duration_s=PULSE_DURATION_S,
                t_eval_s=times,
                dt_s=dt_s,
                skin_thickness_m=skin,
            )
            fill_labels[index] = np.float32(actual_fill) if name == "water" else np.nan

        # Опора известна с ошибкой: локальный бездефектный фон под этой
        # ячейкой отличается от усреднённой по панели опорной зоны.
        ref_mismatch = float(
            np.clip(rng.normal(1.0, REFERENCE_MISMATCH_SIGMA), 0.6, 1.4)
        )
        contrast = (ref_curve * ref_mismatch - sample_curve) / peak_rise
        # Калибровка sim-to-real: 1D-модель систематически завышает контраст,
        # потому что не воспроизводит боковую диффузию тепла, потери на
        # стенках сот и разброс излучательной способности (раздел 10).
        # Множитель измерен по всем 12 зонам water1+water4 и оказался
        # одинаковым для воды (0.300) и смолы (0.290) — расхождение
        # систематическое, а не случайное, поэтому один множитель законен.
        contrast = contrast * SIM_TO_REAL_SCALE
        if netd_k > 0.0:
            # Шум добавляется ПОСЛЕ калибровки: NETD — свойство камеры,
            # снимающей реальную панель, он не должен масштабироваться
            # вместе с сигналом.
            contrast = contrast + rng.normal(0.0, netd_k / peak_rise, size=times.size)
        curves[index] = contrast.astype(np.float32)

    return curves, substance_labels, fill_labels


def _sample_fill(rng: np.random.Generator, substance: str) -> float:
    """Выбирает долю заполнения по веществу.

    Вода заливалась на 20..100%, смола — один пролив почти на всю глубину
    (см. `WATER_FILL_RANGE` / `EPOXY_FILL_RANGE`). Пустая ячейка — воздух.
    """
    if substance == "empty":
        return 0.0
    low, high = WATER_FILL_RANGE if substance == "water" else EPOXY_FILL_RANGE
    return float(rng.uniform(low, high))


def _empty_cell_curve(
    rng: np.random.Generator,
    times: np.ndarray,
    dt_s: float,
    reference_cache: dict[tuple[float, float], tuple[np.ndarray, float]],
    local_gain: float,
) -> np.ndarray:
    """Кривая реальной бездефектной ячейки: воздух со своим джиттером.

    Переиспользует тот же кэш, что и опорные кривые: физически это та же
    симуляция (воздух, fill=0), отличаются только условия съёмки. Уравнение
    линейно по мощности, поэтому локальная освещённость учитывается
    домножением кэшированной кривой, без ещё одной симуляции.
    """
    power_factor = _quantize_jitter(
        float(rng.uniform(1.0 - POWER_JITTER, 1.0 + POWER_JITTER))
    )
    skin_factor = _quantize_jitter(
        float(rng.uniform(1.0 - SKIN_JITTER, 1.0 + SKIN_JITTER))
    )
    key = (power_factor, skin_factor)
    if key not in reference_cache:
        reference_cache[key] = reference_curve_and_peak(
            NOMINAL_POWER_W_PER_M2 * power_factor,
            SKIN_THICKNESS_M * skin_factor,
            times,
            dt_s=dt_s,
        )
    return reference_cache[key][0] * local_gain
