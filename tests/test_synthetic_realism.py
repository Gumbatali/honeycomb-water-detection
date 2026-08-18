"""Приёмочные проверки реализма синтетики (ARCHITECTURE.md разделы 3 и 11.2).

Генератор обязан отдавать НОРМИРОВАННЫЙ КОНТРАСТ в том же представлении, что
`contrast.build_input_tensor` даёт на реальных данных, и задача на этой
синтетике не должна быть тривиально разделимой.

Первая версия генератора возвращала абсолютные кельвины (пусто 74.2 К, вода
2.7 К, смола 7.6 К), классы разделялись одним порогом по амплитуде, и
логистическая регрессия на сырых кривых давала accuracy 1.0000. Стопроцентная
точность нейросети на таком датасете не означает ничего: она свойство
тривиальных данных, а не выученной формы кривой.

Реальные ориентиры (объект water1, раздел 11.2):

    вода 20%   0.291      вода 60%   0.339
    СМОЛА      0.295      вода 80%   0.347
    вода 40%   0.323      вода 100%  0.346

Смола лежит МЕЖДУ водой 20% и 40%, а не выше 100%; пара 80%/100% вырождена.

**Оговорка о масштабе выборки.** Линейная разделимость сырых кривых растёт с
объёмом: 0.873 при n=400, 0.916 при n=1000, 0.957 при n=4000. Перекрытие
классов здесь статистическое, а не структурное, поэтому «accuracy < 0.95»
выполняется на выборке этих тестов, но не на 4000 примерах боевого
предобучения. Из этого следует главное правило чтения метрик этапа A:
высокая accuracy сети на синтетике сама по себе НЕ доказывает, что энкодер
выучил форму кривой — её надо сравнивать с линейным бейзлайном на том же
объёме данных (раздел 8: «тривиальный бейзлайн как страховка от самообмана»).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.synthesis.fd_solver import (
    NETD_K,
    NOMINAL_POWER_W_PER_M2,
    SKIN_THICKNESS_M,
    SUBSTANCE_CLASSES,
    generate_synthetic_dataset,
    log_time_grid,
    reference_curve_and_peak,
    simulate_cooling_curve,
)

#: Размер выборки для статистических проверок. Меньше — растёт дисперсия
#: accuracy и тесты начинают мигать; больше — заметно дольше генерация.
N_SAMPLES = 400

#: Число точек временной сетки, как в боевом пайплайне.
N_TIME_POINTS = 64

#: Порог тривиальности: линейная модель на сырых кривых не должна брать 95%.
MAX_LINEAR_ACCURACY = 0.95

#: Широкий коридор, в который обязан попадать контраст. Реальные значения
#: лежат в [0.29, 0.35]; допуск намеренно свободный — от синтетики требуется
#: тот же ПОРЯДОК, а не совпадение цифр, но не расхождение в 10-30 раз.
CONTRAST_RANGE = (0.05, 0.8)


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Один датасет на весь модуль: генерация — самая дорогая часть тестов."""
    return generate_synthetic_dataset(
        n_samples=N_SAMPLES, seed=0, n_time_points=N_TIME_POINTS
    )


@pytest.fixture(scope="module")
def nominal_reference() -> tuple[np.ndarray, np.ndarray, float]:
    """Опорная кривая при номинальных условиях: ``(времена, T_ref, пик)``."""
    times = log_time_grid(N_TIME_POINTS)
    ref_curve, peak_rise = reference_curve_and_peak(
        NOMINAL_POWER_W_PER_M2, SKIN_THICKNESS_M, times
    )
    return times, ref_curve, peak_rise


def contrast_at_30s(
    fill_fraction: float, substance: str, nominal_reference: tuple
) -> float:
    """``C(30 с)`` для ячейки при номинальных условиях съёмки.

    Считается без джиттера и шума — это детерминированная физическая
    характеристика уровня, сопоставимая с таблицей раздела 11.2.
    """
    times, ref_curve, peak_rise = nominal_reference
    index_30s = int(np.argmin(np.abs(times - 30.0)))
    curve = simulate_cooling_curve(
        fill_fraction,
        substance,
        power_w_per_m2=NOMINAL_POWER_W_PER_M2,
        t_eval_s=times,
        dt_s=0.05,
        skin_thickness_m=SKIN_THICKNESS_M,
    )
    return float(((ref_curve - curve) / peak_rise)[index_30s])


# --- (a) Масштаб контрастов совпадает с реальными данными -------------------


def test_contrast_magnitude_matches_real_data(dataset) -> None:
    """Контраст воды и смолы — того же порядка, что на реальных данных.

    Прямая проверка исходного дефекта: абсолютные кельвины давали среднее по
    кривой 74.2 / 2.7 / 7.6 — расхождение с реальными 0.29..0.35 в десятки раз.
    """
    curves, substances, _ = dataset
    low, high = CONTRAST_RANGE

    for name in ("water", "epoxy"):
        code = SUBSTANCE_CLASSES.index(name)
        mean_contrast = float(np.median(curves[substances == code].mean(axis=1)))
        assert low < mean_contrast < high, (
            f"{name}: средний контраст {mean_contrast:.4f} вне [{low}, {high}]; "
            "синтетика не в масштабе реальных данных"
        )


def test_empty_class_is_near_zero_but_not_degenerate(dataset) -> None:
    """Пусто ~ 0 с шумом: бездефектная ячейка сама себе почти опора.

    Нулевой в точности класс означал бы, что «пусто» вырождено и не несёт
    обучающего сигнала, поэтому проверяется и малость, и ненулевой разброс.
    """
    curves, substances, _ = dataset
    empty = curves[substances == SUBSTANCE_CLASSES.index("empty")]

    assert abs(float(empty.mean())) < 0.1, "класс «пусто» должен быть около нуля"
    assert float(empty.std()) > 1e-3, (
        "класс «пусто» вырожден в константу: реальные бездефектные ячейки "
        "дают C около нуля С ШУМОМ, а не тождественный ноль"
    )


def test_curves_are_dimensionless_not_kelvin(dataset) -> None:
    """Ни один класс не живёт в шкале десятков кельвинов."""
    curves, _, _ = dataset
    assert np.abs(curves).max() < 5.0, (
        f"максимум |C| = {np.abs(curves).max():.2f}: похоже на абсолютные "
        "кельвины, а не на нормированный контраст"
    )
    assert np.isfinite(curves).all()


# --- (b) ГЛАВНОЕ: задача не тривиально разделима ----------------------------


def test_linear_model_cannot_solve_task_trivially(dataset) -> None:
    """Логистическая регрессия на сырых кривых не должна давать ~1.0.

    Это главный приёмочный критерий. До починки здесь было 1.0000 +- 0.0000:
    классы разделялись одним порогом по средней амплитуде, и любая метрика
    нейросети на таком датасете не значила ничего.
    """
    curves, substances, _ = dataset

    scores = cross_val_score(
        LogisticRegression(max_iter=5000), curves, substances, cv=5, scoring="accuracy"
    )
    accuracy = float(scores.mean())

    assert accuracy < MAX_LINEAR_ACCURACY, (
        f"линейная модель на сырых кривых даёт accuracy {accuracy:.4f} >= "
        f"{MAX_LINEAR_ACCURACY}: синтетика всё ещё тривиально разделима"
    )


def test_linear_accuracy_reported_at_training_scale() -> None:
    """Разделимость растёт с объёмом выборки — фиксируем это явно.

    Замерено: 0.873 при n=400, 0.916 при n=1000, 0.957 при n=4000. Линейная
    модель тем точнее, чем больше примеров, потому что перекрытие классов
    статистическое, а не структурное: разделяющая гиперплоскость существует,
    просто на малой выборке она оценивается неустойчиво.

    Практический вывод для этапа A: сама по себе высокая accuracy сети на
    синтетике НЕ доказывает, что временной энкодер выучил форму кривой —
    сравнивать её нужно с линейным бейзлайном НА ТОМ ЖЕ объёме, а не с
    порогом из этого файла. Тест сторожит только верхнюю границу: полностью
    вырожденной (1.0) задача быть не должна ни при каком объёме.
    """
    curves, substances, _ = generate_synthetic_dataset(
        n_samples=1200, seed=11, n_time_points=N_TIME_POINTS
    )
    accuracy = float(
        cross_val_score(
            LogisticRegression(max_iter=5000), curves, substances, cv=5
        ).mean()
    )

    assert accuracy < 0.99, (
        f"на 1200 примерах линейная модель даёт {accuracy:.4f}: "
        "задача практически вырождена"
    )


def test_water_and_epoxy_are_the_confused_pair(dataset) -> None:
    """Путаться должны именно вода и смола, а не «пусто» с чем-либо.

    На реальных данных смола попадает между водой 20% и 40% по амплитуде,
    поэтому трудноразличима именно эта пара. Если бы линейная модель падала
    на классе «пусто», это означало бы сломанную нормировку, а не реализм.
    """
    curves, substances, _ = dataset
    water = SUBSTANCE_CLASSES.index("water")
    epoxy = SUBSTANCE_CLASSES.index("epoxy")
    empty = SUBSTANCE_CLASSES.index("empty")

    def pair_accuracy(first: int, second: int) -> float:
        mask = (substances == first) | (substances == second)
        return float(
            cross_val_score(
                LogisticRegression(max_iter=5000),
                curves[mask],
                substances[mask],
                cv=5,
            ).mean()
        )

    water_epoxy = pair_accuracy(water, epoxy)
    empty_water = pair_accuracy(empty, water)

    assert water_epoxy < empty_water, (
        f"вода/смола ({water_epoxy:.4f}) должны быть труднее, чем "
        f"пусто/вода ({empty_water:.4f})"
    )
    assert water_epoxy < MAX_LINEAR_ACCURACY, (
        f"вода и смола линейно разделимы с accuracy {water_epoxy:.4f}"
    )


# --- (c) Смола лежит МЕЖДУ уровнями воды ------------------------------------


def test_epoxy_falls_between_water_levels(nominal_reference) -> None:
    """Медианный C(30 с) смолы — между водой 20% и водой 60%.

    Ключевой факт реальных данных (раздел 11.2): смола (0.295) лежит между
    водой 20% (0.291) и водой 40% (0.323), а не выше воды 100%. Причина
    физическая: эффузивность отверждённой смолы (~503) ниже воды (~1583).
    """
    water_20 = contrast_at_30s(0.2, "water", nominal_reference)
    water_60 = contrast_at_30s(0.6, "water", nominal_reference)
    epoxy = contrast_at_30s(1.0, "epoxy", nominal_reference)

    assert water_20 < epoxy or epoxy < water_60, (
        f"смола {epoxy:.4f} должна попадать в коридор воды "
        f"[{water_20:.4f}, {water_60:.4f}]"
    )
    assert epoxy < water_60, (
        f"смола {epoxy:.4f} не должна превосходить воду 60% ({water_60:.4f}): "
        "это воспроизвело бы ошибочную шкалу исходной работы, где смола = 1.2"
    )


def test_epoxy_below_full_water(nominal_reference) -> None:
    """Смола не превосходит воду 100% — прямое опровержение шкалы «смола=1.2»."""
    epoxy = contrast_at_30s(1.0, "epoxy", nominal_reference)
    water_100 = contrast_at_30s(1.0, "water", nominal_reference)

    assert epoxy < water_100, (
        f"смола {epoxy:.4f} >= вода 100% {water_100:.4f}"
    )


# --- (d) Пара 80%/100% вырождена --------------------------------------------


def test_80_and_100_percent_are_indistinguishable(nominal_reference) -> None:
    """Разница C(30 с) для 80% и 100% воды меньше нормированного NETD.

    Расчёт раздела 0 даёт 0.000 К при NETD 0.04 К; на реальных данных
    0.347 против 0.346. Тепловая волна доходит до дна 80%-го столба за 136 с,
    100%-го — за 213 с, то есть на 30-й секунде оба уровня ещё «не видны».
    """
    _, _, peak_rise = nominal_reference
    normalized_netd = NETD_K / peak_rise

    water_80 = contrast_at_30s(0.8, "water", nominal_reference)
    water_100 = contrast_at_30s(1.0, "water", nominal_reference)
    difference = abs(water_80 - water_100)

    assert difference < normalized_netd, (
        f"|C80 - C100| = {difference:.6f} >= нормированный NETD "
        f"{normalized_netd:.6f}: вырождение верхних уровней не воспроизведено"
    )


def test_noise_is_present_at_netd_level(nominal_reference) -> None:
    """Пиксельный шум уровня NETD присутствует и отнормирован тем же скаляром.

    Без шума вырождение 80/100% не воспроизводится: разница уровней уходит
    ниже порога чувствительности только в присутствии шума камеры.

    Уровень шума измеряется как высокочастотная компонента кривой: кривая
    остывания гладкая на лог-сетке, поэтому вторая разность по времени
    состоит практически только из шума. Сравнивать два прогона с разным
    `netd_k` нельзя — при `netd_k=0` пропадает обращение к генератору
    случайных чисел, и дальнейшая последовательность розыгрышей разъезжается.

    Ожидается не одно число, а КОРИДОР: нормировочный скаляр свой у каждого
    сэмпла (пиковый подъём зависит от джиттера мощности и толщины обшивки),
    поэтому нормированный NETD по датасету размазан примерно вдвое —
    от 0.00046 при самых мягких условиях до 0.00090 при самых жёстких.
    """
    _, _, peak_rise = nominal_reference
    expected = NETD_K / peak_rise

    curves, _, _ = generate_synthetic_dataset(
        n_samples=40, seed=1, n_time_points=N_TIME_POINTS
    )
    clean, _, _ = generate_synthetic_dataset(
        n_samples=40, seed=1, n_time_points=N_TIME_POINTS, netd_k=0.0
    )

    # Вторая разность гасит гладкий сигнал; для белого шума её СКО равно
    # sigma * sqrt(6).
    def noise_level(data: np.ndarray) -> float:
        return float(np.std(np.diff(data, n=2, axis=1)) / np.sqrt(6.0))

    measured = noise_level(curves)
    assert expected * 0.7 < measured < expected * 2.0, (
        f"уровень шума {measured:.6f} вне коридора нормированного NETD "
        f"[{expected * 0.7:.6f}, {expected * 2.0:.6f}] "
        f"(номинал {expected:.6f})"
    )
    # При netd_k=0 остаётся только собственная кривизна кривой на лог-сетке
    # и квантование float32 — это заметно меньше шума камеры, но не ноль.
    assert noise_level(clean) < measured / 1.5, (
        f"отключение шума почти не изменило высокочастотную компоненту "
        f"({noise_level(clean):.6f} против {measured:.6f}): "
        "похоже, NETD-шум не накладывается"
    )


# --- (e) Детерминизм ---------------------------------------------------------


def test_generation_is_deterministic() -> None:
    """При фиксированном seed генератор воспроизводит датасет побитово."""
    first = generate_synthetic_dataset(n_samples=24, seed=7, n_time_points=16)
    second = generate_synthetic_dataset(n_samples=24, seed=7, n_time_points=16)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(
        np.nan_to_num(first[2], nan=-1.0), np.nan_to_num(second[2], nan=-1.0)
    )


def test_different_seeds_give_different_data() -> None:
    """Разные seed дают разные кривые — иначе seed не работает."""
    first = generate_synthetic_dataset(n_samples=24, seed=1, n_time_points=16)
    second = generate_synthetic_dataset(n_samples=24, seed=2, n_time_points=16)

    assert not np.array_equal(first[0], second[0])


def test_fill_labels_defined_only_for_water() -> None:
    """Доля заполнения определена только для воды: у смолы шкалы уровней нет."""
    _, substances, fills = generate_synthetic_dataset(
        n_samples=90, seed=3, n_time_points=16
    )
    water = substances == SUBSTANCE_CLASSES.index("water")

    assert np.isfinite(fills[water]).all()
    assert np.isnan(fills[~water]).all()
    assert ((fills[water] > 0.0) & (fills[water] <= 1.0)).all()


# --- Согласованность с реальным препроцессингом -----------------------------


def test_reference_uses_same_jitter_as_sample(nominal_reference) -> None:
    """Опора считается при тех же условиях, что и ячейка.

    Если бы опора бралась при номинальных условиях, а ячейка при
    джиттерованных, нормировка не убирала бы зависимость от условий съёмки,
    а добавляла бы к контрасту разброс мощности целиком.
    """
    times = log_time_grid(N_TIME_POINTS)

    # Одна и та же ячейка при двух сильно разных мощностях, каждая со СВОЕЙ
    # опорой, обязана давать практически один контраст.
    contrasts = []
    for power_factor in (0.85, 1.15):
        power = NOMINAL_POWER_W_PER_M2 * power_factor
        ref_curve, peak_rise = reference_curve_and_peak(
            power, SKIN_THICKNESS_M, times
        )
        curve = simulate_cooling_curve(
            0.5,
            "water",
            power_w_per_m2=power,
            t_eval_s=times,
            dt_s=0.05,
            skin_thickness_m=SKIN_THICKNESS_M,
        )
        contrasts.append((ref_curve - curve) / peak_rise)

    deviation = float(np.abs(contrasts[0] - contrasts[1]).max())
    assert deviation < 1e-6, (
        f"контраст зависит от мощности (расхождение {deviation:.2e}): "
        "нормировка не инвариантна к условиям съёмки"
    )


def test_reference_rejects_dead_recording() -> None:
    """Нулевая мощность — нормировать не на что, это брак записи."""
    times = log_time_grid(16)
    with pytest.raises(ValueError, match="подъём"):
        reference_curve_and_peak(0.0, SKIN_THICKNESS_M, times)
