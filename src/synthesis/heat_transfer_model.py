"""Физическая модель нагрева и остывания ячейки сотовой панели.

Модель одномерная: тепловой импульс подаётся на обшивку, тепло уходит
в заполнитель ячейки (воздух/вода/гель/смола) и в окружающую среду.
Разные вещества отличаются объёмной теплоёмкостью и теплопроводностью,
что и даёт различимые кривые остывания.

Назначение — генерация синтетических температурных профилей для тех
степеней заполнения, которых нет в реальных экспериментах.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """Теплофизические свойства вещества.

    Attributes:
        name: имя класса вещества.
        density: плотность, кг/м^3.
        heat_capacity: удельная теплоёмкость, Дж/(кг*К).
        conductivity: теплопроводность, Вт/(м*К).
    """

    name: str
    density: float
    heat_capacity: float
    conductivity: float

    @property
    def volumetric_capacity(self) -> float:
        """Объёмная теплоёмкость, Дж/(м^3*К) — определяет тепловую инерцию."""
        return self.density * self.heat_capacity


# Справочные значения при комнатной температуре.
AIR = Material("air", density=1.2, heat_capacity=1005.0, conductivity=0.026)
WATER = Material("water", density=998.0, heat_capacity=4182.0, conductivity=0.6)
GEL = Material("gel", density=1050.0, heat_capacity=3800.0, conductivity=0.5)
EPOXY = Material("epoxy", density=1150.0, heat_capacity=1100.0, conductivity=0.2)

MATERIALS: dict[str, Material] = {m.name: m for m in (AIR, WATER, GEL, EPOXY)}

# Параметры обшивки (стеклопластик, 0.2 мм) из методики контроля.
SKIN_THICKNESS = 0.2e-3
SKIN_DENSITY = 1800.0
SKIN_CAPACITY = 1000.0
CELL_DEPTH = 10e-3


@dataclass(frozen=True)
class HeatingConfig:
    """Условия эксперимента.

    `power_density` откалибрована по измеренному образцу: галогенная лампа
    500 Вт даёт на обшивке перегрев порядка 10–20 К (медиана амплитуды по
    панели 1.1 К, 95-й перцентиль 20.7 К). Геометрическая оценка потока
    от лампы завышает перегрев на порядок, поэтому берётся значение,
    воспроизводящее наблюдаемый масштаб.
    """

    power_density: float = 1600.0  # Вт/м^2, поглощённая плотность потока
    heating_time: float = 5.0  # с
    duration: float = 300.0  # с
    fps: float = 10.0
    ambient_loss: float = 12.0  # Вт/(м^2*К), конвекция + излучение


def simulate_profile(
    fill_level: float,
    material: Material | str = WATER,
    config: HeatingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Считает температурный профиль обшивки над ячейкой.

    Модель сосредоточенных параметров: обшивка обменивается теплом с
    заполнителем (пропорционально степени заполнения) и со средой.

    Args:
        fill_level: доля заполнения ячейки веществом, 0..1.
        material: вещество-заполнитель.
        config: условия нагрева и регистрации.

    Returns:
        (time, temperature_rise) — время в секундах и перегрев над фоном в К.
    """
    if not 0.0 <= fill_level <= 1.0:
        raise ValueError(f"fill_level должен быть в [0, 1], получено {fill_level}")

    substance = MATERIALS[material] if isinstance(material, str) else material
    settings = config or HeatingConfig()

    time = np.arange(0.0, settings.duration, 1.0 / settings.fps)
    temperature = np.zeros_like(time)

    skin_capacity = SKIN_THICKNESS * SKIN_DENSITY * SKIN_CAPACITY

    # Заполнитель участвует в теплообмене той частью объёма, что занята
    # веществом; остальное — воздух.
    filled_depth = CELL_DEPTH * fill_level
    backing_capacity = (
        filled_depth * substance.volumetric_capacity
        + (CELL_DEPTH - filled_depth) * AIR.volumetric_capacity
    )

    # Проводимость к заполнителю растёт с долей контакта: пустая ячейка
    # отдаёт тепло только через воздух.
    contact = substance.conductivity * fill_level + AIR.conductivity * (1 - fill_level)
    coupling = contact / max(CELL_DEPTH, 1e-9)

    dt = 1.0 / settings.fps
    backing_temperature = 0.0

    for index, moment in enumerate(time):
        flux = settings.power_density if moment < settings.heating_time else 0.0
        to_backing = coupling * (temperature[index - 1] - backing_temperature) if index else 0.0
        to_ambient = settings.ambient_loss * (temperature[index - 1] if index else 0.0)

        previous = temperature[index - 1] if index else 0.0
        temperature[index] = previous + dt * (flux - to_backing - to_ambient) / skin_capacity
        backing_temperature += dt * to_backing / max(backing_capacity, 1e-9)

    return time, temperature


def synthesize_dataset(
    fill_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
    materials: tuple[str, ...] = ("water", "gel", "epoxy"),
    config: HeatingConfig | None = None,
    noise_sigma: float = 0.05,
    repeats: int = 5,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Генерирует набор синтетических профилей для обучения.

    Returns:
        Список записей с профилем, веществом и степенью заполнения.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    settings = config or HeatingConfig()
    dataset: list[dict] = []

    for material in materials:
        for fill in fill_levels:
            time, profile = simulate_profile(fill, material, settings)
            for _ in range(repeats):
                noisy = profile + generator.normal(0.0, noise_sigma, size=profile.shape)
                dataset.append(
                    {
                        "time": time,
                        "profile": noisy.astype(np.float32),
                        "substance": material if fill > 0 else "norm",
                        "fill_level": float(fill),
                    }
                )
    return dataset


def profile_to_features(
    time: np.ndarray, profile: np.ndarray, fps: float = 10.0
) -> np.ndarray:
    """Считает те же признаки, что и `extract_features`, но по одному профилю.

    Позволяет обучать классификатор на смеси реальных зон и синтетики.
    """
    from src.preprocessing.features import FEATURE_NAMES

    peak_index = int(np.argmax(profile))
    amplitude = float(profile[peak_index])
    dt = 1.0 / fps

    tail_start = max(len(profile) - max(1, len(profile) // 20), 0)
    tail = float(profile[tail_start:].mean())
    safe_amplitude = amplitude if abs(amplitude) > 1e-6 else 1e-6

    cooling_span = max((len(profile) - 1 - peak_index) * dt, dt)
    cooling_rate = (amplitude - tail) / cooling_span / safe_amplitude

    after_peak = profile[peak_index:]
    below_half = np.where(after_peak < amplitude * 0.5)[0]
    half_decay = float(below_half[0] * dt) if below_half.size else float(len(after_peak) * dt)

    values = {
        "amplitude_max": amplitude,
        "time_to_peak": peak_index * dt,
        "cooling_rate": cooling_rate,
        "tail_ratio": tail / safe_amplitude,
        "integral": float(profile.sum() * dt),
        "half_decay_time": half_decay,
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
