"""Синтетический датасет этапа A: кривая остывания -> вход `HoneycombNet`.

Два несовпадения, которые закрывает этот модуль.

**Формы.** `generate_synthetic_dataset` даёт одномерные кривые (N, 64) —
профили отдельных «пикселей» без пространственной структуры, потому что
1D-модель теплопередачи её принципиально не воспроизводит (ARCHITECTURE.md
раздел 3). `HoneycombNet` ждёт куб (B, 3, T, H, W) плюс карту ячеек (B, H, W).
Адаптер разворачивает кривую в псевдо-патч `PATCH_SIZE x PATCH_SIZE`, все
пиксели которого принадлежат одной ячейке. Именно так на этапе A и должно
быть: `SpatialBlock` здесь обучается вырожденно и фактически лишь
инициализируется (раздел 6).

**Шум обязателен.** Без него все пиксели патча идентичны, `CellPooling`
усредняет одинаковые значения, и модель не видит ни одного примера того шума,
с которым столкнётся на реальной записи. Здесь добавляется именно
ПРОСТРАНСТВЕННЫЙ разброс между пикселями патча; временной шум уровня NETD
накладывает сам генератор (`fd_solver.generate_synthetic_dataset`).

**Кривые приходят уже нормированными.** Генератор отдаёт
``C(t) = (T_ref(t) - T(t)) / max_t(T_ref(t) - T_ref(0))`` — то же
представление, что `contrast.build_input_tensor` даёт на реальных данных.
Поэтому канал 0 берёт контраст как есть: повторное деление на «типичный
подъём» увело бы этап A в масштаб, которого на этапе B не существует, и
перенос весов не работал бы. Канал 1 — лог-производная (форма кривой без
амплитуды, главный признак, различающий воду и эпоксидку); канал 2 — форма,
растянутая в [0, 1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.contrast import log_derivative
from src.synthesis.fd_solver import generate_synthetic_dataset, log_time_grid

#: Сторона псевдо-патча. Минимум, на котором `SpatialBlock` с рецептивным
#: полем 15 px не вырождается в чистую точечную свёртку, но патч остаётся
#: дешёвым: 4x4 пикселя на пример.
PATCH_SIZE: int = 4

#: Единственная ячейка патча. 0 в `cell_index` зарезервирован под фон.
CELL_ID: int = 1
N_CELLS: int = 1

#: NETD камеры Optris PI 450, К (ARCHITECTURE.md раздел 0).
NETD_K: float = 0.04

#: Типичный пиковый подъём опорной зоны, К. Нужен только чтобы выразить NETD
#: в единицах контраста: сами кривые генератор уже отдаёт нормированными.
REFERENCE_PEAK_RISE_K: float = 75.0

#: Сигма ПРОСТРАНСТВЕННОГО шума между пикселями псевдо-патча, в единицах
#: контраста. Временной шум уровня NETD генератор накладывает сам; здесь
#: нужен разброс ПО ПИКСЕЛЯМ, иначе `CellPooling` усредняет идентичные
#: значения и модель не видит того шума, с которым столкнётся на записи.
NOISE_SIGMA_NORMALIZED: float = NETD_K / REFERENCE_PEAK_RISE_K

#: Границы градаций заполнения (ARCHITECTURE.md раздел 5.4): <30 / 30-70 / >70.
GRADE_BOUNDARIES: tuple[float, float] = (0.3, 0.7)
N_GRADES: int = 3

#: Метка «градация не определена» — для ячеек, где вещество не вода.
#: Совпадает с `ignore_index` по умолчанию в CORAL-loss.
IGNORE_GRADE: int = -1

#: Класс «вода» в кодировке `SUBSTANCE_CLASSES`.
WATER_CLASS: int = 1

#: Версия формата кэша кривых. Инкрементируется при смене ПРЕДСТАВЛЕНИЯ
#: синтетики. Версия 1 хранила абсолютные кельвины, версия 2 — нормированный
#: контраст. Без этого поля кэш от старого генератора подхватился бы молча:
#: формы и dtype массивов совпадают, а данные несопоставимы.
CACHE_FORMAT_VERSION: int = 2


def fill_to_grade(fill_fraction: float) -> int:
    """Переводит долю заполнения 0..1 в градацию 0/1/2.

    Непрерывная шкала процента неоправданна: пара 80%/100% даёт разницу
    сигналов 0.000 К при NETD 0.04 К (ARCHITECTURE.md раздел 0), поэтому
    физически различимы только три градации.

    Args:
        fill_fraction: доля заполнения ячейки водой, 0..1. NaN означает
            «вещество не вода».

    Returns:
        0 (<30%), 1 (30-70%), 2 (>70%) либо `IGNORE_GRADE` для NaN.
    """
    if not np.isfinite(fill_fraction):
        return IGNORE_GRADE
    low, high = GRADE_BOUNDARIES
    if fill_fraction < low:
        return 0
    if fill_fraction <= high:
        return 1
    return 2


def build_channels(curves: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Строит 3 канала входа из пачки кривых остывания.

    Каналы (те же, что в `build_input_tensor` реального пайплайна):
        0. ``C(t)`` — нормированный контраст КАК ЕСТЬ. Генератор уже отдаёт
           кривые нормированными на пиковый подъём опорной кривой, поэтому
           делить их здесь ещё раз нельзя: это увело бы этап A в масштаб,
           которого на реальных данных нет;
        1. ``d ln(dT) / d ln(t)`` — форма кривой без амплитуды;
        2. кривая, растянутая в [0, 1] по своему размаху — форма остывания,
           инвариантная и к мощности нагрева, и к сдвигу.

    Args:
        curves: (N, T) float — нормированный контраст `C(t)`, безразмерный.
        times: (T,) — моменты логарифмической сетки, с.

    Returns:
        (N, 3, T) float32.
    """
    if curves.ndim != 2:
        raise ValueError(f"Ожидается (N, T), получено {curves.shape}")
    if times.shape[0] != curves.shape[1]:
        raise ValueError(f"Ось времени {times.shape} не совпадает с {curves.shape}")

    amplitude = curves.astype(np.float64)
    # `log_derivative` работает с кубом (H, W, T): подставляем (N, 1, T).
    slope = log_derivative(curves[:, None, :].astype(np.float64), times)[:, 0, :]
    shape = _stretch_unit_range(curves.astype(np.float64))

    return np.stack([amplitude, slope, shape], axis=1).astype(np.float32)


def _stretch_unit_range(curves: np.ndarray) -> np.ndarray:
    """Растягивает каждую кривую в [0, 1] по её собственному размаху."""
    low = curves.min(axis=1, keepdims=True)
    span = curves.max(axis=1, keepdims=True) - low
    return (curves - low) / np.where(span < 1e-12, 1.0, span)


class SyntheticCurveDataset(Dataset):
    """Кривые физической синтетики как вход `HoneycombNet`.

    Каждый пример — псевдо-патч `PATCH_SIZE x PATCH_SIZE`, все пиксели
    которого несут одну и ту же кривую плюс независимый пиксельный шум
    уровня NETD. Все пиксели принадлежат ячейке `CELL_ID`, так что
    `CellPooling` возвращает ровно одну ячейку на пример.

    Датасет детерминирован: при одном `seed` и одинаковых параметрах
    `__getitem__` всегда даёт тот же тензор (шум генерируется один раз при
    инициализации, а не на каждом обращении).
    """

    def __init__(
        self,
        n_samples: int = 4000,
        seed: int = 0,
        n_time_points: int = 64,
        patch_size: int = PATCH_SIZE,
        noise_sigma: float = NOISE_SIGMA_NORMALIZED,
        cache_path: Path | str | None = None,
    ) -> None:
        """Готовит датасет, при необходимости поднимая кривые из кэша.

        Args:
            n_samples: число синтетических кривых.
            seed: зерно генерации кривых и пиксельного шума.
            n_time_points: длина временной оси.
            patch_size: сторона псевдо-патча.
            noise_sigma: СКО пиксельного шума в единицах контраста.
            cache_path: файл `.npz` для кэша кривых. Генерация одной кривой
                занимает ~0.15 с, поэтому 4000 примеров без кэша — это ~10
                минут на каждый запуск.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples должен быть >= 1, получено {n_samples}")
        if patch_size < 1:
            raise ValueError(f"patch_size должен быть >= 1, получено {patch_size}")
        if noise_sigma < 0.0:
            raise ValueError(f"noise_sigma должен быть >= 0, получено {noise_sigma}")

        self.patch_size = patch_size
        self.noise_sigma = noise_sigma
        self.times = log_time_grid(n_time_points)

        curves, substances, fills = load_or_generate(
            n_samples, seed, n_time_points, cache_path
        )
        self.substances = torch.from_numpy(substances.astype(np.int64))
        self.grades = torch.tensor(
            [fill_to_grade(float(value)) for value in fills], dtype=torch.long
        )
        self.channels = torch.from_numpy(build_channels(curves, self.times))
        self.noise = self._make_noise(seed, len(curves), n_time_points)

    def __len__(self) -> int:
        return self.channels.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает один псевдо-патч.

        Returns:
            ``(x (3, T, H, W) float32, cell_index (H, W) int64,
            substance () int64, water_grade () int64)`` — где `water_grade`
            равен `IGNORE_GRADE` для ячеек не из класса «вода».
        """
        size = self.patch_size
        # (3, T) -> (3, T, H, W): одна кривая на все пиксели патча.
        patch = self.channels[index][:, :, None, None].expand(-1, -1, size, size)
        x = patch + self.noise[index]

        cell_index = torch.full((size, size), CELL_ID, dtype=torch.long)
        return x.contiguous(), cell_index, self.substances[index], self.grades[index]

    def _make_noise(self, seed: int, n_samples: int, n_time_points: int) -> torch.Tensor:
        """Независимый пиксельный шум уровня NETD, (N, 3, T, H, W).

        Шум фиксируется один раз: датасет должен быть детерминированным, а
        роль шума здесь — заставить `CellPooling` усреднять неидентичные
        пиксели, а не имитировать аугментацию.
        """
        if self.noise_sigma == 0.0:
            return torch.zeros(1, 1, 1, 1, 1)

        generator = torch.Generator().manual_seed(seed + 1)
        shape = (n_samples, 3, n_time_points, self.patch_size, self.patch_size)
        return torch.randn(shape, generator=generator) * self.noise_sigma


def load_or_generate(
    n_samples: int,
    seed: int,
    n_time_points: int,
    cache_path: Path | str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Читает кривые из кэша или генерирует и сохраняет их.

    Кэш признаётся годным только при полном совпадении параметров генерации —
    иначе молча обучались бы на чужих данных.

    Returns:
        ``(curves (N, T), substance_labels (N,), fill_labels (N,))``.
    """
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        cached = _read_cache(path, n_samples, seed, n_time_points)
        if cached is not None:
            return cached

    curves, substances, fills = generate_synthetic_dataset(
        n_samples=n_samples, seed=seed, n_time_points=n_time_points
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            curves=curves,
            substance_labels=substances,
            fill_labels=fills,
            n_samples=n_samples,
            seed=seed,
            n_time_points=n_time_points,
            format_version=CACHE_FORMAT_VERSION,
        )
    return curves, substances, fills


def _read_cache(
    path: Path, n_samples: int, seed: int, n_time_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Возвращает содержимое кэша или None, если параметры не совпали.

    Кэш без поля `format_version` считается устаревшим (версия 1, абсолютные
    кельвины) и отбрасывается.
    """
    with np.load(path) as data:
        version = int(data["format_version"]) if "format_version" in data else 1
        matches = (
            version == CACHE_FORMAT_VERSION
            and int(data["n_samples"]) == n_samples
            and int(data["seed"]) == seed
            and int(data["n_time_points"]) == n_time_points
        )
        if not matches:
            return None
        return (
            data["curves"],
            data["substance_labels"],
            data["fill_labels"],
        )
