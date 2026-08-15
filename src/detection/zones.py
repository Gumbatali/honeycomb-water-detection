"""Автоматическая детекция и локализация дефектных зон на картах признаков.

Дефектные зоны (ячейки, заполненные водой/гелем/смолой) отличаются от
бездефектной обшивки по тепловой инерции: они дольше держат тепло и
имеют иной интеграл перегрева. Детектор работает по картам признаков,
не требуя доступа к исходному кубу.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing.features import FeatureMaps


@dataclass(frozen=True)
class Zone:
    """Прямоугольная дефектная зона в пиксельных координатах."""

    row0: int
    col0: int
    row1: int
    col1: int
    score: float

    @property
    def height(self) -> int:
        return self.row1 - self.row0

    @property
    def width(self) -> int:
        return self.col1 - self.col0

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> tuple[float, float]:
        return ((self.row0 + self.row1) / 2, (self.col0 + self.col1) / 2)

    def as_dict(self) -> dict:
        return {
            "row0": self.row0,
            "col0": self.col0,
            "row1": self.row1,
            "col1": self.col1,
            "score": round(self.score, 4),
        }


def panel_mask(features: FeatureMaps, margin_percentile: float = 60.0) -> np.ndarray:
    """Выделяет область самой панели, отсекая фон и оснастку.

    Панель нагревается заметно сильнее окружающего фона, поэтому
    порог по амплитуде надёжно отделяет объект контроля от сцены.
    """
    from scipy import ndimage

    amplitude = features["amplitude_max"]
    threshold = np.percentile(amplitude, margin_percentile)
    mask = amplitude > threshold
    # Замыкание убирает шумовые проколы, иначе панель распадается на куски.
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    return _largest_component(mask)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Оставляет крупнейшую связную компоненту и заполняет внутренние дыры.

    Дефектные зоны и шум выбивают в маске отверстия. Без их заполнения
    последующая эрозия отступа разъедает панель изнутри, а не с кромки.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    keep = int(np.argmax(sizes)) + 1
    largest = labels == keep
    return ndimage.binary_fill_holes(largest)


def detect_zones(
    features: FeatureMaps,
    feature_name: str = "half_decay_time",
    min_area: int = 200,
    threshold_sigma: float = 1.0,
    invert: bool = True,
    max_aspect_ratio: float = 3.0,
    min_fill_ratio: float = 0.5,
    edge_margin: int = 6,
) -> list[Zone]:
    """Находит дефектные зоны как связные аномалии внутри панели.

    Args:
        features: карты признаков образца.
        feature_name: карта, по которой ищется контраст.
        min_area: минимальная площадь зоны в пикселях (отсекает шум).
        threshold_sigma: порог в стандартных отклонениях от медианы панели.
        invert: True — искать зоны с пониженным значением признака
            (характерно для half_decay_time и integral в этих данных).
        max_aspect_ratio: отсекает вытянутые артефакты (кромки, оснастка).
        min_fill_ratio: доля заполнения bounding box — зона сот компактна,
            а артефакты крепежа имеют рваную форму.
        edge_margin: отступ от границы панели, где детекции не учитываются.

    Returns:
        Список зон, отсортированный по убыванию контраста.
    """
    from scipy import ndimage

    mask = panel_mask(features)
    values = features[feature_name]
    inside = values[mask]
    if inside.size == 0:
        return []

    median = float(np.median(inside))
    spread = float(np.std(inside))
    if spread <= 0:
        return []

    if invert:
        anomaly = (values < median - threshold_sigma * spread) & mask
    else:
        anomaly = (values > median + threshold_sigma * spread) & mask

    # Замыкание ядром шире промежутка между зонами сваривает их в один блок,
    # после чего разделить зоны уже нечем. Ядро 3x3 закрывает шум, но
    # сохраняет перемычки шириной от 2 px.
    anomaly = ndimage.binary_opening(anomaly, np.ones((3, 3)))
    anomaly = ndimage.binary_closing(anomaly, np.ones((3, 3)))

    # Зоны у самой кромки панели — почти всегда оснастка/крепёж, а не дефект.
    # Эрозия идёт итерациями по 3x3: ширина отступа равна edge_margin пикселям
    # независимо от размера панели.
    if edge_margin > 0:
        # border_value=1: край кадра не считается кромкой панели, иначе
        # у образцов, обрезанных границей кадра, съедается вся площадь.
        interior = ndimage.binary_erosion(
            mask, np.ones((3, 3)), iterations=edge_margin, border_value=1
        )
        # Если панель мала и эрозия съела её целиком — отступ не применяем.
        if interior.any():
            anomaly &= interior

    labels, count = ndimage.label(anomaly)
    zones: list[Zone] = []
    for index in range(1, count + 1):
        component = labels == index
        if int(component.sum()) < min_area:
            continue

        # При низком пороге соседние зоны сливаются в один блоб —
        # разделяем его по водоразделу, иначе он отсеется по форме.
        for piece in _split_merged(component, min_area):
            area = int(piece.sum())
            if area < min_area:
                continue

            rows, cols = np.where(piece)
            row0, row1 = int(rows.min()), int(rows.max()) + 1
            col0, col1 = int(cols.min()), int(cols.max()) + 1
            height, width = row1 - row0, col1 - col0

            aspect = max(height, width) / max(min(height, width), 1)
            if aspect > max_aspect_ratio:
                continue
            if area / max(height * width, 1) < min_fill_ratio:
                continue

            contrast = abs(float(values[piece].mean()) - median) / spread
            zones.append(Zone(row0=row0, col0=col0, row1=row1, col1=col1, score=contrast))

    return sorted(zones, key=lambda z: z.score, reverse=True)


def _split_merged(component: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Разделяет слипшиеся зоны водоразделом по карте расстояний.

    Признак слипания — несколько устойчивых максимумов карты расстояний
    внутри компоненты. Форма для этого не годится: сетка 2x3 при слиянии
    даёт компактный блоб, неотличимый по aspect ratio от одиночной зоны.
    """
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(component)
    if distance.max() < 3:
        return [component]

    # Порог по максимуму расстояния оставляет только ядра зон;
    # перемычки между слипшимися зонами тоньше и отсекаются.
    seeds, n_seeds = ndimage.label(distance > distance.max() * 0.55)
    if n_seeds < 2:
        return [component]

    filled = _watershed_from_seeds(distance, seeds, component)
    pieces = [filled == i for i in range(1, n_seeds + 1)]
    kept = [p for p in pieces if int(p.sum()) >= min_area]
    return kept if len(kept) >= 2 else [component]


def _watershed_from_seeds(
    distance: np.ndarray, seeds: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Наращивает семена до границ маски (водораздел без внешних зависимостей)."""
    from scipy import ndimage

    labels = seeds.copy()
    structure = np.ones((3, 3))
    # Итеративная дилатация: каждое семя растёт, не перекрывая соседей.
    for _ in range(int(distance.max()) + 2):
        grown = ndimage.grey_dilation(labels, footprint=structure)
        update = (labels == 0) & mask & (grown > 0)
        if not update.any():
            break
        labels = np.where(update, grown, labels)
    return labels


def zone_profile(features: FeatureMaps, zone: Zone) -> dict[str, float]:
    """Усредняет все признаки внутри зоны — вектор для классификатора."""
    patch = features.maps[zone.row0 : zone.row1, zone.col0 : zone.col1, :]
    return {
        name: float(np.nanmean(patch[:, :, i]))
        for i, name in enumerate(features.names)
    }
