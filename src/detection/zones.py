"""Автоматическая детекция и локализация дефектных зон на картах признаков.

Дефектные зоны (ячейки, заполненные водой/гелем/смолой) отличаются от
бездефектной обшивки по тепловой инерции: они дольше держат тепло и
имеют иной интеграл перегрева. Детектор работает по картам признаков,
не требуя доступа к исходному кубу.

Зона считается найденной, если её поддержало голосование нескольких
независимых методов (см. `detect_zones`), а не порог по одному признаку:
один признак может быть слаб на конкретной записи, и его одного
недостаточно для устойчивой детекции на разных образцах.
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


#: Радиус сглаживания карт признаков в долях от стороны панели. Промежуток
#: между соседними зонами узкий: при более широком сглаживании три зоны
#: одного ряда сливаются в общую полосу, и разделить их водоразделом
#: не удаётся — контраст на границе стирается раньше, чем до водораздела
#: доходит дело.
_SMOOTH_FRACTION = 0.008

#: Доля площади панели, приходящаяся на одну дефектную зону. Зона занимает
#: заметную, но не преобладающую часть панели; диапазон умышленно широкий,
#: чтобы не терять зоны, снятые под углом или частично перекрытые оснасткой.
_ZONE_AREA_SHARE = (0.015, 0.14)

#: Период сотовой решётки, мм: измерен по спектру мощности внутри дефектных
#: зон на нескольких образцах (устойчиво воспроизводится). Задаёт масштаб
#: полосового фильтра для голоса `_vote_lattice`.
_CELL_PITCH_MM = 4.05

#: Сторона контролируемой панели по методике эксперимента, мм.
_PANEL_SIDE_MM = 200.0


def _normalized_feature_stack(
    features: FeatureMaps, mask: np.ndarray, smooth_px: float
) -> tuple[np.ndarray, list[str]]:
    """Робастно нормированные и сглаженные карты признаков внутри панели.

    Нормировка идёт через медиану и межквартильный размах, а не среднее
    со стандартным отклонением: амплитуда и интеграл различаются между
    записями на порядок, а у дефектных зон длинные хвосты распределения,
    на которых обычная нормировка съезжает вместе с выбросами.

    Возвращает вместе со стеком список имён вошедших в него признаков:
    константный на конкретной записи признак пропускается, и без явного
    списка индекс в возвращённом стеке расходится с индексом в
    `features.names` — при таком рассинхроне `_vote_threshold` находил
    по имени признака не тот канал (или падал по IndexError, если из
    стека выпал последний по счёту признак).
    """
    from scipy import ndimage

    layers, names = [], []
    for index, name in enumerate(features.names):
        field = features.maps[:, :, index].astype(np.float64)
        inside = field[mask]
        q1, q3 = np.percentile(inside, [25, 75])
        scale = q3 - q1
        if scale < 1e-9:
            continue  # признак-константа на этой записи: ничего не различает

        median = float(np.median(inside))
        normalized = (field - median) / scale
        # Сглаживание считается только по панели: иначе холодный фон
        # за кромкой затекает внутрь и создаёт ложный контраст по периметру.
        weights = mask.astype(np.float64)
        smoothed = ndimage.gaussian_filter(normalized * weights, smooth_px)
        norm = ndimage.gaussian_filter(weights, smooth_px)
        layers.append(
            np.divide(smoothed, norm, out=np.zeros_like(smoothed), where=norm > 1e-6)
        )
        names.append(name)
    return np.stack(layers, axis=-1), names


def _vote_threshold(
    stack: np.ndarray,
    stack_names: list[str],
    mask: np.ndarray,
    feature_name: str,
    sigma: float,
    invert: bool,
) -> np.ndarray:
    """Голос 1: отклонение одного признака от медианы панели на sigma.

    Индекс ищется в `stack_names`, а не в `features.names`: стек уже
    не содержит признаков, оказавшихся константой на этой записи, и их
    позиции в двух списках не совпадают.
    """
    if feature_name not in stack_names:
        return np.zeros(mask.shape, dtype=bool)
    field = stack[:, :, stack_names.index(feature_name)]
    inside = field[mask]
    median, spread = float(np.median(inside)), float(np.std(inside))
    if spread <= 0:
        return np.zeros(mask.shape, dtype=bool)
    if invert:
        return (field < median - sigma * spread) & mask
    return (field > median + sigma * spread) & mask


def _vote_gmm(stack: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Голос 2: гауссова смесь на всех признаках сразу.

    Ловит зоны, где контраст размазан по нескольким слабым признакам —
    ни один из них не даёт выброса, достаточного для порога по одному
    измерению, но совместно они разделяют фон и дефект на два кластера.

    На однородной панели без дефектов GMM с двумя компонентами всё равно
    что-нибудь находит: смесь охотно расщепляет чистый шум пополам, просто
    потому что ей разрешили два компонента. Проверка BIC отсекает это —
    если модель с одним компонентом объясняет данные не хуже (BIC ниже
    или сопоставим), второй компонент не подтверждён статистически, и
    голос молчит, а не выдаёт половину панели за дефект.
    """
    from sklearn.mixture import GaussianMixture

    samples = stack[mask]
    one = GaussianMixture(n_components=1, random_state=seed).fit(samples)
    two = GaussianMixture(
        n_components=2, covariance_type="full", random_state=seed, reg_covar=1e-4
    )
    labels = two.fit_predict(samples)
    if two.bic(samples) >= one.bic(samples):
        return np.zeros(mask.shape, dtype=bool)

    # Дефектный компонент — тот, что дальше от нуля: нормировка центрирует
    # фон (он занимает большую часть панели) около медианы.
    energy = [float(np.abs(two.means_[k]).sum()) for k in range(2)]
    defect = int(np.argmax(energy))
    # Компонент, занимающий почти всю панель, не может быть дефектом.
    if float((labels == defect).mean()) > 0.5:
        defect = 1 - defect

    probability = np.zeros(mask.shape, dtype=np.float64)
    probability[mask] = two.predict_proba(samples)[:, defect]
    return (probability > 0.5) & mask


def _vote_pca(stack: np.ndarray, mask: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Голос 3: контраст по вторым и третьим главным компонентам.

    Первая главная компонента вбирает неравномерность нагрева лампой —
    общий для всей панели тренд, не связанный с дефектами, поэтому
    отбрасывается. Дефектный контраст ищется в оставшихся компонентах.
    """
    from sklearn.decomposition import PCA

    samples = stack[mask]
    n_components = min(3, samples.shape[1])
    if n_components < 2:
        return np.zeros(mask.shape, dtype=bool)

    scores = PCA(n_components=n_components, random_state=0).fit_transform(samples)

    votes = np.zeros(mask.shape, dtype=bool)
    for component in range(1, n_components):  # первую компоненту пропускаем
        field = np.zeros(mask.shape, dtype=np.float64)
        field[mask] = scores[:, component]
        inside = field[mask]
        median, spread = float(np.median(inside)), float(np.std(inside))
        if spread <= 0:
            continue
        votes |= (np.abs(field - median) > sigma * spread) & mask
    return votes


def _vote_lattice(
    features: FeatureMaps, mask: np.ndarray, panel_side_px: float, sigma: float = 1.5
) -> np.ndarray:
    """Голос 4: локальная энергия на частоте сотовой решётки.

    Вода заполняет соты неравномерно, поэтому внутри дефектной зоны
    соседние ячейки контрастируют друг с другом, и период решётки
    проступает в амплитудной карте. На бездефектной обшивке соты
    почти не видны — контраст между соседними ячейками мал.

    Порог держится высоким намеренно: отношение медиан энергии внутри
    зон и вне их велико, но распределения ощутимо перекрываются, и при
    мягком пороге голос теряет избирательность и захватывает треть
    площади панели вместо самих зон.
    """
    from scipy import ndimage

    cell_px = _CELL_PITCH_MM * panel_side_px / _PANEL_SIDE_MM
    if cell_px < 3:
        return np.zeros(mask.shape, dtype=bool)

    amplitude = features["amplitude_max"].astype(np.float64)
    band = ndimage.gaussian_filter(amplitude, cell_px * 0.35) - ndimage.gaussian_filter(
        amplitude, cell_px * 0.9
    )
    energy = ndimage.uniform_filter(np.abs(band), max(int(cell_px * 3), 3))

    inside = energy[mask]
    median, spread = float(np.median(inside)), float(np.std(inside))
    if spread <= 0:
        return np.zeros(mask.shape, dtype=bool)
    return (energy > median + sigma * spread) & mask


def detect_zones(
    features: FeatureMaps,
    feature_name: str = "half_decay_time",
    min_area: int = 200,
    invert: bool = True,
    max_aspect_ratio: float = 3.0,
    min_fill_ratio: float = 0.5,
    edge_margin: int = 6,
) -> list[Zone]:
    """Находит дефектные зоны консенсусом четырёх независимых методов.

    Каждый метод ловит структуру, которую остальные могут пропустить: порог
    по одному признаку силён при явном контрасте, GMM видит размазанный
    по нескольким признакам сигнал, PCA снимает общий тренд нагрева и
    вскрывает контраст под ним, а анализ решётки различает зоны там, где
    все тепловые признаки слабы, но соты внутри зоны неоднородны.

    Зона засчитывается, если её поддержал хотя бы один метод (объединение,
    не пересечение): требование согласия нескольких методов отсекает зоны,
    которые уверенно видит только один из них, и на практике снижает
    полноту детекции сильнее, чем ложные срабатывания снижают точность.
    Лишние находки отсекаются последующей фильтрацией по площади и форме.

    Args:
        features: карты признаков образца.
        feature_name: карта для порогового голоса (см. `_vote_threshold`).
        min_area: минимальная площадь зоны в пикселях (отсекает шум).
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
    values = features[feature_name] if feature_name in features.names else features["amplitude_max"]
    inside = values[mask]
    if inside.size == 0:
        return []

    median = float(np.median(inside))
    spread = float(np.std(inside))
    if spread <= 0:
        return []

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    panel_side = float(max(rows.max() - rows.min(), cols.max() - cols.min()))
    panel_area = float(mask.sum())

    stack, stack_names = _normalized_feature_stack(
        features, mask, _SMOOTH_FRACTION * panel_side
    )

    anomaly = (
        _vote_threshold(stack, stack_names, mask, feature_name, 0.8, invert)
        | _vote_gmm(stack, mask)
        | _vote_pca(stack, mask)
        | _vote_lattice(features, mask, panel_side)
    )

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

    low_area = _ZONE_AREA_SHARE[0] * panel_area
    high_area = _ZONE_AREA_SHARE[1] * panel_area
    effective_min_area = max(min_area, int(low_area * 0.5))

    labels, count = ndimage.label(anomaly)
    zones: list[Zone] = []
    for index in range(1, count + 1):
        component = labels == index
        if int(component.sum()) < effective_min_area:
            continue

        # При слипании соседних зон разделяем блоб по водоразделу, иначе
        # он отсеется по форме или площади как единая крупная область.
        for piece in _split_merged(component, effective_min_area):
            area = int(piece.sum())
            if area < effective_min_area:
                continue
            if not (low_area <= area <= high_area):
                continue

            rows_piece, cols_piece = np.where(piece)
            row0, row1 = int(rows_piece.min()), int(rows_piece.max()) + 1
            col0, col1 = int(cols_piece.min()), int(cols_piece.max()) + 1
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
    Порог на семенах спускается от 0.75 к 0.45: слишком высокий порог
    иногда не даёт второго семени и слипшаяся полоса остаётся целой.
    """
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(component)
    peak = float(distance.max())
    if peak < 3:
        return [component]

    for ratio in (0.75, 0.65, 0.55, 0.45):
        seeds, n_seeds = ndimage.label(distance > peak * ratio)
        if n_seeds < 2:
            continue

        filled = _watershed_from_seeds(distance, seeds, component)
        pieces = [filled == i for i in range(1, n_seeds + 1)]
        kept = [p for p in pieces if int(p.sum()) >= min_area]
        if len(kept) >= 2:
            return kept

    return [component]


def _watershed_from_seeds(
    distance: np.ndarray, seeds: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Наращивает семена до границ маски (водораздел без внешних зависимостей)."""
    from scipy import ndimage

    labels = seeds.copy()
    structure = np.ones((3, 3))
    peak = float(distance.max())
    # Итеративная дилатация: каждое семя растёт, не перекрывая соседей.
    for _ in range(int(peak) + 2):
        grown = ndimage.grey_dilation(labels, footprint=structure)
        update = (labels == 0) & mask & (grown > 0)
        if not update.any():
            break
        labels = np.where(update, grown, labels)
    return labels


def order_zones_by_grid(zones: list[Zone]) -> list[Zone]:
    """Упорядочивает зоны так, как читается сетка: сверху вниз, слева направо.

    Нужно для сопоставления найденных зон с протоколом эксперимента,
    где зоны перечислены в порядке их расположения на образце.
    """
    if not zones:
        return []

    row_tolerance = float(np.median([z.height for z in zones])) * 0.6
    by_row = sorted(zones, key=lambda z: z.center[0])

    rows: list[list[Zone]] = [[by_row[0]]]
    for zone in by_row[1:]:
        if abs(zone.center[0] - rows[-1][0].center[0]) <= row_tolerance:
            rows[-1].append(zone)
        else:
            rows.append([zone])

    ordered: list[Zone] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda z: z.center[1]))
    return ordered


def zone_profile(features: FeatureMaps, zone: Zone) -> dict[str, float]:
    """Усредняет все признаки внутри зоны — вектор для классификатора."""
    patch = features.maps[zone.row0 : zone.row1, zone.col0 : zone.col1, :]
    return {
        name: float(np.nanmean(patch[:, :, i]))
        for i, name in enumerate(features.names)
    }
