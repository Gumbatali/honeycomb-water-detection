"""Классификация вещества по PCT-признакам зоны.

Метод, который на текущем объёме данных работает лучше нейросети:
83.3% на leave-one-object-out против 0% у `HoneycombNet` (обе версии
заморозки коллапсировали в константное предсказание — 12 обучающих зон
против даже 139 обучаемых параметров).

Почему PCT здесь выигрывает. Разложение по эмпирическим ортогональным
функциям нормировано по построению, поэтому не зависит от абсолютной
амплитуды сигнала — а именно она разводила синтетику и реальные записи
втрое (`fd_solver.SIM_TO_REAL_SCALE`). Классификатор поверх PCT видит
форму пространственно-временной картины, и она переносится между
панелями: измеренное разделение «вода минус смола» по третьей компоненте
равно 1.0 / 1.0 / 1.7 на трёх записях.

Ограничение: по ОТДЕЛЬНЫМ зонам классы перекрываются (вода доходит до
-0.98 при диапазоне смолы -1.43..-0.74), поэтому одного порога мало —
нужна комбинация компонент, что и делает логистическая регрессия.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.preprocessing.pct import sign_invariant_feature_maps
from src.train.real_dataset import RealZoneDataset

#: Сколько компонент PCT использовать. Первые четыре несут почти всю
#: полезную дисперсию: EOF1 отделяет дефект от фона, EOF3 — воду от смолы.
N_COMPONENTS: int = 4


@dataclass(frozen=True)
class ZoneFeatures:
    """Признаки одной зоны и её метка."""

    video: str
    substance: int
    features: np.ndarray


def zone_features(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Вектор признаков зоны: средние и разбросы PCT-компонент.

    Разброс нужен наравне со средним: у смолы картина внутри зоны
    однороднее, чем у воды, и это различает их дополнительно к уровню.

    Args:
        cube: (H, W, T) — контраст записи.
        mask: (H, W) bool — маска зоны.

    Returns:
        (2 * N_COMPONENTS,) float64.
    """
    maps = sign_invariant_feature_maps(cube, n_components=N_COMPONENTS)
    means = [float(maps[k][mask].mean()) for k in range(N_COMPONENTS)]
    stds = [float(maps[k][mask].std()) for k in range(N_COMPONENTS)]
    return np.array(means + stds, dtype=np.float64)


def collect_features(data_root: Path | str, videos: list[str]) -> list[ZoneFeatures]:
    """Собирает признаки всех зон перечисленных записей."""
    collected: list[ZoneFeatures] = []
    for video in videos:
        example = RealZoneDataset(data_root, [video]).examples[0]
        cube = np.moveaxis(example["x"][0].numpy(), 0, 2)
        maps = sign_invariant_feature_maps(cube, n_components=N_COMPONENTS)
        cell_index = example["cell_index"].numpy()
        substance = example["substance"].numpy()

        for cell_id in range(1, example["n_cells"] + 1):
            mask = cell_index == cell_id
            if not mask.any():
                continue
            means = [float(maps[k][mask].mean()) for k in range(N_COMPONENTS)]
            stds = [float(maps[k][mask].std()) for k in range(N_COMPONENTS)]
            collected.append(
                ZoneFeatures(
                    video=video,
                    substance=int(substance[cell_id - 1]),
                    features=np.array(means + stds, dtype=np.float64),
                )
            )
    return collected


def build_classifier() -> LogisticRegression:
    """Логистическая регрессия с балансировкой классов.

    `class_weight="balanced"` обязателен: воды в пять раз больше, чем
    смолы (5 зон против 1 на объект), и без балансировки классификатор
    выигрывает, всегда предсказывая воду.
    """
    return LogisticRegression(max_iter=5000, class_weight="balanced")


def leave_one_object_out(
    samples: list[ZoneFeatures],
) -> tuple[float, dict[str, tuple[int, int]]]:
    """Оценка по протоколу «тест на невиденной панели».

    Единственный честный протокол на этих данных: зоны одной панели
    снимались за одну запись с общим нагревом и общей подложкой, поэтому
    случайный сплит зон дал бы утечку (ARCHITECTURE.md раздел 2).

    Returns:
        ``(accuracy, {видео: (верно, всего)})``.
    """
    features = np.stack([s.features for s in samples])
    labels = np.array([s.substance for s in samples])
    videos = np.array([s.video for s in samples])

    per_video: dict[str, tuple[int, int]] = {}
    correct = 0
    for held_out in sorted(set(videos)):
        train, test = videos != held_out, videos == held_out
        model = build_classifier().fit(features[train], labels[train])
        predicted = model.predict(features[test])
        hits = int((predicted == labels[test]).sum())
        per_video[held_out] = (hits, int(test.sum()))
        correct += hits

    return correct / len(samples), per_video
