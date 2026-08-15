"""Классификация содержимого дефектных зон по тепловым признакам.

Решаются две связанные задачи:
  * тип вещества (вода / гель / эпоксидная смола / норма) — классификация;
  * степень заполнения ячейки (0..1) — регрессия.

Модель работает по вектору признаков зоны, а не по сырому кубу, поэтому
обучение и инференс не требуют доступа к гигабайтным данным.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.detection.zones import Zone, zone_profile
from src.preprocessing.features import FEATURE_NAMES, FeatureMaps

SUBSTANCE_CLASSES: tuple[str, ...] = ("norm", "water", "gel", "epoxy")


@dataclass(frozen=True)
class ZoneSample:
    """Обучающий пример: вектор признаков зоны и её разметка."""

    features: np.ndarray
    substance: str
    fill_level: float
    sample_name: str

    def __post_init__(self) -> None:
        if self.substance not in SUBSTANCE_CLASSES:
            raise ValueError(
                f"Неизвестный класс '{self.substance}', ожидается один из {SUBSTANCE_CLASSES}"
            )
        if not 0.0 <= self.fill_level <= 1.0:
            raise ValueError(f"fill_level должен быть в [0, 1], получено {self.fill_level}")


def zone_to_vector(features: FeatureMaps, zone: Zone) -> np.ndarray:
    """Преобразует зону в вектор признаков в фиксированном порядке."""
    profile = zone_profile(features, zone)
    return np.array([profile[name] for name in features.names], dtype=np.float32)


def build_matrix(samples: list[ZoneSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Собирает матрицу признаков и целевые векторы."""
    if not samples:
        raise ValueError("Пустой список примеров")
    features = np.stack([s.features for s in samples])
    substances = np.array([s.substance for s in samples])
    fills = np.array([s.fill_level for s in samples], dtype=np.float32)
    return features, substances, fills


@dataclass
class ThermalClassifier:
    """Классификатор вещества + регрессор степени заполнения.

    Использует градиентный бустинг: на выборках в сотни зон он устойчивее
    нейросети и не требует подбора архитектуры, оставаясь сравнимым по
    точности с baseline-MLP из методики ТПУ.
    """

    substance_model: object | None = None
    fill_model: object | None = None
    scaler: object | None = None
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def fit(self, samples: list[ZoneSample]) -> "ThermalClassifier":
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestClassifier,
        )
        from sklearn.preprocessing import StandardScaler

        features, substances, fills = build_matrix(samples)

        self.scaler = StandardScaler().fit(features)
        scaled = self.scaler.transform(features)

        if len(set(substances)) > 1:
            self.substance_model = RandomForestClassifier(
                n_estimators=200, random_state=0, class_weight="balanced"
            ).fit(scaled, substances)

        self.fill_model = GradientBoostingRegressor(random_state=0).fit(scaled, fills)
        return self

    def predict(self, features: np.ndarray) -> dict:
        """Предсказывает вещество и степень заполнения для вектора признаков."""
        if self.fill_model is None:
            raise RuntimeError("Модель не обучена — вызовите fit()")

        vector = np.atleast_2d(features)
        scaled = self.scaler.transform(vector) if self.scaler is not None else vector

        fill = float(np.clip(self.fill_model.predict(scaled)[0], 0.0, 1.0))
        result: dict = {"fill_level": fill}

        if self.substance_model is not None:
            probabilities = self.substance_model.predict_proba(scaled)[0]
            classes = list(self.substance_model.classes_)
            best = int(np.argmax(probabilities))
            result["substance"] = classes[best]
            result["confidence"] = float(probabilities[best])
            result["probabilities"] = {
                cls: float(p) for cls, p in zip(classes, probabilities)
            }
        return result

    def save(self, path: str | Path) -> Path:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(path: str | Path) -> "ThermalClassifier":
        import pickle

        with open(path, "rb") as handle:
            return pickle.load(handle)


def evaluate(
    classifier: ThermalClassifier, samples: list[ZoneSample]
) -> dict[str, float]:
    """Оценивает точность классификации и ошибку определения заполнения."""
    predictions = [classifier.predict(s.features) for s in samples]

    fill_errors = [
        abs(p["fill_level"] - s.fill_level) for p, s in zip(predictions, samples)
    ]
    metrics = {
        "fill_mae": float(np.mean(fill_errors)),
        "fill_mae_percent": float(np.mean(fill_errors) * 100),
        "n_samples": float(len(samples)),
    }

    if all("substance" in p for p in predictions):
        correct = sum(
            p["substance"] == s.substance for p, s in zip(predictions, samples)
        )
        metrics["substance_accuracy"] = correct / len(samples)
    return metrics
