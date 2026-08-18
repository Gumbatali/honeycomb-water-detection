"""Ablation: вклад аугментации и синтетики в качество классификации.

Сравниваются четыре режима обучения на одной и той же тестовой выборке:
  1. только синтетика (baseline);
  2. синтетика + аугментация векторов;
  3. меньший объём синтетики (проверка чувствительности к размеру выборки);
  4. синтетика по расширенной сетке заполнения.

Тест всегда строится из профилей, не участвовавших в обучении.

Usage:
    python scripts/ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation.augment import augment_vector  # noqa: E402
from src.classification.classifier import (  # noqa: E402
    ThermalClassifier,
    ZoneSample,
    evaluate,
)
from src.synthesis.heat_transfer_model import (  # noqa: E402
    HeatingConfig,
    profile_to_features,
    synthesize_dataset,
)

CONFIG = HeatingConfig(duration=120.0)
SUBSTANCES = ("water", "gel", "epoxy")


def to_samples(dataset: list[dict]) -> list[ZoneSample]:
    return [
        ZoneSample(
            features=profile_to_features(row["time"], row["profile"], CONFIG.fps),
            substance=row["substance"],
            fill_level=row["fill_level"],
            sample_name="synthetic",
        )
        for row in dataset
    ]


def make_test_set(seed: int = 99) -> list[ZoneSample]:
    """Тестовая выборка на промежуточных уровнях заполнения."""
    dataset = synthesize_dataset(
        fill_levels=(0.15, 0.35, 0.55, 0.75, 0.95),
        materials=SUBSTANCES,
        config=CONFIG,
        repeats=4,
        noise_sigma=0.08,
        rng=np.random.default_rng(seed),
    )
    return to_samples(dataset)


def train_baseline(repeats: int, fills: tuple[float, ...], seed: int) -> list[ZoneSample]:
    dataset = synthesize_dataset(
        fill_levels=fills,
        materials=SUBSTANCES,
        config=CONFIG,
        repeats=repeats,
        noise_sigma=0.05,
        rng=np.random.default_rng(seed),
    )
    return to_samples(dataset)


def with_augmentation(samples: list[ZoneSample], factor: int, seed: int) -> list[ZoneSample]:
    rng = np.random.default_rng(seed)
    expanded = list(samples)
    for sample in samples:
        for vector in augment_vector(sample.features, n_variants=factor, rng=rng):
            expanded.append(
                ZoneSample(vector, sample.substance, sample.fill_level, sample.sample_name)
            )
    return expanded


def run(name: str, train: list[ZoneSample], test: list[ZoneSample]) -> dict:
    model = ThermalClassifier().fit(train)
    metrics = evaluate(model, test)
    print(
        f"{name:38s} n={len(train):5d}  "
        f"MAE={metrics['fill_mae_percent']:5.1f}%  "
        f"acc={metrics.get('substance_accuracy', float('nan')):.3f}"
    )
    return metrics


def main() -> None:
    test = make_test_set()
    print(f"Тестовая выборка: {len(test)} примеров (промежуточные заполнения)\n")

    coarse = (0.0, 0.25, 0.5, 0.75, 1.0)
    fine = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    small = train_baseline(repeats=2, fills=coarse, seed=1)
    base = train_baseline(repeats=6, fills=coarse, seed=1)

    results = {
        "1. синтетика, малый объём": run("1. синтетика, малый объём", small, test),
        "2. синтетика, базовый объём": run("2. синтетика, базовый объём", base, test),
        "3. синтетика + аугментация": run(
            "3. синтетика + аугментация", with_augmentation(base, 4, seed=2), test
        ),
        "4. синтетика по частой сетке": run(
            "4. синтетика по частой сетке",
            train_baseline(repeats=6, fills=fine, seed=1),
            test,
        ),
    }

    best = min(results.items(), key=lambda item: item[1]["fill_mae"])
    print(f"\nЛучший режим: {best[0]} (MAE {best[1]['fill_mae_percent']:.1f}%)")
    print("\nПримечание: оценка выполнена на синтетических данных, поэтому она")
    print("измеряет способность модели интерполировать физическую модель,")
    print("а не точность на реальном образце. Для последнего нужна разметка.")


if __name__ == "__main__":
    main()
