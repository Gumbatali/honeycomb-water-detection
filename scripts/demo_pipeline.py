"""Сквозная демонстрация конвейера на реальном образце.

Шаги: карты признаков -> детекция зон -> векторы признаков зон ->
аугментация -> обучение классификатора -> оценка на отложенных зонах.

Разметка зон здесь выводится из геометрии сетки 2x3 (порядковый номер
зоны слева направо, сверху вниз). Это временная разметка для проверки
работоспособности конвейера — реальные метки заполнения должны прийти
из протокола эксперимента.

Usage:
    python scripts/demo_pipeline.py [--sample NAME]
"""
from __future__ import annotations

import argparse
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
from src.detection.zones import detect_zones, order_zones_by_grid  # noqa: E402
from src.preprocessing.dataset import load_features  # noqa: E402

PROCESSED = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=None, help="Имя образца (без .npz)")
    parser.add_argument("--variants", type=int, default=40, help="Аугментаций на зону")
    args = parser.parse_args()

    candidates = sorted(PROCESSED.glob("*.npz"))
    if not candidates:
        print("Нет обработанных образцов в data/processed")
        return

    path = (
        PROCESSED / f"{args.sample}.npz"
        if args.sample
        else max(candidates, key=lambda p: p.stat().st_size)
    )
    features, meta = load_features(path)
    print(f"Образец: {meta.name}")
    print(f"  {meta.height}x{meta.width}, {meta.n_frames} кадров, {meta.fps} Гц")
    print(f"  ориентация: {meta.orientation}\n")

    zones = order_zones_by_grid(detect_zones(features))
    print(f"Найдено зон: {len(zones)}")
    if len(zones) < 2:
        print("Недостаточно зон для демонстрации обучения")
        return

    # Временная разметка: равномерная шкала заполнения по позиции в сетке.
    fill_levels = np.linspace(1.0, 0.2, len(zones))
    from src.classification.classifier import zone_to_vector

    train: list[ZoneSample] = []
    holdout: list[ZoneSample] = []
    rng = np.random.default_rng(0)

    for index, (zone, fill) in enumerate(zip(zones, fill_levels)):
        vector = zone_to_vector(features, zone)
        substance = "water"
        print(
            f"  зона {index+1}: bbox=({zone.row0},{zone.col0})-({zone.row1},{zone.col1}) "
            f"score={zone.score:.2f} -> метка заполнения {fill:.2f}"
        )

        variants = augment_vector(vector, n_variants=args.variants, rng=rng)
        # Часть аугментаций уходит в отложенную выборку: обучение и оценка
        # не должны делить одни и те же варианты.
        split = int(len(variants) * 0.75)
        for augmented in variants[:split]:
            train.append(ZoneSample(augmented, substance, float(fill), meta.name))
        for augmented in variants[split:]:
            holdout.append(ZoneSample(augmented, substance, float(fill), meta.name))

    print(f"\nОбучающих примеров: {len(train)}, отложенных: {len(holdout)}")

    model = ThermalClassifier().fit(train)
    metrics = evaluate(model, holdout)

    print("\nМетрики на отложенной выборке:")
    print(f"  MAE заполнения: {metrics['fill_mae']:.4f} ({metrics['fill_mae_percent']:.1f}%)")
    if "substance_accuracy" in metrics:
        print(f"  точность вещества: {metrics['substance_accuracy']:.3f}")

    reference = 0.15
    verdict = "в пределах" if metrics["fill_mae"] < reference else "хуже"
    print(f"\nОриентир методики ТПУ — 15% ошибки; результат {verdict} ориентира.")
    print("\nВНИМАНИЕ: метки заполнения здесь синтетические (по позиции в сетке).")
    print("Для настоящей оценки нужна разметка из протокола эксперимента.")


if __name__ == "__main__":
    main()
