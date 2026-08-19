"""Пересчёт калибровочного множителя `SIM_TO_REAL_SCALE`.

Множитель выравнивает амплитуду синтетического контраста с реальным.
Запускать при смене условий съёмки: другая лампа, другое покрытие панели,
другая геометрия сот — всё это меняет соотношение.

Метод: средний контраст по зонам каждого класса на реальных записях
делится на средний контраст того же класса в синтетике. Классы считаются
раздельно, чтобы увидеть, систематическое расхождение или случайное: если
коэффициенты воды и смолы близки, единый множитель законен; если они
расходятся, дело не в общей амплитуде, а в физике конкретного вещества, и
масштабированием это не лечится.

Запуск::

    python -m scripts.calibrate_sim_to_real --data-root ../hc-data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.heads import SUBSTANCE_CLASSES  # noqa: E402
from src.synthesis.fd_solver import (  # noqa: E402
    SIM_TO_REAL_SCALE,
    generate_synthetic_dataset,
)
from src.train.real_dataset import RealZoneDataset  # noqa: E402

#: Записи, по которым калибруемся. Held-out объект сюда не входит: иначе
#: тестовая запись повлияла бы на обучающую синтетику.
CALIBRATION_VIDEOS: tuple[str, ...] = ("water1", "water4")

#: Расхождение коэффициентов между классами, выше которого единый
#: множитель считается необоснованным.
MAX_CLASS_SPREAD: float = 0.15


def synthetic_means(n_samples: int, seed: int) -> dict[int, float]:
    """Средний контраст синтетики по классам вещества.

    Синтетика генерируется БЕЗ калибровки — множитель делится обратно,
    иначе калибровка считалась бы от уже откалиброванных данных.
    """
    curves, substances, _ = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    raw = curves / SIM_TO_REAL_SCALE
    return {
        cls: float(raw[substances == cls].mean())
        for cls in range(len(SUBSTANCE_CLASSES))
        if np.any(substances == cls)
    }


def real_means(data_root: Path, videos: tuple[str, ...]) -> dict[int, float]:
    """Средний контраст реальных зон по классам вещества."""
    per_class: dict[int, list[float]] = {}
    for video in videos:
        example = RealZoneDataset(data_root, [video]).examples[0]
        cell_index = example["cell_index"]
        substance = example["substance"]
        for cell_id in range(1, example["n_cells"] + 1):
            mask = cell_index == cell_id
            if not bool(mask.any()):
                continue
            value = float(example["x"][0][:, mask].mean())
            per_class.setdefault(int(substance[cell_id - 1]), []).append(value)
    return {cls: float(np.mean(values)) for cls, values in per_class.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Калибровка sim-to-real множителя")
    parser.add_argument("--data-root", type=Path, default=Path("../hc-data"))
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)

    synthetic = synthetic_means(args.n_samples, args.seed)
    real = real_means(args.data_root, CALIBRATION_VIDEOS)

    print(f"{'класс':<10} {'синтетика':>12} {'реальность':>12} {'коэффициент':>13}")
    ratios: list[float] = []
    for cls, name in enumerate(SUBSTANCE_CLASSES):
        if cls not in synthetic or cls not in real:
            continue
        # Класс «пусто» исключён: его контраст около нуля по построению,
        # деление на него численно неустойчиво и физического смысла не несёт.
        if name == "empty":
            continue
        ratio = real[cls] / synthetic[cls]
        ratios.append(ratio)
        print(f"{name:<10} {synthetic[cls]:>12.4f} {real[cls]:>12.4f} {ratio:>13.3f}")

    if not ratios:
        raise SystemExit("Не удалось сопоставить ни одного класса")

    spread = max(ratios) - min(ratios)
    recommended = float(np.mean(ratios))
    print()
    print(f"Разброс между классами: {spread:.3f}")
    print(f"Рекомендуемый SIM_TO_REAL_SCALE: {recommended:.3f}")
    print(f"Текущий в коде:                  {SIM_TO_REAL_SCALE:.3f}")

    if spread > MAX_CLASS_SPREAD:
        print(
            f"\nВНИМАНИЕ: разброс {spread:.3f} > {MAX_CLASS_SPREAD}. Расхождение "
            "зависит от вещества, а не от общей амплитуды — единый множитель "
            "не оправдан, нужна ревизия теплофизических констант модели."
        )


if __name__ == "__main__":
    main()
