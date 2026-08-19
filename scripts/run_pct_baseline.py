"""Прогон PCT-baseline (`src/baseline/pct_classifier.py`) на реальных записях.

Единственный метод, который сейчас работает на 12 реальных зонах:
83.3% на leave-one-object-out против 0% у `HoneycombNet` (обе версии
заморозки коллапсировали в константу — см. ARCHITECTURE.md раздел 11.9).
Этот скрипт — воспроизводимая проверка того числа, не разовый расчёт
в блокноте.

Запуск::

    python -m scripts.run_pct_baseline --data-root ../hc-data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.pct_classifier import collect_features, leave_one_object_out  # noqa: E402
from src.model.heads import SUBSTANCE_CLASSES  # noqa: E402

#: Записи, участвующие в leave-one-object-out. `water120` размечен как
#: смола, а не 120% воды (ARCHITECTURE.md раздел 11.1) — здесь используется
#: как есть, поверх разметки `src/train/real_dataset.py`.
DEFAULT_VIDEOS: tuple[str, ...] = ("water1", "water2", "water4")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PCT-baseline на реальных данных")
    parser.add_argument("--data-root", type=Path, default=Path("../hc-data"))
    parser.add_argument(
        "--videos",
        nargs="+",
        default=list(DEFAULT_VIDEOS),
        help="Записи для leave-one-object-out (по умолчанию все три панели)",
    )
    args = parser.parse_args(argv)

    samples = collect_features(args.data_root, args.videos)
    if not samples:
        raise SystemExit(
            f"Не найдено ни одной зоны в {args.data_root} для {args.videos}"
        )

    accuracy, per_video = leave_one_object_out(samples)

    print(f"Зон всего: {len(samples)}")
    print(f"Классы: {SUBSTANCE_CLASSES}")
    print()
    print(f"{'held-out':<10} {'верно/всего':>14}")
    for video in sorted(per_video):
        hits, total = per_video[video]
        print(f"{video:<10} {hits:>6}/{total:<6}")

    print()
    print(f"Leave-one-object-out accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
