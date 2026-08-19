"""Этапы B и C: fine-tune на water1/water4, оценка на water2 (ARCHITECTURE.md раздел 6).

Датасет здесь — `RealZoneDataset`: один пример на ОБЪЕКТ (не на ячейку), все
зоны которого (6 штук) обрабатываются одним прямым проходом. Это отличает
цикл от `pretrain.run_epoch`, рассчитанного на батч фиксированного `N_CELLS`.

Валидация внутри train — leave-one-zone-out по 12 зонам двух обучающих
объектов, а не случайный сплит: ячейки одной зоны разделяют заливку,
подложку и режим нагрева, и попадание зоны в train и val одновременно
маскирует переобучение (ARCHITECTURE.md раздел 2).

Запуск::

    python -m src.train.finetune --checkpoint data/checkpoints/pretrain.pt \\
        --mode default --epochs 30 --out data/checkpoints/finetune.pt
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.model.heads import SUBSTANCE_CLASSES, WATER_GRADES
from src.model.net import HoneycombNet
from src.train.dataset import IGNORE_GRADE
from src.train.losses import combined_loss
from src.train.metrics import EpochMetrics, format_confusion
from src.train.real_dataset import RealZoneDataset

DEFAULT_DATA_ROOT = Path("../hc-data")
DEFAULT_PRETRAIN_CHECKPOINT = Path("data/checkpoints/pretrain.pt")
DEFAULT_FINETUNE_CHECKPOINT = Path("data/checkpoints/finetune.pt")

TRAIN_VIDEOS: tuple[str, ...] = ("water1", "water4")
HELD_OUT_VIDEO: str = "water2"

# Fine-tune — сильная регуляризация (ARCHITECTURE.md раздел 6): бюджет
# параметров даже в 'conservative' режиме велик относительно 12 зон train.
DEFAULT_LR: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 1e-2
DEFAULT_PATIENCE: int = 6


@dataclass
class FinetuneConfig:
    """Параметры прогона fine-tune."""

    data_root: Path = DEFAULT_DATA_ROOT
    pretrain_checkpoint: Path = DEFAULT_PRETRAIN_CHECKPOINT
    mode: str = "default"
    epochs: int = 30
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    patience: int = DEFAULT_PATIENCE
    w_ordinal: float = 1.0
    device: str = "cpu"
    out: Path = DEFAULT_FINETUNE_CHECKPOINT
    verbose: bool = True


def run_example(
    model: HoneycombNet,
    example: dict,
    device: torch.device,
    w_ordinal: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Один прямой (и, при заданном оптимизаторе, обратный) проход по объекту.

    Батч здесь — все зоны одного объекта разом: `x` без ведущей размерности
    батча в `RealZoneDataset`, добавляем её здесь.

    Returns:
        Метрики этого единственного примера (все зоны объекта).
    """
    is_training = optimizer is not None
    model.train(is_training)
    metrics = EpochMetrics()

    x = example["x"].unsqueeze(0).to(device)
    cell_index = example["cell_index"].unsqueeze(0).to(device)
    targets = {
        "substance": example["substance"].unsqueeze(0).to(device),
        "water_grade": example["water_grade"].unsqueeze(0).to(device),
    }

    with torch.set_grad_enabled(is_training):
        outputs = model(x, cell_index, example["n_cells"])
        loss, components = combined_loss(outputs, targets, w_ordinal=w_ordinal)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    metrics.update(outputs, targets, components["total"], batch_size=example["n_cells"])
    return metrics


def leave_one_zone_out_loss(
    model: HoneycombNet,
    examples: list[dict],
    device: torch.device,
    w_ordinal: float,
) -> float:
    """Средний val loss при поочерёдном исключении каждого объекта из train.

    Полноценный leave-one-ZONE-out (не объекта) потребовал бы дробить
    прямой проход по отдельным зонам, а `CellPooling`/`SpatialBlock`
    рассчитаны на кадр целиком. При всего двух train-объектах
    leave-one-object-out — практичное приближение той же идеи: модель не
    должна проверяться на данных из того же объекта, на котором училась.
    Не используется как метрика приёмки — только как индикатор для выбора
    режима заморозки (раздел 6).
    """
    losses = []
    for held_index in range(len(examples)):
        train_subset = [e for i, e in enumerate(examples) if i != held_index]
        val_example = examples[held_index]

        probe = HoneycombNet().to(device)
        probe.load_state_dict(model.state_dict())
        probe.freeze_for_finetune(mode="conservative")  # быстрая проба, не полный прогон
        optimizer = torch.optim.Adam(
            (p for p in probe.parameters() if p.requires_grad), lr=DEFAULT_LR
        )
        for _ in range(5):
            for example in train_subset:
                run_example(probe, example, device, w_ordinal, optimizer)

        val_metrics = run_example(probe, val_example, device, w_ordinal)
        losses.append(val_metrics.loss)
    return sum(losses) / len(losses)


def finetune(config: FinetuneConfig) -> dict[str, object]:
    """Полный цикл этапа B (fine-tune) + этапа C (оценка на held-out).

    Returns:
        Словарь с историей, метриками на train и на held-out объекте.
    """
    device = torch.device(config.device)
    started = time.time()

    if config.verbose:
        print(f"Загрузка train {TRAIN_VIDEOS} и held-out {HELD_OUT_VIDEO}...")
    train_examples = RealZoneDataset(config.data_root, list(TRAIN_VIDEOS)).examples
    held_out_example = RealZoneDataset(config.data_root, [HELD_OUT_VIDEO]).examples[0]

    model = HoneycombNet().to(device)
    checkpoint = torch.load(config.pretrain_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    n_trainable = model.freeze_for_finetune(mode=config.mode)

    if config.verbose:
        n_zones = sum(e["n_cells"] for e in train_examples)
        print(
            f"Fine-tune режим '{config.mode}': {n_trainable} обучаемых параметров, "
            f"{n_zones} зон train на {n_trainable / n_zones:.0f} параметров/зону"
        )

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_train_loss = float("inf")
    best_epoch = -1

    for epoch in range(config.epochs):
        epoch_metrics = EpochMetrics()
        for example in train_examples:
            m = run_example(model, example, device, config.w_ordinal, optimizer)
            epoch_metrics.loss_sum += m.loss_sum
            epoch_metrics.loss_weight += m.loss_weight
            epoch_metrics.substance_correct += m.substance_correct
            epoch_metrics.substance_total += m.substance_total
            epoch_metrics.grade_abs_error += m.grade_abs_error
            epoch_metrics.grade_total += m.grade_total
            epoch_metrics.confusion += m.confusion
            epoch_metrics.grade_confusion += m.grade_confusion

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_metrics.loss,
                "train_acc": epoch_metrics.substance_accuracy,
                "train_grade_mae": epoch_metrics.grade_mae,
            }
        )
        if config.verbose:
            print(f"эпоха {epoch + 1:>3}/{config.epochs} | {epoch_metrics.format_line('train')}")

        if epoch_metrics.loss < best_train_loss - 1e-5:
            best_train_loss = epoch_metrics.loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            if config.verbose:
                print(f"Ранняя остановка: train loss не улучшался {config.patience} эпох")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    held_out_metrics = run_example(model, held_out_example, device, config.w_ordinal)
    _save_checkpoint(model, config, best_epoch, history, held_out_metrics)

    elapsed = time.time() - started
    if config.verbose:
        _print_report(held_out_metrics, best_epoch, config.out, elapsed)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "held_out_metrics": held_out_metrics,
        "elapsed_s": elapsed,
    }


def _save_checkpoint(
    model: HoneycombNet,
    config: FinetuneConfig,
    best_epoch: int,
    history: list[dict[str, float]],
    held_out_metrics: EpochMetrics,
) -> None:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stage": "B_finetune_real",
            "mode": config.mode,
            "best_epoch": best_epoch,
            "history": history,
            "held_out_video": HELD_OUT_VIDEO,
            "held_out_loss": held_out_metrics.loss,
            "held_out_substance_accuracy": held_out_metrics.substance_accuracy,
            "held_out_grade_mae": held_out_metrics.grade_mae,
        },
        config.out,
    )


def _print_report(
    held_out_metrics: EpochMetrics, best_epoch: int, out: Path, elapsed: float
) -> None:
    print(f"\nЛучшая эпоха {best_epoch + 1}, чекпоинт: {out}")
    print(f"Время обучения: {elapsed:.1f} с")
    print(f"\nВещество ({HELD_OUT_VIDEO}, held-out): {held_out_metrics.format_line('')}")
    print(format_confusion(held_out_metrics.confusion, SUBSTANCE_CLASSES))
    print(f"\nГрадации заполнения ({HELD_OUT_VIDEO}), MAE {held_out_metrics.grade_mae:.3f}:")
    print(format_confusion(held_out_metrics.grade_confusion, WATER_GRADES))
    print(
        f"\nПРЕДУПРЕЖДЕНИЕ: held-out объект даёт всего 6 зон — единственный "
        f"тестовый пример на статистику, доверительный интервал не имеет "
        f"смысла (ARCHITECTURE.md раздел 8). Числа читать как один прецедент."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Этапы B/C: fine-tune HoneycombNet на реальных данных + оценка на held-out"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument(
        "--mode", type=str, default="default", choices=["conservative", "default", "full"]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--w-ordinal", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_FINETUNE_CHECKPOINT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    finetune(
        FinetuneConfig(
            data_root=args.data_root,
            pretrain_checkpoint=args.checkpoint,
            mode=args.mode,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            w_ordinal=args.w_ordinal,
            device=args.device,
            out=args.out,
        )
    )


if __name__ == "__main__":
    main()
