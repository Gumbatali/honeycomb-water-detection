"""Этап A: предобучение `HoneycombNet` на физической синтетике.

На этом этапе обучается вся сеть (ARCHITECTURE.md раздел 6) — сеть учится
форме отклика, а не абсолютным температурам. `SpatialBlock` здесь обучается
вырожденно: 1D-модель даёт вход, константный по (H, W) с точностью до
пиксельного шума, — поэтому фактически он лишь инициализируется, и это ещё
один довод держать его замороженным на этапе B.

Случайный сплит train/val здесь допустим: примеры синтетики независимы по
построению (у каждого свои заполнение, мощность и толщина обшивки). На
реальных данных так делать нельзя — там сплит только по зонам (раздел 2).

Запуск::

    python -m src.train.pretrain --n-samples 4000 --epochs 15 --batch-size 64
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.model.heads import SUBSTANCE_CLASSES, WATER_GRADES
from src.model.net import HoneycombNet
from src.train.dataset import N_CELLS, SyntheticCurveDataset
from src.train.losses import combined_loss
from src.train.metrics import EpochMetrics, format_confusion

DEFAULT_CHECKPOINT = Path("data/checkpoints/pretrain.pt")
DEFAULT_CACHE_DIR = Path("data/synthetic")

# Гиперпараметры этапа A (ARCHITECTURE.md раздел 6).
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-2
DEFAULT_VAL_FRACTION: float = 0.2
DEFAULT_PATIENCE: int = 4


@dataclass
class TrainConfig:
    """Параметры прогона предобучения."""

    n_samples: int = 4000
    epochs: int = 15
    batch_size: int = 64
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    val_fraction: float = DEFAULT_VAL_FRACTION
    patience: int = DEFAULT_PATIENCE
    seed: int = 0
    device: str = "cpu"
    w_ordinal: float = 1.0
    out: Path = DEFAULT_CHECKPOINT
    cache: Path | None = None
    verbose: bool = True


def split_indices(
    n_samples: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Случайный сплит индексов на train/val.

    Допустим только для синтетики: примеры независимы по построению.

    Returns:
        ``(train_indices, val_indices)``, оба непустые.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction должен быть в (0, 1), получено {val_fraction}")
    if n_samples < 2:
        raise ValueError(f"Нужно минимум 2 примера, получено {n_samples}")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator).tolist()
    n_val = max(1, min(n_samples - 1, int(round(n_samples * val_fraction))))
    return permutation[n_val:], permutation[:n_val]


def run_epoch(
    model: HoneycombNet,
    loader: DataLoader,
    device: torch.device,
    w_ordinal: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Прогоняет одну эпоху. Без `optimizer` — режим валидации.

    Returns:
        Накопленные за эпоху метрики.
    """
    is_training = optimizer is not None
    model.train(is_training)
    metrics = EpochMetrics()

    with torch.set_grad_enabled(is_training):
        for x, cell_index, substance, grade in loader:
            x = x.to(device)
            cell_index = cell_index.to(device)
            # (B,) -> (B, n_cells): метка ячейки, а не примера.
            targets = {
                "substance": substance.to(device).unsqueeze(-1),
                "water_grade": grade.to(device).unsqueeze(-1),
            }

            outputs = model(x, cell_index, N_CELLS)
            loss, components = combined_loss(outputs, targets, w_ordinal=w_ordinal)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            metrics.update(outputs, targets, components["total"], x.shape[0])

    return metrics


def pretrain(config: TrainConfig) -> dict[str, object]:
    """Полный цикл предобучения с ранней остановкой и чекпоинтом.

    Returns:
        Словарь истории и итоговых метрик: `history`, `best_val_loss`,
        `best_epoch`, `train_metrics`, `val_metrics`, `elapsed_s`.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    started = time.time()

    dataset = SyntheticCurveDataset(
        n_samples=config.n_samples, seed=config.seed, cache_path=config.cache
    )
    train_indices, val_indices = split_indices(
        len(dataset), config.val_fraction, config.seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=config.batch_size, shuffle=False
    )

    model = HoneycombNet().to(device)
    model.unfreeze_all()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.verbose:
        report = model.parameter_report()
        print(
            f"Параметров: всего {report['total']}, энкодер {report['temporal_encoder']}, "
            f"spatial {report['spatial_block']}, головы {report['heads']}"
        )
        print(f"Примеров: train {len(train_indices)}, val {len(val_indices)}, устройство {device}")

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: tuple[EpochMetrics, EpochMetrics] | None = None

    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, device, config.w_ordinal, optimizer)
        val_metrics = run_epoch(model, val_loader, device, config.w_ordinal)
        history.append(_history_row(epoch, train_metrics, val_metrics))

        if config.verbose:
            print(
                f"эпоха {epoch + 1:>3}/{config.epochs} | "
                f"{train_metrics.format_line('train')} || {val_metrics.format_line('val')}"
            )

        if val_metrics.loss < best_val_loss - 1e-5:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = (train_metrics, val_metrics)
        elif epoch - best_epoch >= config.patience:
            if config.verbose:
                print(f"Ранняя остановка: val loss не улучшался {config.patience} эпох")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _save_checkpoint(model, config, best_val_loss, best_epoch, history)

    elapsed = time.time() - started
    if config.verbose and best_metrics is not None:
        _print_report(best_metrics[1], best_epoch, config.out, elapsed)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_metrics": best_metrics[0] if best_metrics else None,
        "val_metrics": best_metrics[1] if best_metrics else None,
        "elapsed_s": elapsed,
    }


def _history_row(
    epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics
) -> dict[str, float]:
    """Одна строка истории обучения."""
    return {
        "epoch": float(epoch),
        "train_loss": train_metrics.loss,
        "val_loss": val_metrics.loss,
        "train_acc": train_metrics.substance_accuracy,
        "val_acc": val_metrics.substance_accuracy,
        "train_grade_mae": train_metrics.grade_mae,
        "val_grade_mae": val_metrics.grade_mae,
    }


def _save_checkpoint(
    model: HoneycombNet,
    config: TrainConfig,
    best_val_loss: float,
    best_epoch: int,
    history: list[dict[str, float]],
) -> None:
    """Сохраняет лучшие веса вместе с конфигурацией прогона."""
    config.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stage": "A_pretrain_synthetic",
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "n_samples": config.n_samples,
            "seed": config.seed,
            "history": history,
        },
        config.out,
    )


def _print_report(
    val_metrics: EpochMetrics, best_epoch: int, out: Path, elapsed: float
) -> None:
    """Печатает итоговую сводку лучшей эпохи."""
    print(f"\nЛучшая эпоха {best_epoch + 1}, чекпоинт: {out}")
    print(f"Время обучения: {elapsed:.1f} с")
    print(f"\nВещество (val): {val_metrics.format_line('')}")
    print(format_confusion(val_metrics.confusion, SUBSTANCE_CLASSES))
    print(f"\nГрадации заполнения (val), MAE {val_metrics.grade_mae:.3f}:")
    print(format_confusion(val_metrics.grade_confusion, WATER_GRADES))


def build_parser() -> argparse.ArgumentParser:
    """CLI предобучения."""
    parser = argparse.ArgumentParser(
        description="Этап A: предобучение HoneycombNet на физической синтетике"
    )
    parser.add_argument("--n-samples", type=int, default=4000, help="число кривых синтетики")
    parser.add_argument("--epochs", type=int, default=15, help="максимум эпох")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu", help="cpu (по умолчанию), mps или cuda"
    )
    parser.add_argument("--w-ordinal", type=float, default=1.0, help="вес CORAL-вклада")
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="кэш кривых .npz; по умолчанию data/synthetic/curves_<N>_<seed>.npz",
    )
    parser.add_argument("--no-cache", action="store_true", help="не кэшировать синтетику")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Точка входа CLI."""
    args = build_parser().parse_args(argv)

    cache = args.cache
    if cache is None and not args.no_cache:
        cache = DEFAULT_CACHE_DIR / f"curves_{args.n_samples}_{args.seed}.npz"

    pretrain(
        TrainConfig(
            n_samples=args.n_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            w_ordinal=args.w_ordinal,
            out=args.out,
            cache=None if args.no_cache else cache,
        )
    )


if __name__ == "__main__":
    main()
