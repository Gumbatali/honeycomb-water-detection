"""Сборка модели: TemporalEncoder -> SpatialBlock -> CellPooling -> головы.

Модель одна и та же на всех этапах обучения (ARCHITECTURE.md раздел 6).
Между этапами меняется не архитектура, а то, какие веса открыты для градиента:

| Этап | Данные | Обучается |
|---|---|---|
| A. Предобучение | синтетика, 20k кривых | вся сеть |
| B. Fine-tune | water1/water4 — 12 зон, 864 ячейки | по расписанию заморозки |
| C. Оценка | water2 — held-out объект | ничего |

`SpatialBlock` заморожен на fine-tune всегда: он самый position-sensitive узел
сети и главный риск заучивания раскладки зон в кадре. Вдобавок на этапе A он
обучается вырожденно — 1D-синтетика даёт вход, константный по (H, W).
"""
from __future__ import annotations

from typing import Iterable

from torch import Tensor, nn

from src.model.cell_pooling import CellPooling
from src.model.heads import ClassificationHead, CoralOrdinalHead, TemporalStats
from src.model.spatial_block import SpatialBlock
from src.model.temporal_encoder import TemporalEncoder

FinetuneMode = str

#: Какие модули размораживаются в каждом режиме fine-tune.
UNFREEZE_SCHEDULE: dict[FinetuneMode, tuple[str, ...]] = {
    # только головы — при признаках переобучения на валидации по зонам
    "conservative": ("cls_head", "coral_head"),
    # последний блок энкодера + головы — режим по умолчанию
    "default": ("temporal_encoder.block4", "cls_head", "coral_head"),
    # два последних блока — только если leave-one-zone-out не показывает
    # разрыва train/val
    "full": ("temporal_encoder.block3", "temporal_encoder.block4", "cls_head", "coral_head"),
}


class HoneycombNet(nn.Module):
    """Полная модель детекции воды в сотовой панели."""

    def __init__(self, in_channels: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.temporal_encoder = TemporalEncoder(in_channels=in_channels, hidden=hidden)
        self.spatial_block = SpatialBlock(channels=hidden)
        self.cell_pooling = CellPooling()
        # Обе головы читают ОДНУ временную сводку: статистики окна — общее
        # представление, головы различаются лишь тем, как они его взвешивают.
        # Отдельные копии дублировали бы веса смешивания без пользы.
        self.temporal_stats = TemporalStats()
        self.cls_head = ClassificationHead(channels=hidden, stats=self.temporal_stats)
        self.coral_head = CoralOrdinalHead(channels=hidden, stats=self.temporal_stats)

    def forward(
        self, x: Tensor, cell_index: Tensor, n_cells: int
    ) -> dict[str, Tensor]:
        """Прямой проход от куба признаков до пер-ячеечных предсказаний.

        Args:
            x: (B, in_channels, T, H, W) — [C(t), dlnDT/dlnt, DT_normalized].
            cell_index: (B, H, W) int64, 0 = фон/стенка/игнор.
            n_cells: число ячеек в разметке.

        Returns:
            dict с ключами:
              'substance_logits'      (B, n_cells, 3) — логиты пусто/вода/эпоксидка;
              'water_ordinal_logits'  (B, n_cells, 2) — логиты порогов, вход для loss;
              'water_ordinal'         (B, n_cells, 2) — P(градация > k), для инференса.
        """
        features = self.temporal_encoder(x)
        features = self.spatial_block(features)
        cells = self.cell_pooling(features, cell_index, n_cells)
        return {
            "substance_logits": self.cls_head(cells),
            "water_ordinal_logits": self.coral_head.logits(cells),
            "water_ordinal": self.coral_head(cells),
        }

    def freeze_for_finetune(self, mode: FinetuneMode = "default") -> int:
        """Применяет расписание разморозки этапа B.

        `SpatialBlock` остаётся замороженным в любом режиме.

        Args:
            mode: 'conservative' (только головы), 'default' (последний блок
                энкодера + головы) или 'full' (два последних блока + головы).

        Returns:
            Число обучаемых параметров после применения расписания.
        """
        if mode not in UNFREEZE_SCHEDULE:
            raise ValueError(
                f"Неизвестный режим '{mode}', ожидается один из {tuple(UNFREEZE_SCHEDULE)}"
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name in UNFREEZE_SCHEDULE[mode]:
            for parameter in self._module_by_path(name).parameters():
                parameter.requires_grad_(True)

        assert not any(p.requires_grad for p in self.spatial_block.parameters()), (
            "SpatialBlock обязан оставаться замороженным на fine-tune"
        )
        return self.count_trainable_parameters()

    def unfreeze_all(self) -> int:
        """Открывает все веса — режим этапа A (предобучение на синтетике)."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self.count_trainable_parameters()

    def count_trainable_parameters(self) -> int:
        """Число параметров, по которым идёт градиент."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_report(self) -> dict[str, int]:
        """Параметры по компонентам — для отчёта рядом с числом независимых зон.

        `temporal_stats` общий для обеих голов, поэтому считается один раз:
        наивная сумма по головам завысила бы бюджет на размер общего модуля.
        """
        heads = _count(
            _unique(list(self.cls_head.parameters()) + list(self.coral_head.parameters()))
        )
        return {
            "temporal_encoder": _count(self.temporal_encoder.parameters()),
            "spatial_block": _count(self.spatial_block.parameters()),
            "heads": heads,
            "total": _count(self.parameters()),
        }

    def _module_by_path(self, path: str) -> nn.Module:
        """Возвращает подмодуль по точечному пути вида 'temporal_encoder.block4'."""
        module: nn.Module = self
        for part in path.split("."):
            if not hasattr(module, part):
                raise AttributeError(f"В модели нет подмодуля '{path}'")
            module = getattr(module, part)
        return module


def _count(parameters: Iterable[nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters)


def _unique(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    """Убирает повторы общих параметров, сохраняя порядок."""
    seen: dict[int, nn.Parameter] = {}
    for parameter in parameters:
        seen.setdefault(id(parameter), parameter)
    return list(seen.values())
