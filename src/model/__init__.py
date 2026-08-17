"""Модель детекции воды в сотовой панели (ARCHITECTURE.md разделы 5-6).

Пайплайн: TemporalEncoder -> SpatialBlock -> CellPooling -> две головы.
Ключевые проектные решения — ограниченное пространственное рецептивное поле
(1.31 ячейки), агрегация по ячейкам до loss и отказ от max-пуллинга по времени.
"""
from __future__ import annotations

from src.model.cell_pooling import CellPooling
from src.model.heads import (
    SUBSTANCE_CLASSES,
    WATER_GRADES,
    ClassificationHead,
    CoralOrdinalHead,
    TemporalStats,
)
from src.model.net import UNFREEZE_SCHEDULE, HoneycombNet
from src.model.spatial_block import SpatialBlock, spatial_receptive_field
from src.model.temporal_encoder import (
    TemporalEncoder,
    measure_receptive_field,
    receptive_field,
)

__all__ = [
    "SUBSTANCE_CLASSES",
    "UNFREEZE_SCHEDULE",
    "WATER_GRADES",
    "CellPooling",
    "ClassificationHead",
    "CoralOrdinalHead",
    "HoneycombNet",
    "SpatialBlock",
    "TemporalEncoder",
    "TemporalStats",
    "measure_receptive_field",
    "receptive_field",
    "spatial_receptive_field",
]
