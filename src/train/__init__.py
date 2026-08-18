"""Пайплайн обучения (ARCHITECTURE.md раздел 6).

Этап A — предобучение на физической синтетике, `pretrain.py`.
Этапы B (fine-tune) и C (оценка) опираются на те же loss'ы из `losses.py`,
но на реальных данных и с расписанием заморозки `HoneycombNet.freeze_for_finetune`.
"""
from __future__ import annotations

from src.train.dataset import (
    GRADE_BOUNDARIES,
    IGNORE_GRADE,
    SyntheticCurveDataset,
    fill_to_grade,
)
from src.train.losses import combined_loss, coral_loss, grades_to_coral_targets

__all__ = [
    "GRADE_BOUNDARIES",
    "IGNORE_GRADE",
    "SyntheticCurveDataset",
    "combined_loss",
    "coral_loss",
    "fill_to_grade",
    "grades_to_coral_targets",
]
