"""Метрики этапа A: точность по веществу и MAE по градациям заполнения.

Здесь метрики считаются по примерам синтетики — это допустимо только на
предобучении. На реальных данных единица независимости — зона, а не ячейка
(ARCHITECTURE.md разделы 0 и 8), и доверительные интервалы там считаются
бутстрепом по зонам.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.model.heads import SUBSTANCE_CLASSES
from src.train.dataset import IGNORE_GRADE, N_GRADES


@dataclass
class EpochMetrics:
    """Накопитель метрик за эпоху.

    Считает взвешенные суммы, чтобы неполный последний батч не искажал
    средние: усреднение средних по батчам разного размера смещено.
    """

    loss_sum: float = 0.0
    loss_weight: int = 0
    substance_correct: int = 0
    substance_total: int = 0
    grade_abs_error: float = 0.0
    grade_total: int = 0
    confusion: Tensor = field(
        default_factory=lambda: torch.zeros(
            len(SUBSTANCE_CLASSES), len(SUBSTANCE_CLASSES), dtype=torch.long
        )
    )
    grade_confusion: Tensor = field(
        default_factory=lambda: torch.zeros(N_GRADES, N_GRADES, dtype=torch.long)
    )

    def update(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        loss: float,
        batch_size: int,
    ) -> None:
        """Добавляет результаты одного батча."""
        self.loss_sum += loss * batch_size
        self.loss_weight += batch_size

        predicted = outputs["substance_logits"].argmax(dim=-1).reshape(-1)
        actual = targets["substance"].reshape(-1)
        self.substance_correct += int((predicted == actual).sum())
        self.substance_total += int(actual.numel())
        self._accumulate(self.confusion, actual, predicted)

        grades = targets["water_grade"].reshape(-1)
        valid = grades != IGNORE_GRADE
        if not bool(valid.any()):
            return
        # Градация = число порогов, преодолевших 0.5 (CORAL-декодирование).
        probabilities = outputs["water_ordinal"].reshape(-1, outputs["water_ordinal"].shape[-1])
        predicted_grade = (probabilities > 0.5).sum(dim=-1)[valid]
        actual_grade = grades[valid]
        self.grade_abs_error += float((predicted_grade - actual_grade).abs().sum())
        self.grade_total += int(actual_grade.numel())
        self._accumulate(self.grade_confusion, actual_grade, predicted_grade)

    @staticmethod
    def _accumulate(matrix: Tensor, actual: Tensor, predicted: Tensor) -> None:
        """Прибавляет пары (истина, предсказание) в матрицу ошибок."""
        size = matrix.shape[0]
        flat = actual.cpu() * size + predicted.cpu()
        matrix += torch.bincount(flat, minlength=size * size).reshape(size, size)

    @property
    def loss(self) -> float:
        """Средний loss на пример."""
        return self.loss_sum / max(self.loss_weight, 1)

    @property
    def substance_accuracy(self) -> float:
        """Доля верно определённых веществ."""
        return self.substance_correct / max(self.substance_total, 1)

    @property
    def grade_mae(self) -> float:
        """MAE по градациям в единицах градации (0/1/2)."""
        return self.grade_abs_error / max(self.grade_total, 1)

    def balanced_accuracy(self) -> float:
        """Среднее по классам recall — устойчиво к перекосу классов."""
        support = self.confusion.sum(dim=1)
        present = support > 0
        if not bool(present.any()):
            return 0.0
        recalls = self.confusion.diag()[present].double() / support[present].double()
        return float(recalls.mean())

    def format_line(self, prefix: str) -> str:
        """Однострочная сводка для лога эпохи."""
        return (
            f"{prefix} loss {self.loss:.4f} | acc {self.substance_accuracy:.3f} "
            f"| bal_acc {self.balanced_accuracy():.3f} | grade MAE {self.grade_mae:.3f}"
        )


def format_confusion(matrix: Tensor, labels: tuple[str, ...]) -> str:
    """Матрица ошибок в виде выровненной таблицы (строки — истина)."""
    width = max(len(name) for name in labels) + 2
    header = " " * width + "".join(f"{name:>9}" for name in labels)
    rows = [
        f"{labels[i]:<{width}}" + "".join(f"{int(value):>9}" for value in matrix[i])
        for i in range(len(labels))
    ]
    return "\n".join([header, *rows])
