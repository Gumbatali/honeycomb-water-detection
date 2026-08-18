"""Loss'ы обучения: CE по веществу + CORAL по градации заполнения.

Головы разделены по зоне ответственности (ARCHITECTURE.md раздел 5.4):
`ClassificationHead` решает «пусто / вода / эпоксидка», а порядковая голова
применяется ТОЛЬКО к ячейкам класса «вода» — градации «нет воды» в CORAL нет.
Отсюда главное требование к этому модулю: порядковый вклад считается по маске
`water_grade >= 0`, и батч, в котором воды не оказалось вовсе, должен давать
нулевой вклад, а не NaN от деления на пустое множество.

CORAL (Cao et al., 2020): для K градаций обучаются K-1 бинарных
классификаторов с общим вектором весов и раздельными смещениями. Таргет для
порога k — индикатор `1[grade > k]`. Монотонность порогов обеспечена
конструкцией `CoralOrdinalHead.thresholds` (`-cumsum(softplus(...))`), поэтому
штрафов за нарушение порядка в loss не нужно.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from src.train.dataset import IGNORE_GRADE, N_GRADES

#: Число порогов CORAL: для K градаций ровно K-1.
N_THRESHOLDS: int = N_GRADES - 1

#: Отсечка вероятностей на случай, если в `coral_loss` приходят вероятности,
#: а не логиты. Основной путь — логиты (см. `coral_loss`), где отсечка не нужна.
PROB_EPS: float = 1e-7


def grades_to_coral_targets(grades: Tensor) -> Tensor:
    """Разворачивает градации в бинарные таргеты порогов.

    Для градации g таргет порога k равен ``1[g > k]``. Градация 2 при двух
    порогах даёт [1, 1], градация 1 — [1, 0], градация 0 — [0, 0].

    Args:
        grades: (...,) int64 — индексы градаций 0..K-1.

    Returns:
        (..., N_THRESHOLDS) float32.
    """
    levels = torch.arange(N_THRESHOLDS, device=grades.device)
    return (grades.unsqueeze(-1) > levels).float()


def coral_loss(
    logits: Tensor, grades: Tensor, ignore_index: int = IGNORE_GRADE
) -> Tensor:
    """Порядковый CORAL-loss по маске «это вода».

    Принимает ЛОГИТЫ, а не вероятности, и использует
    `binary_cross_entropy_with_logits`. Это тот же путь, что в референсной
    реализации CORAL (Cao, Mirjalili, Raschka, 2020): там `CoralLayer`
    возвращает логиты, а `coral_loss` внутри применяет `logsigmoid`.

    Почему это существенно. Приём вероятностей требует отсечки перед
    логарифмом, а отсечка убивает градиент на насыщении сигмоиды: при логите
    -20 сигмоида равна 2e-9, отсечка на 1e-7 срезает значение, loss замирает
    на -log(1e-7), а градиент обращается в НОЛЬ — уверенно ошибающаяся ячейка
    перестаёт обучаться навсегда. Через логиты loss растёт линейно (|логит|),
    а градиент остаётся -1, и ячейка продолжает исправляться.

    Элементы с ``grades == ignore_index`` (вещество не вода — градация не
    определена) исключаются полностью. Если таких элементов оказался весь
    батч, возвращается нуль, сохраняющий связь с графом вычислений: так
    оптимизатор получает корректный нулевой градиент вместо NaN.

    Args:
        logits: (..., N_THRESHOLDS) — логиты порогов из `CoralOrdinalHead`
            (свойство `logits`, не `forward`).
        grades: (...,) int64 — градации либо `ignore_index`.
        ignore_index: метка «градация не определена».

    Returns:
        Скаляр — среднее BCE-with-logits по валидным элементам и порогам.
    """
    if logits.shape[:-1] != grades.shape:
        raise ValueError(
            f"Формы не согласованы: логиты {tuple(logits.shape)}, "
            f"градации {tuple(grades.shape)}"
        )
    if logits.shape[-1] != N_THRESHOLDS:
        raise ValueError(
            f"Ожидается {N_THRESHOLDS} порогов, получено {logits.shape[-1]}"
        )

    valid = grades != ignore_index
    if not bool(valid.any()):
        # Умножение на 0 сохраняет граф: `logits.sum() * 0` даёт нулевой
        # градиент по весам головы, тогда как свежий tensor(0.0) оторвал бы
        # голову от графа и сломал бы `backward` в некоторых конфигурациях.
        return logits.sum() * 0.0

    # Отрицательные метки нельзя подавать в сравнение с порогами: заменяем их
    # нулём и отбрасываем маской уже после.
    safe_grades = torch.where(valid, grades, torch.zeros_like(grades))
    targets = grades_to_coral_targets(safe_grades)

    per_threshold = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return per_threshold[valid].mean()


def combined_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    w_ordinal: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Полный loss этапа обучения: CE по веществу + взвешенный CORAL.

    Args:
        outputs: словарь из `HoneycombNet.forward` — ключи
            `substance_logits` (B, n_cells, 3) и `water_ordinal_logits`
            (B, n_cells, 2). Ключ `water_ordinal` (вероятности) для loss не
            используется: логиты численно устойчивее.
        targets: ключи `substance` (B, n_cells) int64 и
            `water_grade` (B, n_cells) int64 с `IGNORE_GRADE` вне класса «вода».
        w_ordinal: вес порядкового вклада.

    Returns:
        ``(total, components)`` — тензор для `backward` и словарь float'ов
        `{'total', 'substance', 'ordinal'}` для логирования.
    """
    logits = outputs["substance_logits"]
    ordinal_logits = outputs["water_ordinal_logits"]
    substance = targets["substance"]
    grades = targets["water_grade"]

    if logits.shape[:-1] != substance.shape:
        raise ValueError(
            f"Логиты {tuple(logits.shape)} не согласованы с метками "
            f"{tuple(substance.shape)}"
        )

    # CE ждёт (N, C): сворачиваем батч и ячейки в одну ось.
    substance_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), substance.reshape(-1)
    )
    ordinal_loss = coral_loss(ordinal_logits, grades)

    total = substance_loss + w_ordinal * ordinal_loss
    components = {
        "total": float(total.detach()),
        "substance": float(substance_loss.detach()),
        "ordinal": float(ordinal_loss.detach()),
    }
    return total, components
