"""Две головы поверх пер-ячеечных признаков (B, n_cells, C, T).

**Пуллинг по времени — не максимум.** За 5 с импульса прогрев проникает лишь
на ~1.5 мм, меньше даже 20%-го столба воды (2 мм), поэтому вблизи пика все
уровни заполнения физически неразличимы (ARCHITECTURE.md раздел 0). `max` по
времени выбирал бы именно ту точку, где информации нет. Вместо этого —
конкатенация статистик информативного окна: среднее, значения на нескольких
фиксированных отсчётах и наклон лог-производной.

Головы намеренно крошечные (десятки параметров): на 12 независимых зон в train
любая заметная ёмкость здесь переобучается (раздел 6).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SUBSTANCE_CLASSES: tuple[str, ...] = ("empty", "water", "epoxy")
WATER_GRADES: tuple[str, ...] = ("<30", "30-70", ">70")
N_THRESHOLDS: int = len(WATER_GRADES) - 1  # CORAL: для K градаций нужно K-1 порогов
N_PROBE_POINTS: int = 3


class TemporalStats(nn.Module):
    """Сводит (B, n_cells, C, T) к (B, n_cells, C) без max по времени.

    Статистики считаются ПОКАНАЛЬНО и смешиваются обучаемыми весами в один
    скаляр на канал. Усреднять каналы до голов нельзя: энкодер учит в разных
    каналах разные участки формы кривой, и усреднение стёрло бы ровно то, ради
    чего он обучался.

    Набор статистик информативного окна `t ∈ [5.5; 100] с`: среднее, значения
    на `N_PROBE_POINTS` фиксированных отсчётах и наклон лог-производной.
    Максимума по времени здесь нет намеренно — см. докстринг модуля.
    """

    def __init__(self, n_probe_points: int = N_PROBE_POINTS) -> None:
        super().__init__()
        self.n_probe_points = n_probe_points
        # Смешивание статистик в один скаляр на канал: веса общие для всех
        # каналов, поэтому вклад в бюджет параметров минимален.
        self.mix = nn.Linear(self.n_raw_stats, 1, bias=False)

    @property
    def n_raw_stats(self) -> int:
        """Число сырых статистик на канал."""
        return self.n_probe_points + 2  # проб-точки + среднее + наклон

    def forward(self, x: Tensor) -> Tensor:
        """(B, n_cells, C, T) -> (B, n_cells, C)."""
        if x.dim() != 4:
            raise ValueError(
                f"Ожидается 4D тензор (B,n_cells,C,T), получено {tuple(x.shape)}"
            )
        frames = x.shape[-1]

        mean = x.mean(dim=-1, keepdim=True)
        probes = x[..., self._probe_indices(frames, x.device)]
        slope = ((x[..., -1] - x[..., 0]) / max(frames - 1, 1)).unsqueeze(-1)

        stats = torch.cat([mean, probes, slope], dim=-1)  # (B, n_cells, C, n_raw)
        return self.mix(stats).squeeze(-1)

    def _probe_indices(self, frames: int, device: torch.device) -> Tensor:
        """Равномерно распределённые по лог-сетке отсчёты информативного окна."""
        return torch.linspace(
            0, frames - 1, self.n_probe_points, device=device
        ).round().long()


class ClassificationHead(nn.Module):
    """Тип вещества: пусто / вода / эпоксидная смола.

    Вход (B, n_cells, C, T) -> логиты (B, n_cells, 3).
    """

    def __init__(
        self,
        channels: int = 32,
        n_probe_points: int = N_PROBE_POINTS,
        stats: TemporalStats | None = None,
    ) -> None:
        super().__init__()
        self.stats = stats if stats is not None else TemporalStats(n_probe_points)
        self.linear = nn.Linear(channels, len(SUBSTANCE_CLASSES))

    def forward(self, x: Tensor) -> Tensor:
        """(B, n_cells, C, T) -> (B, n_cells, 3) логиты."""
        return self.linear(self.stats(x))


class CoralOrdinalHead(nn.Module):
    """CORAL для количества воды: 3 градации (<30 / 30-70 / >70) -> 2 порога.

    Градации «нет воды» здесь нет — отсутствие воды решает
    `ClassificationHead`, а порядковая голова применяется к ячейкам класса
    «вода». Для K градаций CORAL требует ровно K-1 порогов.

    Пороги монотонны ПО ПОСТРОЕНИЮ: общий вектор весов даёт одно скалярное
    значение на ячейку, а смещения строятся как `b = -cumsum(softplus(raw))`,
    то есть строго убывают при любых значениях `raw`. Отсюда
    P(градация > 0) >= P(градация > 1) автоматически, без штрафов в loss.
    """

    def __init__(
        self,
        channels: int = 32,
        n_probe_points: int = N_PROBE_POINTS,
        stats: TemporalStats | None = None,
    ) -> None:
        super().__init__()
        self.stats = stats if stats is not None else TemporalStats(n_probe_points)
        self.projection = nn.Linear(channels, 1, bias=False)
        self.bias_raw = nn.Parameter(torch.zeros(N_THRESHOLDS))

    @property
    def thresholds(self) -> Tensor:
        """Строго убывающие смещения порогов, (N_THRESHOLDS,)."""
        return -torch.cumsum(F.softplus(self.bias_raw), dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """(B, n_cells, C, T) -> (B, n_cells, 2), вероятности P(градация > k)."""
        score = self.projection(self.stats(x))  # (B, n_cells, 1)
        return torch.sigmoid(score + self.thresholds)

    def predict_grade(self, probabilities: Tensor) -> Tensor:
        """Переводит P(градация > k) в индекс градации 0..2 порогом 0.5."""
        return (probabilities > 0.5).sum(dim=-1)
