"""Временной энкодер: дилатированные свёртки только вдоль оси времени.

Ядро Conv3d имеет форму (k, 1, 1), поэтому свёртка идёт исключительно по
времени, а веса общие для всех пикселей кадра. Это принципиально: сеть учит
*форму кривой остывания*, а не расположение зон в кадре — при 12 независимых
зонах в train любая пространственно-специфичная ёмкость мгновенно
переобучается (ARCHITECTURE.md раздел 0).

Рецептивное поле: RF = 1 + sum((k - 1) * d) = 1 + 4 * (1 + 2 + 4 + 8) = 61
отсчёт из 64 — почти всё информативное окно [5.5; 300] с.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

TEMPORAL_KERNEL: int = 5
DILATIONS: tuple[int, ...] = (1, 2, 4, 8)
GROUPS: int = 4


def receptive_field(
    kernel: int = TEMPORAL_KERNEL, dilations: tuple[int, ...] = DILATIONS
) -> int:
    """Теоретическое рецептивное поле стека дилатированных свёрток."""
    return 1 + sum((kernel - 1) * d for d in dilations)


class TemporalConvBlock(nn.Module):
    """Один блок: Conv3d(k,1,1) -> GroupNorm -> GELU.

    GroupNorm, а не BatchNorm: независимых обучающих единиц всего 12, батчи
    получаются маленькими, и батч-статистики становятся шумом.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (TEMPORAL_KERNEL - 1) // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(TEMPORAL_KERNEL, 1, 1),
            dilation=(dilation, 1, 1),
            padding=(pad, 0, 0),
        )
        self.norm = nn.GroupNorm(GROUPS, out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """(B, C_in, T, H, W) -> (B, C_out, T, H, W)."""
        return self.act(self.norm(self.conv(x)))


class TemporalEncoder(nn.Module):
    """Четыре блока Conv3d с дилатациями 1-2-4-8, рецептивное поле 61 из 64.

    Блоки — именованные атрибуты `block1..block4`, чтобы протокол обучения
    (ARCHITECTURE.md раздел 6) мог размораживать их поимённо.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 32) -> None:
        super().__init__()
        if hidden % GROUPS != 0:
            raise ValueError(f"hidden={hidden} должно делиться на {GROUPS} групп")

        channels = (in_channels, hidden, hidden, hidden)
        self.block1 = TemporalConvBlock(channels[0], hidden, DILATIONS[0])
        self.block2 = TemporalConvBlock(channels[1], hidden, DILATIONS[1])
        self.block3 = TemporalConvBlock(channels[2], hidden, DILATIONS[2])
        self.block4 = TemporalConvBlock(channels[3], hidden, DILATIONS[3])

        self.in_channels = in_channels
        self.hidden = hidden

    @property
    def blocks(self) -> tuple[TemporalConvBlock, ...]:
        """Блоки в порядке применения — для расписания разморозки."""
        return (self.block1, self.block2, self.block3, self.block4)

    @property
    def receptive_field(self) -> int:
        """Рецептивное поле по времени в отсчётах."""
        return receptive_field()

    def forward(self, x: Tensor) -> Tensor:
        """(B, in_channels, T, H, W) -> (B, hidden, T, H, W)."""
        if x.dim() != 5:
            raise ValueError(f"Ожидается 5D тензор (B,C,T,H,W), получено {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Ожидается {self.in_channels} входных канала, получено {x.shape[1]}"
            )
        for block in self.blocks:
            x = block(x)
        return x


def measure_receptive_field(
    encoder: TemporalEncoder, n_frames: int = 64, disable_norm: bool = True
) -> int:
    """Эмпирически измеряет рецептивное поле по градиенту центрального отсчёта.

    Ненулевой градиент входа по выходу в центре временной оси покрывает ровно
    те отсчёты, которые влияют на выход, — это и есть рецептивное поле.

    `disable_norm=True` обязателен для честного измерения: GroupNorm считает
    среднее и дисперсию по всей временной оси, поэтому через её статистики
    каждый отсчёт влияет на каждый. Это глобальная связь нормировки, а не
    свёрточная связность, и без её отключения измерение даёт тривиальные
    `n_frames` вместо реального рецептивного поля.
    """
    probe = copy.deepcopy(encoder)
    if disable_norm:
        for block in probe.blocks:
            block.norm = nn.Identity()

    probe.eval()
    x = torch.zeros(1, probe.in_channels, n_frames, 1, 1, requires_grad=True)
    out = probe(x)
    out[0, :, n_frames // 2, 0, 0].sum().backward()

    grad = x.grad
    if grad is None:
        raise RuntimeError("Градиент по входу не посчитан")
    influential = (grad.abs().sum(dim=(0, 1, 3, 4)) > 0).nonzero().flatten()
    if influential.numel() == 0:
        raise RuntimeError(
            "Нулевой градиент на всех отсчётах — веса вырождены, "
            "измерение невозможно"
        )
    return int(influential.max() - influential.min()) + 1
