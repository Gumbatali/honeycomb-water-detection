"""Пространственный блок с НАМЕРЕННО ограниченным рецептивным полем.

Conv2d применяется к каждому временному срезу независимо (через reshape по
оси времени), дилатации 1-2-4, ядро 3. Рецептивное поле:

    RF = 1 + sum(2 * d) = 1 + 2 * (1 + 2 + 4) = 15 px = 1.31 ячейки
    (при масштабе 11.46 px на ячейку)

Расширять это поле нельзя. Обучающих объектов три, раскладка зон в кадре
фиксирована, и сеть с большим пространственным контекстом заучит именно
раскладку («в левом верхнем углу всегда 20%»), а не тепловой сигнал
(ARCHITECTURE.md разделы 0 и 5.2). Полутора ячеек хватает на подавление шума
и учёт непосредственных соседей — и не хватает на запоминание карты зон.
"""
from __future__ import annotations

from torch import Tensor, nn

SPATIAL_KERNEL: int = 3
SPATIAL_DILATIONS: tuple[int, ...] = (1, 2, 4)
GROUPS: int = 4
PIXELS_PER_CELL: float = 11.46


def spatial_receptive_field(
    kernel: int = SPATIAL_KERNEL, dilations: tuple[int, ...] = SPATIAL_DILATIONS
) -> int:
    """Рецептивное поле стека в пикселях (15 px при настройках по умолчанию)."""
    return 1 + sum((kernel - 1) * d for d in dilations)


class SpatialConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> GELU с сохранением размера кадра."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (SPATIAL_KERNEL - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=SPATIAL_KERNEL, dilation=dilation, padding=pad
        )
        self.norm = nn.GroupNorm(GROUPS, channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """(N, C, H, W) -> (N, C, H, W)."""
        return self.act(self.norm(self.conv(x)))


class SpatialBlock(nn.Module):
    """Три дилатированных Conv2d, применяемые к каждому временному срезу.

    Рецептивное поле 15 px = 1.31 ячейки — жёсткое проектное ограничение,
    см. докстринг модуля.
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        if channels % GROUPS != 0:
            raise ValueError(f"channels={channels} должно делиться на {GROUPS} групп")
        self.blocks = nn.ModuleList(
            SpatialConvBlock(channels, d) for d in SPATIAL_DILATIONS
        )
        self.channels = channels

    @property
    def receptive_field(self) -> int:
        """Рецептивное поле в пикселях."""
        return spatial_receptive_field()

    @property
    def receptive_field_cells(self) -> float:
        """Рецептивное поле в ячейках сотовой панели."""
        return self.receptive_field / PIXELS_PER_CELL

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T, H, W) -> (B, C, T, H, W).

        Временная ось сворачивается в батч, чтобы Conv2d обрабатывал каждый
        срез независимо и не смешивал время с пространством.
        """
        if x.dim() != 5:
            raise ValueError(f"Ожидается 5D тензор (B,C,T,H,W), получено {tuple(x.shape)}")
        batch, channels, frames, height, width = x.shape

        # (B,C,T,H,W) -> (B*T,C,H,W): время уходит в батч.
        slices = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        for block in self.blocks:
            slices = block(slices)
        return slices.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
