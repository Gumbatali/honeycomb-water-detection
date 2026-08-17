"""Агрегация признаков внутри ячейки — ДО голов и ДО loss.

Почему пуллинг стоит внутри forward, а не после инференса
(ARCHITECTURE.md раздел 5.3):

Вода заполняет ячейку целиком, поэтому истинная метка определена *на ячейку*,
а не на пиксель. Попиксельный loss заставлял бы модель тратить ёмкость на
различение физически одинаковых пикселей и раздувал бы мнимый размер выборки
до ~10^5 при реальных 864 ячейках и 12 независимых зонах. После пуллинга loss
считается по ячейкам — это соответствует реальной структуре данных.

Пиксели с `cell_index == 0` (фон, стенки сот, игнор) исключаются полностью:
стенки холоднее за счёт теплоотвода вдоль стенки и смещали бы среднее.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

BACKGROUND_INDEX: int = 0


class CellPooling(nn.Module):
    """Среднее признаков по пикселям каждой ячейки.

    Реализовано через `scatter_add_` по id ячейки с делением на счётчик
    пикселей. Операция дифференцируема: градиент по каждому пикселю равен
    1/n_pixels его ячейки. Ячейки без пикселей дают нули, а не NaN.
    """

    def forward(self, x: Tensor, cell_index: Tensor, n_cells: int) -> Tensor:
        """Агрегирует признаки по ячейкам.

        Args:
            x: (B, C, T, H, W) — карты признаков.
            cell_index: (B, H, W) int64 — id ячейки на пиксель,
                0 = фон/стенка/игнор, 1..n_cells = ячейки.
            n_cells: число ячеек в разметке.

        Returns:
            (B, n_cells, C, T) — по одному вектору признаков на ячейку.
        """
        self._validate(x, cell_index, n_cells)
        batch, channels, frames, height, width = x.shape
        pixels = height * width

        flat_index = cell_index.reshape(batch, pixels)
        # Слот 0 накапливает фон и затем отбрасывается — так фон не протекает
        # в ячейки и при этом не нужна маскированная индексация.
        counts = self._pixel_counts(flat_index, n_cells)

        # (B,C,T,H,W) -> (B, C*T, H*W) для scatter_add_ по пространству.
        values = x.reshape(batch, channels * frames, pixels)
        expanded = flat_index.unsqueeze(1).expand(batch, channels * frames, pixels)
        sums = torch.zeros(
            batch, channels * frames, n_cells + 1, dtype=x.dtype, device=x.device
        )
        sums = sums.scatter_add(2, expanded, values)

        means = sums / counts.clamp(min=1).unsqueeze(1).to(x.dtype)
        # Отбрасываем слот фона -> (B, n_cells, C, T).
        cells = means[:, :, 1:].reshape(batch, channels, frames, n_cells)
        return cells.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _pixel_counts(flat_index: Tensor, n_cells: int) -> Tensor:
        """(B, H*W) -> (B, n_cells+1): число пикселей на ячейку, слот 0 — фон."""
        batch = flat_index.shape[0]
        counts = torch.zeros(
            batch, n_cells + 1, dtype=torch.long, device=flat_index.device
        )
        return counts.scatter_add(1, flat_index, torch.ones_like(flat_index))

    @staticmethod
    def _validate(x: Tensor, cell_index: Tensor, n_cells: int) -> None:
        if x.dim() != 5:
            raise ValueError(f"Ожидается 5D тензор (B,C,T,H,W), получено {tuple(x.shape)}")
        if cell_index.dim() != 3:
            raise ValueError(
                f"cell_index должен быть (B,H,W), получено {tuple(cell_index.shape)}"
            )
        if cell_index.dtype != torch.long:
            raise TypeError(f"cell_index должен быть int64, получено {cell_index.dtype}")
        if n_cells < 1:
            raise ValueError(f"n_cells должно быть >= 1, получено {n_cells}")

        batch, _, _, height, width = x.shape
        if cell_index.shape != (batch, height, width):
            raise ValueError(
                f"cell_index {tuple(cell_index.shape)} не согласован с x {tuple(x.shape)}"
            )
        if int(cell_index.min()) < BACKGROUND_INDEX or int(cell_index.max()) > n_cells:
            raise ValueError(
                f"cell_index вне диапазона [0, {n_cells}]: "
                f"[{int(cell_index.min())}, {int(cell_index.max())}]"
            )
