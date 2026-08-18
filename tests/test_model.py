"""Тесты модели: формы, рецептивные поля, пуллинг по ячейкам, заморозка."""
from __future__ import annotations

import pytest
import torch

from src.model import (
    CellPooling,
    ClassificationHead,
    CoralOrdinalHead,
    HoneycombNet,
    SpatialBlock,
    TemporalEncoder,
    TemporalStats,
    measure_receptive_field,
    receptive_field,
    spatial_receptive_field,
)

BATCH = 2
N_CELLS = 10
FRAMES = 64
SIZE = 32
HIDDEN = 32


@pytest.fixture
def net() -> HoneycombNet:
    torch.manual_seed(0)
    return HoneycombNet()


@pytest.fixture
def cube() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, 3, FRAMES, SIZE, SIZE)


@pytest.fixture
def cell_index() -> torch.Tensor:
    """Разметка: полосы ячеек 1..N_CELLS, верхняя строка — фон."""
    index = torch.zeros(BATCH, SIZE, SIZE, dtype=torch.long)
    rows_per_cell = (SIZE - 1) // N_CELLS
    for cell in range(1, N_CELLS + 1):
        start = 1 + (cell - 1) * rows_per_cell
        index[:, start : start + rows_per_cell, :] = cell
    return index


class TestShapes:
    """Формы тензоров на каждом этапе пайплайна."""

    def test_temporal_encoder_shape(self, cube: torch.Tensor) -> None:
        out = TemporalEncoder()(cube)
        assert out.shape == (BATCH, HIDDEN, FRAMES, SIZE, SIZE)

    def test_spatial_block_preserves_shape(self) -> None:
        x = torch.randn(BATCH, HIDDEN, FRAMES, SIZE, SIZE)
        assert SpatialBlock(HIDDEN)(x).shape == x.shape

    def test_cell_pooling_shape(self, cell_index: torch.Tensor) -> None:
        x = torch.randn(BATCH, HIDDEN, FRAMES, SIZE, SIZE)
        out = CellPooling()(x, cell_index, N_CELLS)
        assert out.shape == (BATCH, N_CELLS, HIDDEN, FRAMES)

    def test_head_shapes(self) -> None:
        cells = torch.randn(BATCH, N_CELLS, HIDDEN, FRAMES)
        assert ClassificationHead(HIDDEN)(cells).shape == (BATCH, N_CELLS, 3)
        assert CoralOrdinalHead(HIDDEN)(cells).shape == (BATCH, N_CELLS, 2)

    def test_temporal_stats_keep_one_value_per_channel(self) -> None:
        """Статистики поканальные: (B, n_cells, C, T) -> (B, n_cells, C)."""
        stats = TemporalStats()
        cells = torch.randn(BATCH, N_CELLS, HIDDEN, FRAMES)
        assert stats(cells).shape == (BATCH, N_CELLS, HIDDEN)

    def test_temporal_stats_ignore_no_channel(self) -> None:
        """Изменение любого канала обязано менять выход этого канала.

        Защита от регресса к усреднению по каналам, которое стёрло бы то, что
        энкодер выучил в отдельных каналах.
        """
        torch.manual_seed(11)
        stats = TemporalStats()
        cells = torch.zeros(1, 1, HIDDEN, FRAMES)
        baseline = stats(cells)
        for channel in range(HIDDEN):
            perturbed = cells.clone()
            perturbed[0, 0, channel] += 1.0
            assert not torch.allclose(stats(perturbed)[0, 0, channel], baseline[0, 0, channel])

    def test_full_forward_shapes(
        self, net: HoneycombNet, cube: torch.Tensor, cell_index: torch.Tensor
    ) -> None:
        out = net(cube, cell_index, N_CELLS)
        assert out["substance_logits"].shape == (BATCH, N_CELLS, 3)
        assert out["water_ordinal"].shape == (BATCH, N_CELLS, 2)

    def test_ordinal_output_is_probability(
        self, net: HoneycombNet, cube: torch.Tensor, cell_index: torch.Tensor
    ) -> None:
        ordinal = net(cube, cell_index, N_CELLS)["water_ordinal"]
        assert torch.all((ordinal >= 0.0) & (ordinal <= 1.0))


class TestReceptiveField:
    """Рецептивные поля — проектные ограничения, а не гиперпараметры."""

    def test_temporal_receptive_field_theory(self) -> None:
        # RF = 1 + 4 * (1 + 2 + 4 + 8) = 61
        assert receptive_field() == 61

    def test_temporal_receptive_field_empirical(self) -> None:
        """Эмпирика по градиенту должна совпасть с теорией: ровно 61 из 64."""
        torch.manual_seed(7)
        assert measure_receptive_field(TemporalEncoder(), FRAMES) == 61

    def test_measurement_does_not_mutate_encoder(self) -> None:
        """Измерение работает на копии: нормализации исходной сети целы."""
        encoder = TemporalEncoder()
        measure_receptive_field(encoder, FRAMES)
        assert all(
            isinstance(block.norm, torch.nn.GroupNorm) for block in encoder.blocks
        )

    def test_groupnorm_couples_whole_time_axis(self) -> None:
        """С включённой GroupNorm измерение вырождается в n_frames.

        GroupNorm нормирует по всей временной оси, поэтому через её среднее и
        дисперсию каждый отсчёт влияет на каждый. Это связь нормировки, а не
        свёрточное рецептивное поле — отсюда `disable_norm=True` по умолчанию.
        """
        measured = measure_receptive_field(
            TemporalEncoder(), FRAMES, disable_norm=False
        )
        assert measured == FRAMES

    def test_temporal_receptive_field_fits_in_window(self) -> None:
        assert receptive_field() <= FRAMES

    def test_spatial_receptive_field_is_limited(self) -> None:
        """15 px = 1.31 ячейки — расширять нельзя (заучит раскладку зон)."""
        assert spatial_receptive_field() == 15
        assert SpatialBlock().receptive_field_cells == pytest.approx(1.31, abs=0.01)


class TestCellPooling:
    """Ключевой компонент: агрегация до голов и до loss."""

    @staticmethod
    def _synthetic() -> tuple[torch.Tensor, torch.Tensor]:
        """Ячейка 1 = 5.0, ячейка 2 = 7.0, фон = 1000.0."""
        x = torch.full((1, 1, 1, 4, 4), 1000.0)
        index = torch.zeros(1, 4, 4, dtype=torch.long)
        index[0, 0, :] = 1
        index[0, 1, :] = 2
        x[0, 0, 0, 0, :] = 5.0
        x[0, 0, 0, 1, :] = 7.0
        return x, index

    def test_averages_within_cell(self) -> None:
        x, index = self._synthetic()
        out = CellPooling()(x, index, n_cells=2)
        assert out[0, 0, 0, 0].item() == pytest.approx(5.0)
        assert out[0, 1, 0, 0].item() == pytest.approx(7.0)

    def test_background_does_not_leak(self) -> None:
        """Фон 1000.0 не должен влиять ни на одну ячейку."""
        x, index = self._synthetic()
        out = CellPooling()(x, index, n_cells=2)
        assert out.max().item() < 10.0

    def test_empty_cell_is_zero_not_nan(self) -> None:
        x, index = self._synthetic()
        out = CellPooling()(x, index, n_cells=5)  # ячейки 3..5 пустые
        empty = out[0, 2:]
        assert torch.all(empty == 0.0)
        assert not torch.isnan(out).any()

    def test_differentiable(self, cell_index: torch.Tensor) -> None:
        x = torch.randn(BATCH, 4, 8, SIZE, SIZE, requires_grad=True)
        CellPooling()(x, cell_index, N_CELLS).pow(2).mean().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_is_zero_on_background(self) -> None:
        """Пиксели фона исключены, значит и градиента по ним нет."""
        x, index = self._synthetic()
        x.requires_grad_(True)
        CellPooling()(x, index, n_cells=2).sum().backward()
        assert x.grad is not None
        assert torch.all(x.grad[0, 0, 0, 2:, :] == 0.0)

    def test_rejects_out_of_range_index(self) -> None:
        x = torch.randn(1, 1, 1, 4, 4)
        index = torch.full((1, 4, 4), 9, dtype=torch.long)
        with pytest.raises(ValueError, match="вне диапазона"):
            CellPooling()(x, index, n_cells=2)


class TestCoralMonotonicity:
    """Пороги CORAL монотонны по построению, а не за счёт штрафа в loss."""

    def test_two_thresholds_for_three_grades(self) -> None:
        assert CoralOrdinalHead(HIDDEN).thresholds.numel() == 2

    def test_monotonic_at_init(self) -> None:
        torch.manual_seed(3)
        head = CoralOrdinalHead(HIDDEN)
        with torch.no_grad():
            head.bias_raw.normal_(0.0, 2.0)
        thresholds = head.thresholds
        assert torch.all(thresholds[1:] < thresholds[:-1])

    def test_monotonic_after_optimizer_steps(self) -> None:
        torch.manual_seed(4)
        head = CoralOrdinalHead(HIDDEN)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.5)
        cells = torch.randn(BATCH, N_CELLS, HIDDEN, FRAMES)
        target = torch.rand(BATCH, N_CELLS, 2)

        for _ in range(20):
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy(head(cells), target).backward()
            optimizer.step()
            assert torch.all(head.thresholds[1:] < head.thresholds[:-1])

    def test_probabilities_are_ordered(self) -> None:
        """P(градация > 0) >= P(градация > 1) для каждой ячейки."""
        torch.manual_seed(5)
        head = CoralOrdinalHead(HIDDEN)
        with torch.no_grad():
            head.bias_raw.normal_(0.0, 1.0)
        probabilities = head(torch.randn(BATCH, N_CELLS, HIDDEN, FRAMES))
        assert torch.all(probabilities[..., 0] >= probabilities[..., 1])


class TestFreezeSchedule:
    """Расписание разморозки этапа B (ARCHITECTURE.md раздел 6)."""

    def test_component_parameter_counts(self, net: HoneycombNet) -> None:
        report = net.parameter_report()
        assert report["temporal_encoder"] == 16_224
        assert report["spatial_block"] == 27_936

    def test_heads_stay_within_budget(self, net: HoneycombNet) -> None:
        """Головы — сотня с небольшим параметров, как в ARCHITECTURE.md (~134).

        Раскладка: 3*32+3 = 99 (cls) + 32 (проекция coral) + 2 (зазоры порогов)
        + 1 (общий сдвиг порогов) + 5 (общее смешивание статистик) = 139.
        """
        assert net.parameter_report()["heads"] == 139

    def test_heads_share_one_temporal_summary(self, net: HoneycombNet) -> None:
        """Общий модуль статистик, а не две копии весов смешивания."""
        assert net.cls_head.stats is net.coral_head.stats

    def test_report_does_not_double_count_shared(self, net: HoneycombNet) -> None:
        """Сумма по компонентам сходится с total без задвоения общего модуля."""
        report = net.parameter_report()
        components = (
            report["temporal_encoder"] + report["spatial_block"] + report["heads"]
        )
        assert components == report["total"]

    def test_conservative_is_heads_only(self, net: HoneycombNet) -> None:
        trainable = net.freeze_for_finetune("conservative")
        heads = net.parameter_report()["heads"]
        assert trainable == heads
        assert trainable < 200

    def test_default_is_last_block_plus_heads(self, net: HoneycombNet) -> None:
        heads = net.parameter_report()["heads"]
        assert net.freeze_for_finetune("default") == 5_216 + heads

    def test_full_is_two_blocks_plus_heads(self, net: HoneycombNet) -> None:
        heads = net.parameter_report()["heads"]
        assert net.freeze_for_finetune("full") == 2 * 5_216 + heads

    def test_modes_are_strictly_nested(self, net: HoneycombNet) -> None:
        counts = [net.freeze_for_finetune(m) for m in ("conservative", "default", "full")]
        assert counts[0] < counts[1] < counts[2]

    @pytest.mark.parametrize("mode", ["conservative", "default", "full"])
    def test_spatial_block_always_frozen(self, net: HoneycombNet, mode: str) -> None:
        net.freeze_for_finetune(mode)
        assert not any(p.requires_grad for p in net.spatial_block.parameters())

    @pytest.mark.parametrize("mode", ["conservative", "default", "full"])
    def test_early_encoder_blocks_frozen(self, net: HoneycombNet, mode: str) -> None:
        net.freeze_for_finetune(mode)
        for block in (net.temporal_encoder.block1, net.temporal_encoder.block2):
            assert not any(p.requires_grad for p in block.parameters())

    def test_unfreeze_all_opens_everything(self, net: HoneycombNet) -> None:
        assert net.unfreeze_all() == net.parameter_report()["total"]

    def test_unknown_mode_rejected(self, net: HoneycombNet) -> None:
        with pytest.raises(ValueError, match="Неизвестный режим"):
            net.freeze_for_finetune("everything")


class TestForwardBackward:
    """Полный проход вперёд-назад на обоих loss."""

    def test_full_forward_backward(
        self, net: HoneycombNet, cube: torch.Tensor, cell_index: torch.Tensor
    ) -> None:
        out = net(cube, cell_index, N_CELLS)
        labels = torch.randint(0, 3, (BATCH, N_CELLS))
        grades = torch.rand(BATCH, N_CELLS, 2)

        loss = torch.nn.functional.cross_entropy(
            out["substance_logits"].reshape(-1, 3), labels.reshape(-1)
        ) + torch.nn.functional.binary_cross_entropy(out["water_ordinal"], grades)
        loss.backward()

        gradients = [p.grad for p in net.parameters() if p.grad is not None]
        assert gradients
        assert not any(torch.isnan(g).any() for g in gradients)

    def test_backward_only_touches_unfrozen(
        self, net: HoneycombNet, cube: torch.Tensor, cell_index: torch.Tensor
    ) -> None:
        net.freeze_for_finetune("default")
        net(cube, cell_index, N_CELLS)["substance_logits"].pow(2).mean().backward()

        assert all(p.grad is None for p in net.temporal_encoder.block1.parameters())
        assert any(p.grad is not None for p in net.temporal_encoder.block4.parameters())
