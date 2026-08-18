"""Тесты пайплайна обучения: датасет, CORAL-loss, смоук-тест этапа A."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.heads import CoralOrdinalHead
from src.model.net import HoneycombNet
from src.train.dataset import (
    CELL_ID,
    IGNORE_GRADE,
    N_CELLS,
    PATCH_SIZE,
    SyntheticCurveDataset,
    build_channels,
    fill_to_grade,
)
from src.train.losses import (
    N_THRESHOLDS,
    combined_loss,
    coral_loss,
    grades_to_coral_targets,
)
from src.train.metrics import EpochMetrics
from src.train.pretrain import TrainConfig, pretrain, split_indices

FRAMES = 64
SMALL = 24


@pytest.fixture(scope="module")
def dataset() -> SyntheticCurveDataset:
    """Небольшой датасет: генерация одной кривой ~0.15 с, держим модульно."""
    return SyntheticCurveDataset(n_samples=SMALL, seed=0)


# --------------------------------------------------------------------------
# Датасет
# --------------------------------------------------------------------------


def test_dataset_shapes_and_dtypes(dataset: SyntheticCurveDataset) -> None:
    x, cell_index, substance, grade = dataset[0]

    assert x.shape == (3, FRAMES, PATCH_SIZE, PATCH_SIZE)
    assert x.dtype == torch.float32
    assert cell_index.shape == (PATCH_SIZE, PATCH_SIZE)
    assert cell_index.dtype == torch.int64
    assert substance.dtype == torch.int64 and substance.ndim == 0
    assert grade.dtype == torch.int64 and grade.ndim == 0


def test_dataset_length_and_cell_index(dataset: SyntheticCurveDataset) -> None:
    assert len(dataset) == SMALL
    _, cell_index, _, _ = dataset[3]
    # Все пиксели патча принадлежат одной ячейке — фона на синтетике нет.
    assert torch.all(cell_index == CELL_ID)
    assert int(cell_index.max()) == N_CELLS


def test_dataset_is_deterministic() -> None:
    first = SyntheticCurveDataset(n_samples=6, seed=7)
    second = SyntheticCurveDataset(n_samples=6, seed=7)

    for index in range(len(first)):
        x_a, _, substance_a, grade_a = first[index]
        x_b, _, substance_b, grade_b = second[index]
        assert torch.equal(x_a, x_b)
        assert substance_a == substance_b and grade_a == grade_b


def test_repeated_getitem_returns_same_tensor(dataset: SyntheticCurveDataset) -> None:
    """Шум фиксирован при инициализации, а не рисуется на каждом обращении."""
    assert torch.equal(dataset[2][0], dataset[2][0])


def test_pixel_noise_makes_patch_non_uniform(dataset: SyntheticCurveDataset) -> None:
    """Без шума CellPooling усреднял бы идентичные пиксели (см. докстринг модуля)."""
    x, _, _, _ = dataset[1]
    spatial_std = x[0, 10].std()
    assert spatial_std > 0.0


def test_zero_noise_gives_uniform_patch() -> None:
    dataset = SyntheticCurveDataset(n_samples=2, seed=0, noise_sigma=0.0)
    x, _, _, _ = dataset[0]
    assert torch.allclose(x[0, 5], x[0, 5].flatten()[0].expand(PATCH_SIZE, PATCH_SIZE))


def test_grade_labels_follow_boundaries(dataset: SyntheticCurveDataset) -> None:
    for index in range(len(dataset)):
        _, _, substance, grade = dataset[index]
        if int(substance) == 1:  # вода
            assert 0 <= int(grade) <= 2
        else:
            assert int(grade) == IGNORE_GRADE


@pytest.mark.parametrize(
    ("fill", "expected"),
    [(0.05, 0), (0.29, 0), (0.3, 1), (0.5, 1), (0.7, 1), (0.71, 2), (1.0, 2)],
)
def test_fill_to_grade_boundaries(fill: float, expected: int) -> None:
    assert fill_to_grade(fill) == expected


def test_fill_to_grade_nan_is_ignored() -> None:
    assert fill_to_grade(float("nan")) == IGNORE_GRADE


def test_build_channels_are_finite_and_shaped() -> None:
    times = np.logspace(np.log10(5.5), np.log10(300.0), FRAMES)
    curves = np.exp(-times / 50.0)[None, :] * np.array([[5.0], [20.0]])

    channels = build_channels(curves, times)

    assert channels.shape == (2, 3, FRAMES)
    assert np.isfinite(channels).all()
    # Канал 2 — форма, растянутая ровно в [0, 1].
    assert channels[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert channels[:, 2].max() == pytest.approx(1.0, abs=1e-6)


def test_dataset_cache_roundtrip(tmp_path) -> None:
    cache = tmp_path / "curves.npz"
    first = SyntheticCurveDataset(n_samples=4, seed=3, cache_path=cache)
    assert cache.exists()

    second = SyntheticCurveDataset(n_samples=4, seed=3, cache_path=cache)
    assert torch.equal(first.channels, second.channels)
    assert torch.equal(first.substances, second.substances)


def test_cache_with_mismatched_params_is_regenerated(tmp_path) -> None:
    """Кэш от другой конфигурации не должен молча подменять данные."""
    cache = tmp_path / "curves.npz"
    SyntheticCurveDataset(n_samples=4, seed=3, cache_path=cache)

    other = SyntheticCurveDataset(n_samples=5, seed=3, cache_path=cache)
    assert len(other) == 5


# --------------------------------------------------------------------------
# CORAL loss
# --------------------------------------------------------------------------


def test_coral_targets_are_cumulative() -> None:
    grades = torch.tensor([0, 1, 2])
    targets = grades_to_coral_targets(grades)

    assert torch.equal(targets, torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))


def test_coral_loss_ignores_non_water() -> None:
    """Элементы с IGNORE_GRADE не должны влиять на значение loss."""
    probabilities = torch.tensor([[0.9, 0.8], [0.5, 0.5], [0.2, 0.1]])
    grades = torch.tensor([2, IGNORE_GRADE, 0])

    masked = coral_loss(probabilities, grades)
    reference = coral_loss(probabilities[[0, 2]], grades[[0, 2]])

    assert masked == pytest.approx(float(reference), abs=1e-6)


def test_coral_loss_unaffected_by_ignored_predictions() -> None:
    """Меняем предсказание на игнорируемой позиции — loss не двигается."""
    grades = torch.tensor([1, IGNORE_GRADE])
    base = coral_loss(torch.tensor([[0.7, 0.3], [0.5, 0.5]]), grades)
    changed = coral_loss(torch.tensor([[0.7, 0.3], [0.01, 0.99]]), grades)

    assert base == pytest.approx(float(changed), abs=1e-6)


def test_coral_loss_without_water_is_zero_not_nan() -> None:
    """Батч вообще без воды — нулевой вклад, без NaN."""
    probabilities = torch.rand(8, N_THRESHOLDS, requires_grad=True)
    grades = torch.full((8,), IGNORE_GRADE)

    loss = coral_loss(probabilities, grades)

    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()


def test_coral_loss_rewards_correct_ordering() -> None:
    grades = torch.tensor([2, 0])
    good = coral_loss(torch.tensor([[0.95, 0.95], [0.05, 0.05]]), grades)
    bad = coral_loss(torch.tensor([[0.05, 0.05], [0.95, 0.95]]), grades)

    assert float(good) < float(bad)


def test_combined_loss_without_water_is_finite() -> None:
    outputs = {
        "substance_logits": torch.randn(2, 3, 3, requires_grad=True),
        "water_ordinal_logits": torch.randn(2, 3, N_THRESHOLDS, requires_grad=True),
    }
    targets = {
        "substance": torch.tensor([[0, 2, 0], [2, 0, 2]]),
        "water_grade": torch.full((2, 3), IGNORE_GRADE),
    }

    total, components = combined_loss(outputs, targets)
    total.backward()

    assert torch.isfinite(total)
    assert components["ordinal"] == 0.0
    assert components["total"] == pytest.approx(components["substance"], abs=1e-6)


def test_combined_loss_components_sum_with_weight() -> None:
    torch.manual_seed(0)
    outputs = {
        "substance_logits": torch.randn(2, 2, 3),
        "water_ordinal_logits": torch.randn(2, 2, N_THRESHOLDS),
    }
    targets = {
        "substance": torch.tensor([[1, 1], [1, 0]]),
        "water_grade": torch.tensor([[0, 2], [1, IGNORE_GRADE]]),
    }

    total, components = combined_loss(outputs, targets, w_ordinal=0.5)

    expected = components["substance"] + 0.5 * components["ordinal"]
    assert float(total) == pytest.approx(expected, abs=1e-6)


def test_coral_gradient_survives_saturation() -> None:
    """Градиент не должен исчезать при уверенно неверном прогнозе.

    Регрессионный тест. Ранняя редакция принимала вероятности и брала от них
    логарифм с отсечкой на 1e-7. При логите -25 сигмоида равна 1.4e-11, отсечка
    срезала значение, и градиент обращался ровно в НОЛЬ: ячейка, ошибающаяся
    уверенно, переставала обучаться навсегда. Путь через логиты
    (`binary_cross_entropy_with_logits`, как в референсной реализации CORAL)
    даёт loss порядка |логит| и ненулевой градиент.
    """
    logits = torch.full((1, 1, N_THRESHOLDS), -25.0, requires_grad=True)
    grades = torch.full((1, 1), N_THRESHOLDS)  # верхняя градация: таргеты все 1

    loss = coral_loss(logits, grades)
    loss.backward()

    assert float(loss.detach()) > 20.0, "loss обязан расти по логиту, а не упираться в -log(eps)"
    assert logits.grad is not None
    assert torch.all(logits.grad.abs() > 1e-3), "градиент занулился — вернулась отсечка вероятностей"


def test_coral_thresholds_preinitialized_near_uniform() -> None:
    """Пороги стартуют около равновероятных градаций, а не с перекосом вниз.

    Референсная реализация CORAL преинициализирует смещения убывающими
    значениями и отмечает, что это ускоряет сходимость. Нулевая инициализация
    `bias_raw` дала бы softplus(0)=0.693 и стартовые вероятности 0.33/0.20 —
    перекос к нижней градации ещё до первого шага оптимизации.
    """
    head = CoralOrdinalHead()
    start = torch.sigmoid(head.thresholds)

    assert torch.all((start > 0.35) & (start < 0.65)), f"перекошенный старт: {start.tolist()}"
    assert torch.all(head.thresholds[:-1] > head.thresholds[1:])


def test_coral_thresholds_stay_monotonic_after_optimization() -> None:
    """Монотонность порогов — свойство конструкции, а не следствие данных."""
    torch.manual_seed(0)
    model = HoneycombNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(4, 3, FRAMES, PATCH_SIZE, PATCH_SIZE)
    cell_index = torch.ones(4, PATCH_SIZE, PATCH_SIZE, dtype=torch.long)
    targets = {
        "substance": torch.tensor([[1], [1], [1], [2]]),
        "water_grade": torch.tensor([[0], [2], [1], [IGNORE_GRADE]]),
    }

    for _ in range(20):
        loss, _ = combined_loss(model(x, cell_index, N_CELLS), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    thresholds = model.coral_head.thresholds
    assert torch.all(thresholds[:-1] > thresholds[1:])
    # Из монотонности порогов следует P(g>0) >= P(g>1) для любого входа.
    probabilities = model(x, cell_index, N_CELLS)["water_ordinal"]
    assert torch.all(probabilities[..., 0] >= probabilities[..., 1])


# --------------------------------------------------------------------------
# Метрики и сплит
# --------------------------------------------------------------------------


def test_split_indices_are_disjoint_and_complete() -> None:
    train, val = split_indices(100, 0.2, seed=0)

    assert len(val) == 20
    assert len(train) == 80
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(100))


def test_split_indices_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        split_indices(100, 1.5, seed=0)


def test_epoch_metrics_accumulate_correctly() -> None:
    metrics = EpochMetrics()
    outputs = {
        "substance_logits": torch.tensor([[[9.0, 0.0, 0.0]], [[0.0, 9.0, 0.0]]]),
        "water_ordinal": torch.tensor([[[0.1, 0.1]], [[0.9, 0.9]]]),
    }
    targets = {
        "substance": torch.tensor([[0], [1]]),
        "water_grade": torch.tensor([[IGNORE_GRADE], [2]]),
    }

    metrics.update(outputs, targets, loss=0.5, batch_size=2)

    assert metrics.substance_accuracy == pytest.approx(1.0)
    assert metrics.grade_mae == pytest.approx(0.0)
    assert metrics.loss == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Смоук-тест обучения
# --------------------------------------------------------------------------


def test_pretrain_smoke_reduces_loss(tmp_path) -> None:
    """2 эпохи на 200 примерах должны реально снижать train loss."""
    config = TrainConfig(
        n_samples=200,
        epochs=2,
        batch_size=32,
        seed=0,
        out=tmp_path / "pretrain.pt",
        cache=tmp_path / "curves.npz",
        verbose=False,
    )

    result = pretrain(config)
    history = result["history"]

    assert len(history) == 2
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert (tmp_path / "pretrain.pt").exists()


def test_pretrain_checkpoint_is_loadable(tmp_path) -> None:
    config = TrainConfig(
        n_samples=60,
        epochs=1,
        batch_size=16,
        seed=1,
        out=tmp_path / "ckpt.pt",
        cache=tmp_path / "curves.npz",
        verbose=False,
    )
    pretrain(config)

    payload = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    model = HoneycombNet()
    model.load_state_dict(payload["state_dict"])

    assert payload["stage"] == "A_pretrain_synthetic"
    assert payload["n_samples"] == 60
