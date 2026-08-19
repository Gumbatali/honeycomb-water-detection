import numpy as np
import pytest

from src.preprocessing.pct import (
    pct_feature_maps,
    principal_component_thermography,
    sign_invariant_feature_maps,
)


def make_cube(
    height: int = 12,
    width: int = 12,
    n_frames: int = 40,
    defect_sign: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Синтетический куб: фон затухает по экспоненте, дефектная зона — сильнее.

    `defect_sign` переворачивает знак отклонения дефекта от фона, чтобы
    проверить устойчивость `sign_invariant_feature_maps` к произвольному
    знаку компонент SVD.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=np.float64)
    background = np.exp(-t / 15.0)

    cube = np.tile(background, (height, width, 1))
    defect = np.zeros((height, width), dtype=bool)
    defect[height // 2 - 2 : height // 2 + 2, width // 2 - 2 : width // 2 + 2] = True
    cube[defect] += defect_sign * 0.5 * background
    cube += rng.normal(0.0, 0.01, size=cube.shape)
    return cube


def test_pct_rejects_non_3d_input():
    with pytest.raises(ValueError, match="куб"):
        principal_component_thermography(np.zeros((4, 4)))


def test_pct_rejects_out_of_range_n_components():
    cube = make_cube(height=3, width=3, n_frames=5)
    with pytest.raises(ValueError, match="n_components"):
        principal_component_thermography(cube, n_components=99)


def test_pct_rejects_all_constant_pixels():
    cube = np.ones((4, 4, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="постоянны"):
        principal_component_thermography(cube)


def test_pct_output_shapes():
    cube = make_cube()
    spatial, temporal = principal_component_thermography(cube, n_components=4)
    assert spatial.shape == (12, 12, 4)
    assert temporal.shape == (4, 40)


def test_pct_feature_maps_are_unit_scaled():
    cube = make_cube()
    maps = pct_feature_maps(cube, n_components=3)
    assert maps.shape == (3, 12, 12)
    # Каждая карта приведена к единичному стандартному отклонению по
    # построению (см. docstring `pct_feature_maps`).
    stds = maps.std(axis=(1, 2))
    np.testing.assert_allclose(stds, 1.0, atol=1e-5)


def test_sign_invariant_maps_stable_under_svd_sign_flip():
    """Тот же физический сигнал с противоположным знаком дефекта даёт
    компоненты, чей знак фона согласован (background > 0 по построению)."""
    cube_pos = make_cube(defect_sign=1.0, seed=1)
    cube_neg_signal = make_cube(defect_sign=1.0, seed=1)
    cube_neg_signal *= -1.0  # моделирует произвольный глобальный знак SVD

    maps_pos = sign_invariant_feature_maps(cube_pos, n_components=2)
    maps_neg = sign_invariant_feature_maps(cube_neg_signal, n_components=2)

    background_pos = np.median(maps_pos, axis=(1, 2))
    background_neg = np.median(maps_neg, axis=(1, 2))
    # После выравнивания по фону обе версии дают неотрицательный фон.
    assert np.all(background_pos >= -1e-6)
    assert np.all(background_neg >= -1e-6)


def test_sign_invariant_maps_separate_defect_from_background():
    cube = make_cube()
    maps = sign_invariant_feature_maps(cube, n_components=4)

    defect = np.zeros((12, 12), dtype=bool)
    defect[4:8, 4:8] = True
    background_level = np.median(maps[0][~defect])
    defect_level = maps[0][defect].mean()
    assert abs(defect_level - background_level) > 0.5
