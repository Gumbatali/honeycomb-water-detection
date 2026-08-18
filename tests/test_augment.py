import numpy as np
import pytest

from src.augmentation.augment import (
    AMPLITUDE_SCALED,
    add_sensor_noise,
    augment_maps,
    augment_vector,
    flip,
    rotate90,
    scale_amplitude,
)
from src.preprocessing.features import FEATURE_NAMES, FeatureMaps


def make_maps(height: int = 20, width: int = 30) -> FeatureMaps:
    rng = np.random.default_rng(3)
    maps = rng.uniform(1.0, 5.0, size=(height, width, len(FEATURE_NAMES))).astype(np.float32)
    return FeatureMaps(maps=maps, names=FEATURE_NAMES, fps=10.0, source="test")


def test_flip_is_reversible():
    features = make_maps()
    assert np.allclose(flip(flip(features)).maps, features.maps)


def test_rotate90_four_times_returns_original():
    features = make_maps(20, 20)
    assert np.allclose(rotate90(features, k=4).maps, features.maps)


def test_rotate90_swaps_dimensions():
    rotated = rotate90(make_maps(20, 30))
    assert rotated.maps.shape[:2] == (30, 20)


def test_scale_amplitude_affects_only_amplitude_features():
    features = make_maps()
    scaled = scale_amplitude(features, 2.0)

    for index, name in enumerate(FEATURE_NAMES):
        original = features.maps[:, :, index]
        result = scaled.maps[:, :, index]
        if name in AMPLITUDE_SCALED:
            assert np.allclose(result, original * 2.0)
        else:
            assert np.allclose(result, original)


def test_scale_amplitude_rejects_non_positive_factor():
    with pytest.raises(ValueError):
        scale_amplitude(make_maps(), 0.0)


def test_add_sensor_noise_perturbs_without_shifting_mean():
    features = make_maps(60, 60)
    noisy = add_sensor_noise(features, sigma=0.05, rng=np.random.default_rng(0))

    assert not np.allclose(noisy.maps, features.maps)
    assert noisy.maps.mean() == pytest.approx(features.maps.mean(), rel=0.02)


def test_augment_maps_returns_requested_count():
    variants = augment_maps(make_maps(), n_variants=6, rng=np.random.default_rng(1))
    assert len(variants) == 6
    assert all(v.maps.shape[2] == len(FEATURE_NAMES) for v in variants)


def test_augment_vector_shape_and_spread():
    vector = np.arange(1, len(FEATURE_NAMES) + 1, dtype=np.float32)
    variants = augment_vector(vector, n_variants=8, rng=np.random.default_rng(2))

    assert variants.shape == (8, len(FEATURE_NAMES))
    assert not np.allclose(variants[0], variants[1])


def test_augment_vector_stays_near_original():
    """Аугментация расширяет выборку, а не подменяет класс примера."""
    vector = np.full(len(FEATURE_NAMES), 10.0, dtype=np.float32)
    variants = augment_vector(vector, n_variants=50, rng=np.random.default_rng(4))

    assert np.abs(variants.mean(axis=0) - vector).max() < 1.0
