import numpy as np
import pytest

from src.augmentation.gain import DEFAULT_GAIN_RANGE, apply_local_gain, random_local_gain


def make_cube(height: int = 10, width: int = 10, n_frames: int = 8) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(n_frames, dtype=np.float32)
    base = np.exp(-t / 4.0)
    return np.tile(base, (height, width, 1)).astype(np.float32) + rng.normal(
        0.0, 0.001, size=(height, width, n_frames)
    ).astype(np.float32)


def make_zone_mask(height: int = 10, width: int = 10) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[3:6, 3:6] = True
    return mask


def test_apply_local_gain_identity_at_gain_one():
    cube = make_cube()
    mask = make_zone_mask()
    result = apply_local_gain(cube, mask, gain=1.0)
    np.testing.assert_allclose(result, cube, atol=1e-6)


def test_apply_local_gain_leaves_pixels_outside_mask_untouched():
    cube = make_cube()
    mask = make_zone_mask()
    result = apply_local_gain(cube, mask, gain=1.5)
    np.testing.assert_array_equal(result[~mask], cube[~mask])


def test_apply_local_gain_scales_deviation_from_first_frame():
    cube = make_cube()
    mask = make_zone_mask()
    gain = 1.2

    result = apply_local_gain(cube, mask, gain=gain)

    baseline = cube[mask, :1]
    expected = baseline + gain * (cube[mask] - baseline)
    np.testing.assert_allclose(result[mask], expected, atol=1e-6)


def test_apply_local_gain_rejects_empty_mask():
    cube = make_cube()
    empty_mask = np.zeros(cube.shape[:2], dtype=bool)
    with pytest.raises(ValueError, match="пуста"):
        apply_local_gain(cube, empty_mask, gain=1.0)


def test_apply_local_gain_rejects_non_positive_gain():
    cube = make_cube()
    mask = make_zone_mask()
    with pytest.raises(ValueError, match="gain"):
        apply_local_gain(cube, mask, gain=0.0)


def test_apply_local_gain_rejects_mismatched_mask_shape():
    cube = make_cube()
    wrong_mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="Маска"):
        apply_local_gain(cube, wrong_mask, gain=1.0)


def test_random_local_gain_stays_within_default_range():
    cube = make_cube()
    mask = make_zone_mask()
    rng = np.random.default_rng(3)

    for _ in range(20):
        result = random_local_gain(cube, mask, rng)
        baseline = cube[mask, :1]
        deviation = cube[mask] - baseline
        implied_gain = np.divide(
            result[mask] - baseline, deviation, out=np.ones_like(deviation), where=deviation != 0
        )
        finite = implied_gain[np.isfinite(implied_gain)]
        if finite.size:
            assert DEFAULT_GAIN_RANGE[0] - 1e-3 <= finite.mean() <= DEFAULT_GAIN_RANGE[1] + 1e-3


def test_random_local_gain_reproducible_with_seeded_rng():
    cube = make_cube()
    mask = make_zone_mask()
    result1 = random_local_gain(cube, mask, np.random.default_rng(11))
    result2 = random_local_gain(cube, mask, np.random.default_rng(11))
    np.testing.assert_array_equal(result1, result2)
