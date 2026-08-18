import numpy as np
import pytest

from src.preprocessing.features import (
    FEATURE_NAMES,
    extract_features,
    subtract_background,
)
from src.utils.io import fps_from_filename, resolve_fps


def make_cube(
    height: int = 16,
    width: int = 16,
    n_frames: int = 120,
    fps: float = 10.0,
    amplitude: float = 10.0,
    tau: float = 3.0,
    baseline_s: float = 1.0,
    ramp_s: float = 2.0,
) -> np.ndarray:
    """Синтетический куб: плато фона, линейный нагрев, экспоненциальный спад.

    Плато в начале воспроизводит реальную запись, где регистрация стартует
    до включения нагревателя — именно эти кадры служат фоном.
    """
    t = np.arange(n_frames) / fps
    peak_t = baseline_s + ramp_s
    profile = np.zeros_like(t)
    ramp = (t >= baseline_s) & (t < peak_t)
    profile[ramp] = amplitude * (t[ramp] - baseline_s) / ramp_s
    decay = t >= peak_t
    profile[decay] = amplitude * np.exp(-(t[decay] - peak_t) / tau)

    cube = 20.0 + profile[None, None, :]
    return cube.repeat(height, 0).repeat(width, 1).astype(np.float32)


def test_subtract_background_zeroes_start():
    cube = make_cube()
    result = subtract_background(cube, n_background=5)
    assert np.allclose(result[:, :, :5].mean(), 0.0, atol=1e-4)


def test_extract_features_shapes_and_names():
    cube = make_cube()
    features = extract_features(cube, fps=10.0)
    assert features.maps.shape == (16, 16, len(FEATURE_NAMES))
    assert features.names == FEATURE_NAMES


def test_amplitude_matches_synthetic_signal():
    cube = make_cube(amplitude=12.0)
    features = extract_features(cube, fps=10.0)
    assert features["amplitude_max"].mean() == pytest.approx(12.0, rel=0.05)


def test_time_to_peak_matches_synthetic_signal():
    cube = make_cube(fps=10.0, baseline_s=1.0, ramp_s=2.0)
    features = extract_features(cube, fps=10.0)
    assert features["time_to_peak"].mean() == pytest.approx(3.0, abs=0.2)


def test_half_decay_time_tracks_thermal_inertia():
    """Вещество с большей инерцией (большим tau) остывает медленнее."""
    fast = extract_features(make_cube(tau=1.0), fps=10.0)["half_decay_time"].mean()
    slow = extract_features(make_cube(tau=6.0), fps=10.0)["half_decay_time"].mean()
    assert slow > fast


def test_extract_features_rejects_bad_input():
    with pytest.raises(ValueError):
        extract_features(np.zeros((4, 4), dtype=np.float32), fps=10.0)
    with pytest.raises(ValueError):
        extract_features(np.zeros((4, 4, 4), dtype=np.float32), fps=0.0)


def test_fps_from_filename():
    assert fps_from_filename("1 Итоговый образец 100мс 5с нагрев.7z") == 10.0
    assert fps_from_filename("sample 50мс.mat") == 20.0
    assert fps_from_filename("no period here.mat") is None


def test_resolve_fps_priority():
    assert resolve_fps(25.0, "x 100мс.mat") == 25.0  # значение из файла главнее
    assert resolve_fps(0.0, "x 100мс.mat") == 10.0  # затем имя файла
    assert resolve_fps(0.0, "unknown.mat", default=7.0) == 7.0  # затем умолчание
