import numpy as np
import pytest

from src.preprocessing.features import FEATURE_NAMES
from src.synthesis.heat_transfer_model import (
    AIR,
    EPOXY,
    MATERIALS,
    WATER,
    HeatingConfig,
    profile_to_features,
    simulate_profile,
    synthesize_dataset,
)

FAST = HeatingConfig(duration=60.0)


def half_decay(profile: np.ndarray, fps: float = 10.0) -> float:
    peak = int(np.argmax(profile))
    below = np.where(profile[peak:] < profile[peak] * 0.5)[0]
    return float(below[0] / fps) if below.size else float("inf")


def test_water_has_higher_volumetric_capacity_than_air():
    assert WATER.volumetric_capacity > AIR.volumetric_capacity * 100


def test_profile_rises_during_heating_then_decays():
    _, profile = simulate_profile(1.0, WATER, FAST)
    peak = int(np.argmax(profile))

    assert profile[peak] > 0
    # Пик приходится на конец импульса нагрева (5 с при 10 Гц).
    assert 40 <= peak <= 60
    assert profile[-1] < profile[peak]


def test_amplitude_matches_measured_scale():
    """Модель откалибрована по реальному образцу: перегрев порядка 10-20 К."""
    _, profile = simulate_profile(1.0, WATER, FAST)
    assert 5.0 < profile.max() < 30.0


def test_filled_cell_damps_peak_relative_to_empty():
    """Вода поглощает тепло, поэтому перегрев обшивки ниже, чем над воздухом."""
    _, empty = simulate_profile(0.0, WATER, FAST)
    _, full = simulate_profile(1.0, WATER, FAST)
    assert full.max() < empty.max()


@pytest.mark.parametrize("material", ["water", "gel", "epoxy"])
def test_peak_decreases_monotonically_with_fill(material: str):
    peaks = [simulate_profile(f, material, FAST)[1].max() for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(later <= earlier for earlier, later in zip(peaks, peaks[1:]))


def test_materials_are_distinguishable_by_decay():
    """Смола держит тепло дольше воды — основа для их различения."""
    _, water = simulate_profile(1.0, WATER, FAST)
    _, epoxy = simulate_profile(1.0, EPOXY, FAST)
    assert half_decay(epoxy) > half_decay(water)


def test_simulate_rejects_invalid_fill():
    with pytest.raises(ValueError):
        simulate_profile(1.5, WATER, FAST)


def test_material_accepts_name_or_object():
    _, by_name = simulate_profile(1.0, "water", FAST)
    _, by_object = simulate_profile(1.0, MATERIALS["water"], FAST)
    assert np.allclose(by_name, by_object)


def test_profile_to_features_returns_full_vector():
    time, profile = simulate_profile(1.0, WATER, FAST)
    vector = profile_to_features(time, profile)

    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    assert vector[FEATURE_NAMES.index("amplitude_max")] == pytest.approx(profile.max(), rel=1e-3)


def test_synthesize_dataset_covers_requested_grid():
    dataset = synthesize_dataset(
        fill_levels=(0.0, 0.5, 1.0),
        materials=("water", "gel"),
        config=FAST,
        repeats=2,
    )
    assert len(dataset) == 3 * 2 * 2
    assert {row["substance"] for row in dataset} == {"norm", "water", "gel"}


def test_synthetic_zero_fill_is_labelled_norm():
    dataset = synthesize_dataset(
        fill_levels=(0.0,), materials=("water",), config=FAST, repeats=1
    )
    assert dataset[0]["substance"] == "norm"
