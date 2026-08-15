"""Устойчивость детектора к выбору порога и к слипанию соседних зон."""
import numpy as np
import pytest

from src.detection.zones import detect_zones
from src.preprocessing.features import FEATURE_NAMES, FeatureMaps


def make_panel(zone_gap: int = 14, zone_size: int = 26, contrast: float = 6.0) -> FeatureMaps:
    """Панель 2x3 с дефектами — геометрия как у реального образца."""
    rng = np.random.default_rng(1)
    height, width = 140, 200
    maps = rng.normal(0.0, 0.05, size=(height, width, len(FEATURE_NAMES))).astype(np.float32)

    amp_idx = FEATURE_NAMES.index("amplitude_max")
    decay_idx = FEATURE_NAMES.index("half_decay_time")
    maps[20:120, 30:170, amp_idx] += 12.0
    maps[20:120, 30:170, decay_idx] += 10.0

    for row_i in range(2):
        for col_i in range(3):
            row = 38 + row_i * (zone_size + zone_gap)
            col = 48 + col_i * (zone_size + zone_gap)
            maps[row : row + zone_size, col : col + zone_size, decay_idx] -= contrast

    return FeatureMaps(maps=maps, names=FEATURE_NAMES, fps=10.0, source="panel-2x3")


@pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2, 1.5])
def test_detects_six_zones_across_threshold_range(sigma: float):
    """Число зон не должно зависеть от точной настройки порога."""
    zones = detect_zones(make_panel(), threshold_sigma=sigma, min_area=150)
    assert len(zones) == 6


def test_merged_zones_are_split_by_watershed():
    """Близко расположенные зоны разделяются, а не отбрасываются как блоб."""
    tight = make_panel(zone_gap=4)
    zones = detect_zones(tight, threshold_sigma=0.9, min_area=150)
    assert len(zones) >= 4


def test_zone_sizes_are_plausible():
    zones = detect_zones(make_panel(zone_size=26), threshold_sigma=1.0, min_area=150)
    for zone in zones:
        assert 18 <= zone.height <= 40
        assert 18 <= zone.width <= 40
