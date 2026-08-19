import numpy as np

from src.detection.zones import Zone, detect_zones, panel_mask, zone_profile
from src.preprocessing.features import FEATURE_NAMES, FeatureMaps


def make_feature_maps(
    height: int = 120,
    width: int = 160,
    n_zones: int = 4,
    zone_size: int = 24,
) -> FeatureMaps:
    """Синтетическая панель: тёплый прямоугольник с квадратными дефектами.

    Дефекты моделируются пониженным half_decay_time — так они выглядят
    на реальных данных из-за иной тепловой инерции наполнителя.
    """
    rng = np.random.default_rng(0)
    maps = rng.normal(0.0, 0.02, size=(height, width, len(FEATURE_NAMES))).astype(np.float32)

    panel = (slice(20, height - 20), slice(30, width - 30))
    maps[panel[0], panel[1], FEATURE_NAMES.index("amplitude_max")] += 12.0
    maps[panel[0], panel[1], FEATURE_NAMES.index("half_decay_time")] += 10.0

    decay_idx = FEATURE_NAMES.index("half_decay_time")
    for i in range(n_zones):
        row = 35 + (i // 2) * (zone_size + 15)
        col = 45 + (i % 2) * (zone_size + 20)
        maps[row : row + zone_size, col : col + zone_size, decay_idx] -= 6.0

    return FeatureMaps(maps=maps, names=FEATURE_NAMES, fps=10.0, source="synthetic")


def test_panel_mask_selects_interior_region():
    features = make_feature_maps()
    mask = panel_mask(features)
    assert mask[60, 80]  # центр панели
    assert not mask[2, 2]  # угол сцены — фон


def test_detect_zones_finds_all_synthetic_defects():
    features = make_feature_maps(n_zones=4)
    zones = detect_zones(features, min_area=100)
    assert len(zones) == 4


def test_detected_zones_match_expected_size():
    """Контур крупнее заданного дефекта, но остаётся правдоподобным.

    Консенсус объединяет голоса через OR: GMM и PCA видят контраст чуть
    шире одиночного порога по half_decay_time, поэтому верхняя граница
    поднята с 32 до 40px — это стабильное расширение контура, а не
    случайный выброс (см. test_zone_robustness.test_zone_sizes_are_plausible).
    """
    features = make_feature_maps(n_zones=4, zone_size=24)
    zones = detect_zones(features, min_area=100)
    for zone in zones:
        assert 18 <= zone.height <= 40
        assert 18 <= zone.width <= 40


def test_detect_zones_returns_empty_without_defects():
    features = make_feature_maps(n_zones=0)
    assert detect_zones(features, min_area=100) == []


def test_zones_sorted_by_descending_score():
    zones = detect_zones(make_feature_maps(), min_area=100)
    scores = [zone.score for zone in zones]
    assert scores == sorted(scores, reverse=True)


def test_aspect_ratio_filter_rejects_elongated_artifacts():
    """Вытянутые аномалии (кромки, оснастка) не должны попадать в результат."""
    features = make_feature_maps(n_zones=0)
    maps = features.maps.copy()
    decay_idx = FEATURE_NAMES.index("half_decay_time")
    maps[50:56, 40:130, decay_idx] -= 6.0  # полоса 6x90 -> aspect 15
    elongated = FeatureMaps(maps=maps, names=FEATURE_NAMES, fps=10.0)
    assert detect_zones(elongated, min_area=100, max_aspect_ratio=3.0) == []


def test_zone_profile_returns_all_features():
    features = make_feature_maps()
    zone = Zone(row0=35, col0=45, row1=59, col1=69, score=1.0)
    profile = zone_profile(features, zone)
    assert set(profile) == set(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in profile.values())
