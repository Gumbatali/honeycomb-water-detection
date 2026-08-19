import numpy as np
import pytest

from src.baseline.pct_classifier import (
    N_COMPONENTS,
    ZoneFeatures,
    build_classifier,
    leave_one_object_out,
    zone_features,
)
from src.preprocessing.pct import sign_invariant_feature_maps


def make_zone_cube(
    height: int = 10, width: int = 10, n_frames: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Куб с одной зоной в центре — вход для `zone_features`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=np.float64)
    background = np.exp(-t / 12.0)
    cube = np.tile(background, (height, width, 1))
    cube += rng.normal(0.0, 0.01, size=cube.shape)

    mask = np.zeros((height, width), dtype=bool)
    mask[3:7, 3:7] = True
    return cube, mask


def test_zone_features_shape():
    cube, mask = make_zone_cube()
    features = zone_features(cube, mask)
    assert features.shape == (2 * N_COMPONENTS,)


def make_synthetic_samples(
    n_per_class: int = 8, seed: int = 0
) -> list[ZoneFeatures]:
    """Синтетические `ZoneFeatures` для трёх "объектов" и двух классов.

    Разделимость заложена намеренно в средних и разбросах — тест проверяет
    протокол оценки (`leave_one_object_out`, балансировку классов), а не
    физику PCT, которая уже покрыта `test_pct.py`.
    """
    rng = np.random.default_rng(seed)
    samples: list[ZoneFeatures] = []
    class_centers = {0: -1.5, 1: 1.5}
    for video in ("obj1", "obj2", "obj3"):
        for substance, center in class_centers.items():
            for _ in range(n_per_class):
                vector = rng.normal(center, 0.3, size=2 * N_COMPONENTS)
                samples.append(
                    ZoneFeatures(
                        video=video, substance=substance, features=vector
                    )
                )
    return samples


def test_build_classifier_uses_balanced_class_weight():
    model = build_classifier()
    assert model.class_weight == "balanced"


def test_leave_one_object_out_separates_synthetic_classes():
    samples = make_synthetic_samples()
    accuracy, per_video = leave_one_object_out(samples)

    assert accuracy > 0.8
    assert set(per_video) == {"obj1", "obj2", "obj3"}
    for hits, total in per_video.values():
        assert 0 <= hits <= total


def test_leave_one_object_out_handles_imbalanced_classes():
    """Пять зон воды на одну зону смолы, как в реальных данных (раздел
    `build_classifier` docstring) — без балансировки классификатор
    вырождается в предсказание мажоритарного класса."""
    rng = np.random.default_rng(1)
    samples: list[ZoneFeatures] = []
    for video in ("obj1", "obj2"):
        for _ in range(10):
            samples.append(
                ZoneFeatures(
                    video=video,
                    substance=0,
                    features=rng.normal(-1.5, 0.3, size=2 * N_COMPONENTS),
                )
            )
        samples.append(
            ZoneFeatures(
                video=video,
                substance=1,
                features=rng.normal(1.5, 0.3, size=2 * N_COMPONENTS),
            )
        )

    accuracy, _ = leave_one_object_out(samples)
    assert accuracy > 0.5


def test_leave_one_object_out_rejects_empty_samples():
    with pytest.raises(ValueError, match="at least one array"):
        leave_one_object_out([])
