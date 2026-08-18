import numpy as np
import pytest

from src.classification.classifier import (
    SUBSTANCE_CLASSES,
    ThermalClassifier,
    ZoneSample,
    build_matrix,
    evaluate,
)
from src.preprocessing.features import FEATURE_NAMES


def make_samples(n_per_class: int = 25, noise: float = 0.35) -> list[ZoneSample]:
    """Синтетические зоны: у веществ разная тепловая инерция и амплитуда.

    Разделимость заложена намеренно — тест проверяет работоспособность
    модели и метрик, а не физику конкретного образца.
    """
    rng = np.random.default_rng(42)
    centers = {
        "norm": (2.0, 1.0, 0.0),
        "water": (9.0, 6.0, 0.8),
        "gel": (7.0, 4.5, 0.6),
        "epoxy": (5.0, 8.0, 1.0),
    }

    samples: list[ZoneSample] = []
    for substance, (amplitude, decay, fill) in centers.items():
        for _ in range(n_per_class):
            vector = rng.normal(0.0, noise, size=len(FEATURE_NAMES)).astype(np.float32)
            vector[FEATURE_NAMES.index("amplitude_max")] += amplitude
            vector[FEATURE_NAMES.index("half_decay_time")] += decay
            vector[FEATURE_NAMES.index("integral")] += amplitude * 2
            samples.append(
                ZoneSample(
                    features=vector,
                    substance=substance,
                    fill_level=fill,
                    sample_name=f"synthetic-{substance}",
                )
            )
    return samples


def test_zone_sample_rejects_unknown_substance():
    with pytest.raises(ValueError, match="Неизвестный класс"):
        ZoneSample(
            features=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            substance="plasma",
            fill_level=0.5,
            sample_name="x",
        )


def test_zone_sample_rejects_out_of_range_fill():
    with pytest.raises(ValueError, match="fill_level"):
        ZoneSample(
            features=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            substance="water",
            fill_level=1.4,
            sample_name="x",
        )


def test_build_matrix_shapes():
    samples = make_samples(n_per_class=5)
    features, substances, fills = build_matrix(samples)
    assert features.shape == (20, len(FEATURE_NAMES))
    assert substances.shape == (20,)
    assert fills.shape == (20,)


def test_build_matrix_rejects_empty():
    with pytest.raises(ValueError):
        build_matrix([])


def test_classifier_learns_separable_substances():
    train = make_samples(n_per_class=30)
    test = make_samples(n_per_class=12, noise=0.4)

    model = ThermalClassifier().fit(train)
    metrics = evaluate(model, test)
    assert metrics["substance_accuracy"] > 0.8


def test_classifier_predicts_fill_level_within_tolerance():
    train = make_samples(n_per_class=30)
    test = make_samples(n_per_class=12, noise=0.4)

    model = ThermalClassifier().fit(train)
    metrics = evaluate(model, test)
    # Ориентир — 15% погрешности из методики ТПУ.
    assert metrics["fill_mae"] < 0.15


def test_predict_returns_probabilities_over_known_classes():
    model = ThermalClassifier().fit(make_samples(n_per_class=20))
    result = model.predict(make_samples(n_per_class=1)[0].features)

    assert result["substance"] in SUBSTANCE_CLASSES
    assert 0.0 <= result["fill_level"] <= 1.0
    assert result["probabilities"] and abs(sum(result["probabilities"].values()) - 1) < 1e-6


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="не обучена"):
        ThermalClassifier().predict(np.zeros(len(FEATURE_NAMES), dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path):
    model = ThermalClassifier().fit(make_samples(n_per_class=20))
    vector = make_samples(n_per_class=1)[0].features

    path = model.save(tmp_path / "model.pkl")
    restored = ThermalClassifier.load(path)

    assert restored.predict(vector)["substance"] == model.predict(vector)["substance"]
