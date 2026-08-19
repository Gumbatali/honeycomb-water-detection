import numpy as np
import pytest

from src.augmentation.spatial import (
    flip_matrix,
    random_affine,
    rotation_matrix,
    transform_bbox,
    transform_cells,
    transform_frame,
)


def test_transform_frame_preserves_shape_2d():
    frame = np.random.default_rng(0).uniform(0, 1, size=(40, 60)).astype(np.float32)
    matrix = rotation_matrix((40, 60), angle_deg=15.0)
    result = transform_frame(frame, matrix)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_transform_frame_preserves_shape_3d():
    cube = np.random.default_rng(0).uniform(0, 1, size=(30, 30, 5)).astype(np.float32)
    matrix = rotation_matrix((30, 30), angle_deg=6.0)
    result = transform_frame(cube, matrix)
    assert result.shape == cube.shape


def test_flip_matrix_horizontal_mirrors_frame():
    frame = np.zeros((10, 20), dtype=np.float32)
    frame[:, 0] = 1.0  # маркер на левом краю

    matrix = flip_matrix((10, 20), axis="horizontal")
    flipped = transform_frame(frame, matrix)

    # После горизонтального зеркала маркер должен оказаться у правого края.
    assert flipped[:, -1].mean() > flipped[:, 0].mean()


def test_flip_matrix_rejects_unknown_axis():
    with pytest.raises(ValueError, match="axis"):
        flip_matrix((10, 10), axis="diagonal")


def test_transform_bbox_identity_matrix_keeps_box():
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    box = (10, 20, 30, 40)
    result = transform_bbox(box, matrix, image_shape=(480, 640))
    assert result == box


def test_transform_bbox_real_zone_survives_60_degree_rotation():
    """Регрессия на находке из сессии: старый rotate_hex не трогал bbox
    вообще, разметка расходилась с кадром. Проверяем на реальном bbox
    water1 (ARCHITECTURE.md), что после поворота зона остаётся внутри
    кадра и не схлопывается."""
    image_shape = (480, 640)
    matrix = rotation_matrix(image_shape, angle_deg=60.0)
    box = (218, 79, 291, 147)  # water80, water1.json

    result = transform_bbox(box, matrix, image_shape)

    assert result is not None
    left, top, right, bottom = result
    h, w = image_shape
    assert 0 <= left < right <= w
    assert 0 <= top < bottom <= h


def test_transform_bbox_returns_none_when_box_leaves_frame():
    image_shape = (100, 100)
    matrix = np.array([[1, 0, 500], [0, 1, 500]], dtype=np.float32)  # огромный сдвиг
    result = transform_bbox((10, 10, 20, 20), matrix, image_shape)
    assert result is None


def test_transform_cells_drops_out_of_frame_zones():
    image_shape = (100, 100)
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cells = [
        {"class_name": "water20", "bbox_xyxy": [10, 10, 20, 20]},
        {"class_name": "water40", "bbox_xyxy": [-500, -500, -490, -490]},
    ]
    result = transform_cells(cells, matrix, image_shape)
    assert [c["class_name"] for c in result] == ["water20"]


def test_transform_cells_preserves_other_fields():
    image_shape = (100, 100)
    matrix = np.array([[1, 0, 2], [0, 1, 3]], dtype=np.float32)
    cells = [{"class_name": "water60", "bbox_xyxy": [10, 10, 20, 20], "extra": 42}]
    result = transform_cells(cells, matrix, image_shape)
    assert result[0]["extra"] == 42
    assert result[0]["bbox_xyxy"] != [10, 10, 20, 20]


def test_random_affine_is_reproducible_with_seeded_rng():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    m1 = random_affine((480, 640), rng1)
    m2 = random_affine((480, 640), rng2)
    np.testing.assert_array_equal(m1, m2)


def test_random_affine_respects_angle_bound():
    rng = np.random.default_rng(0)
    identity_scale_check = []
    for _ in range(20):
        matrix = random_affine((480, 640), rng, max_angle_deg=1.0, max_shift_px=0.0, scale_range=(1.0, 1.0))
        # При малом угле матрица близка к единичной (без учёта сдвига центра).
        identity_scale_check.append(np.allclose(matrix[:, :2], np.eye(2), atol=0.05))
    assert all(identity_scale_check)
