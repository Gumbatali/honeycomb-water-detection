"""Пространственная аугментация термограмм с учётом разметки зон.

Наивная реализация (`scripts/augmentations.py::HoneycombSpatialAugmenter`,
предшествовавшая этому модулю) поворачивала пиксели кадра, но не трогала
`bbox_xyxy` разметки — после произвольного поворота bbox переставал
покрывать повёрнутую зону, и разметка расходилась с данными. У vooterr
(ветка `gru-experiments`, `scripts/create_video_augmentation_experiments.py`)
это решено через трансформацию углов bbox той же аффинной матрицей и взятие
нового axis-aligned прямоугольника — тот же приём перенесён сюда и
проверен на разметке `water1` (6 зон, 480x640): при повороте на 60°, что
у прежней реализации считалось "безопасным" углом для гексагональной
решётки, ни одна зона не выходит за границы кадра и зоны не пересекаются
между собой (крайние зоны отстоят друг от друга на десятки пикселей).

Матрица одна на весь кадр/клип и на все зоны — сдвиг/поворот применяется
целиком к панели, а не к отдельным ячейкам (в отличие от cell permutation).
"""
from __future__ import annotations

import cv2
import numpy as np

#: Углы, кратные этому значению, совпадают с осями симметрии гексагональной
#: решётки сот и меньше искажают взаимное расположение зон при повороте —
#: не обязательное ограничение (bbox трансформируется для любого угла), но
#: удобный набор значений по умолчанию для равномерного покрытия ориентаций.
HEX_SYMMETRY_STEP_DEG: int = 60


def rotation_matrix(image_shape: tuple[int, int], angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """Матрица поворота вокруг центра кадра. ``image_shape`` — (H, W)."""
    h, w = image_shape
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale).astype(np.float32)


def flip_matrix(image_shape: tuple[int, int], axis: str) -> np.ndarray:
    """Матрица зеркального отражения. ``axis``: ``"horizontal"`` или ``"vertical"``."""
    h, w = image_shape
    if axis == "horizontal":
        return np.array([[-1, 0, w - 1], [0, 1, 0]], dtype=np.float32)
    if axis == "vertical":
        return np.array([[1, 0, 0], [0, -1, h - 1]], dtype=np.float32)
    raise ValueError(f"axis должен быть 'horizontal' или 'vertical', получено {axis!r}")


def transform_frame(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Применяет аффинную матрицу к кадру или временному кубу (H, W[, N]).

    Отражение краёв (`BORDER_REFLECT`) не вносит резких нулевых границ —
    важно для последующего вычисления контраста, где скачок на границе
    воспринимался бы как отдельный тепловой объект.
    """
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        return cv2.warpAffine(
            frame.astype(np.float32), matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        ).astype(frame.dtype)

    warped = np.empty_like(frame)
    for i in range(frame.shape[2]):
        warped[:, :, i] = cv2.warpAffine(
            frame[:, :, i].astype(np.float32), matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        ).astype(frame.dtype)
    return warped


def transform_bbox(
    bbox_xyxy: tuple[int, int, int, int], matrix: np.ndarray, image_shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Трансформирует bbox зоны той же матрицей, что и кадр.

    Углы bbox переводятся аффинным преобразованием, затем берётся
    наименьший осе-выровненный прямоугольник, покрывающий все четыре
    трансформированных угла — обычный приём для аффинной аугментации
    детекции/сегментации, а не точное отображение прямоугольника в
    прямоугольник (которое существует только для поворотов, кратных 90°).

    Returns:
        Новый ``bbox_xyxy``, обрезанный по границам кадра, либо ``None``,
        если после обрезки зона схлопнулась (полностью ушла за кадр).
    """
    x1, y1, x2, y2 = bbox_xyxy
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    mapped = cv2.transform(corners, matrix)[0]

    h, w = image_shape
    left, top = np.maximum(np.floor(mapped.min(axis=0)), (0, 0))
    right, bottom = np.minimum(np.ceil(mapped.max(axis=0)), (w, h))
    if right <= left or bottom <= top:
        return None
    return (int(left), int(top), int(right), int(bottom))


def transform_cells(
    cells: list[dict], matrix: np.ndarray, image_shape: tuple[int, int]
) -> list[dict]:
    """Применяет `transform_bbox` ко всем зонам, сохраняя остальные поля.

    Зоны, полностью ушедшие за кадр после трансформации, отбрасываются —
    при малых аффинных возмущениях (повороты в единицы-десятки градусов,
    сдвиги в единицы-десятки пикселей) этого не происходит на реальной
    разметке (проверено на water1/water2/water4), но большие повороты или
    сильные сдвиги могут вытолкнуть крайнюю зону за границу.
    """
    result: list[dict] = []
    for cell in cells:
        transformed = transform_bbox(tuple(cell["bbox_xyxy"]), matrix, image_shape)
        if transformed is None:
            continue
        result.append({**cell, "bbox_xyxy": list(transformed)})
    return result


def random_affine(
    image_shape: tuple[int, int],
    rng: np.random.Generator,
    max_angle_deg: float = 15.0,
    max_shift_px: float = 8.0,
    scale_range: tuple[float, float] = (0.95, 1.05),
) -> np.ndarray:
    """Случайная аффинная матрица: поворот + сдвиг + масштаб, без сдвига (shear).

    Диапазоны по умолчанию (углы до 15°, сдвиг до 8px) намеренно умеренные —
    ARCHITECTURE.md фиксирует минимальный отступ зоны от края кропа в 7px
    (`MIN_RF_MARGIN_PX`), и слишком агрессивная аугментация чаще выталкивала
    бы крайние зоны за пределы `_union_crop`.
    """
    angle = float(rng.uniform(-max_angle_deg, max_angle_deg))
    scale = float(rng.uniform(*scale_range))
    matrix = rotation_matrix(image_shape, angle, scale)
    dx = float(rng.uniform(-max_shift_px, max_shift_px))
    dy = float(rng.uniform(-max_shift_px, max_shift_px))
    matrix[:, 2] += (dx, dy)
    return matrix
