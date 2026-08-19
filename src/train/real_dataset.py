"""Датасет этапа B: реальные записи water1/water2/water4 -> вход `HoneycombNet`.

Разметка (`honeycomb/metadata/videos/*.json`) даёт bbox только на уровне ЗОН
(6 на объект), а не отдельных сотовых ячеек внутри зоны — маска
`*_dark_cells_mask.png` бинарна («тёмная ячейка» да/нет) и не несёт id
отдельных сот. Поэтому `n_cells` здесь равно числу зон объекта (6), а не
числу физических сот (6*72=432): `cell_index` строится из bbox зоны с
эрозией внутрь на `ERODE_PX`, что и есть та самая «единица независимости —
зона» из ARCHITECTURE.md раздела 0.

Класс `water120` в разметке кодирует эпоксидную смолу, а не 120% воды
(ARCHITECTURE.md раздел 11.1) — подтверждено внешней работой автора съёмки и
измерением контраста (смола лежит между 20% и 40% воды, а не выше 100%).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.model.heads import SUBSTANCE_CLASSES
from src.preprocessing.contrast import build_input_tensor
from src.train.dataset import IGNORE_GRADE, fill_to_grade

#: Эрозия маски зоны внутрь от границы bbox, px. Убирает стенки/боковую
#: диффузию у края зоны, где метка пикселя неопределена (ARCHITECTURE.md
#: раздел 7).
ERODE_PX: int = 4

#: Опорная (бездефектная) область термограммы для нормировки контраста.
#: Центральная полоса панели вне всех размеченных зон — тот же прямоугольник,
#: которым в этой сессии уже проверялась физика на water1.
REFERENCE_BBOX: tuple[int, int, int, int] = (220, 180, 460, 215)  # x1,y1,x2,y2

#: Частота регистрации термограмм, Гц (ARCHITECTURE.md раздел 0).
FPS: float = 10.0

#: Класс "смола" в кодировке имени зоны разметки.
EPOXY_ZONE_NAME: str = "water120"


@dataclass(frozen=True)
class ZoneLabel:
    """Истинная метка одной зоны: вещество + доля заполнения (для воды)."""

    substance: int  # индекс в SUBSTANCE_CLASSES
    fill_fraction: float  # доля 0..1, NaN если substance != "water"


def zone_name_to_label(class_name: str) -> ZoneLabel:
    """Переводит имя зоны из разметки в метку вещества и доли заполнения.

    Args:
        class_name: например ``"water80"``, ``"water120"``.

    Returns:
        Метка. ``water120`` -> эпоксидная смола (ARCHITECTURE.md раздел 11.1).
    """
    if class_name == EPOXY_ZONE_NAME:
        return ZoneLabel(substance=SUBSTANCE_CLASSES.index("epoxy"), fill_fraction=float("nan"))

    match = re.fullmatch(r"water(\d+)", class_name)
    if match is None:
        raise ValueError(f"Нераспознанное имя зоны: {class_name!r}")
    percent = int(match.group(1))
    return ZoneLabel(
        substance=SUBSTANCE_CLASSES.index("water"),
        fill_fraction=percent / 100.0,
    )


def _load_cube(
    frames_dir: Path,
    video_name: str,
    n_frames: int,
    crop: tuple[int, int, int, int],
) -> np.ndarray:
    """Собирает куб (h, w, N) из .npy кадров, обрезанный до `crop`.

    Кроп обязателен, а не оптимизация. Полный куб 480x640x3000 в float32
    весит 3.7 ГБ, а `build_input_tensor` держит ещё несколько массивов того
    же размера (контраст, дельта, две ресемплированные копии) — суммарно
    15+ ГБ, что на машине с 8 ГБ уходит в своп и подвешивает процесс.
    Размеченные зоны занимают ~10% кадра, поэтому кроп по их объединяющему
    прямоугольнику (плюс опорная область) снижает пик до сотен мегабайт.

    Args:
        frames_dir: директория с файлами ``{video_name}_frame_NNNNN.npy``.
        video_name: имя записи, например ``"water1"``.
        n_frames: ожидаемое число кадров (для проверки полноты).
        crop: ``(y1, y2, x1, x2)`` — окно, которое остаётся от кадра.

    Returns:
        (y2-y1, x2-x1, N) float32.
    """
    paths = sorted(
        frames_dir.glob(f"{video_name}_frame_*.npy"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if len(paths) != n_frames:
        raise ValueError(
            f"{video_name}: ожидалось {n_frames} кадров, найдено {len(paths)} в {frames_dir}"
        )

    y1, y2, x1, x2 = crop
    cube = np.empty((y2 - y1, x2 - x1, n_frames), dtype=np.float32)
    for i, path in enumerate(paths):
        # mmap: кадр не копируется в память целиком, из него берётся только окно.
        frame = np.load(path, mmap_mode="r")
        cube[:, :, i] = frame[y1:y2, x1:x2]
    return cube


def _union_crop(
    cells: list[dict],
    reference_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin: int = 8,
) -> tuple[int, int, int, int]:
    """Наименьшее окно кадра, покрывающее все зоны и опорную область.

    Args:
        cells: записи разметки с ``bbox_xyxy``.
        reference_bbox: ``(x1, y1, x2, y2)`` опорной бездефектной области.
        image_shape: ``(H, W)`` исходного кадра.
        margin: запас по краям, px — нужен, чтобы `SpatialBlock` с
            рецептивным полем 15 px видел контекст вокруг крайних зон.

    Returns:
        ``(y1, y2, x1, x2)`` в координатах исходного кадра.

    Raises:
        ValueError: если хотя бы одна зона окажется ближе к границе кропа,
            чем половина рецептивного поля `SpatialBlock` (`MIN_RF_MARGIN_PX`).
            Иначе нулевой паддинг свёртки у края кропа искажал бы признаки
            пикселей зоны, а не только фона вокруг неё — риск, для этого
            набора данных проверенный численно, но не гарантированный при
            других margin/зонах без этой проверки.
    """
    height, width = image_shape
    xs: list[int] = [reference_bbox[0], reference_bbox[2]]
    ys: list[int] = [reference_bbox[1], reference_bbox[3]]
    for cell in cells:
        bx1, by1, bx2, by2 = cell["bbox_xyxy"]
        xs += [bx1, bx2]
        ys += [by1, by2]

    crop = (
        max(0, min(ys) - margin),
        min(height, max(ys) + margin),
        max(0, min(xs) - margin),
        min(width, max(xs) + margin),
    )
    _check_receptive_field_margin(cells, crop)
    return crop


#: Половина рецептивного поля `SpatialBlock` (RF=15px, ARCHITECTURE.md
#: раздел 5.2) — минимальный запас от края зоны до края кропа, при котором
#: свёрточный паддинг ещё не достаёт до пикселей зоны.
MIN_RF_MARGIN_PX: int = 7


def _check_receptive_field_margin(
    cells: list[dict], crop: tuple[int, int, int, int]
) -> None:
    """Требует, чтобы каждая зона отстояла от края кропа минимум на RF/2."""
    y1, y2, x1, x2 = crop
    for cell in cells:
        bx1, by1, bx2, by2 = cell["bbox_xyxy"]
        margin = min(bx1 - x1, x2 - bx2, by1 - y1, y2 - by2)
        if margin < MIN_RF_MARGIN_PX:
            raise ValueError(
                f"Зона {cell['class_name']!r} отстоит от края кропа на {margin}px "
                f"< {MIN_RF_MARGIN_PX}px — нулевой паддинг SpatialBlock исказит "
                f"её пиксели. Увеличьте margin в _union_crop."
            )


def _erode_bbox(bbox: tuple[int, int, int, int], erode_px: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Сжимает bbox внутрь на erode_px, ограничивая размером кадра."""
    x1, y1, x2, y2 = bbox
    h, w = shape
    return (
        max(0, x1 + erode_px),
        max(0, y1 + erode_px),
        min(w, x2 - erode_px),
        min(h, y2 - erode_px),
    )


def build_cell_index(
    cells: list[dict], image_shape: tuple[int, int], erode_px: int = ERODE_PX
) -> tuple[np.ndarray, dict[int, str]]:
    """Строит растровую карту id зон из разметки bbox.

    Args:
        cells: список записей ``metadata['cells']`` (bbox_xyxy, class_name).
        image_shape: (H, W) кадра.
        erode_px: эрозия каждого bbox внутрь.

    Returns:
        ``(cell_index (H, W) int64, id_to_class_name)`` — 0 зарезервирован
        под фон/игнор, зоны нумеруются с 1 в порядке `cells`.
    """
    cell_index = np.zeros(image_shape, dtype=np.int64)
    id_to_class: dict[int, str] = {}
    for i, cell in enumerate(cells, start=1):
        x1, y1, x2, y2 = _erode_bbox(tuple(cell["bbox_xyxy"]), erode_px, image_shape)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Эрозия {erode_px}px схлопнула зону {cell['class_name']}")
        cell_index[y1:y2, x1:x2] = i
        id_to_class[i] = cell["class_name"]
    return cell_index, id_to_class


class RealZoneDataset(Dataset):
    """Один пример на объект: полный кадр + карта зон + метки всех зон сразу.

    В отличие от `SyntheticCurveDataset` (одна ячейка на пример), здесь
    `n_cells` равно числу зон объекта (6), потому что реальная разметка не
    делит зону на отдельные соты. `CellPooling` в `HoneycombNet` агрегирует
    пиксели каждой зоны в один вектор — ровно та же операция, что и на
    синтетике, просто с другим числом ячеек на пример.
    """

    def __init__(
        self,
        data_root: Path | str,
        video_names: list[str],
        n_time_points: int = 64,
        erode_px: int = ERODE_PX,
    ) -> None:
        """Загружает и предобрабатывает перечисленные записи целиком.

        Args:
            data_root: директория с `honeycomb/metadata/videos/*.json` и
                `<video>/honeycomb/images/train/*.npy` (структура архива).
            video_names: список записей, например ``["water1", "water4"]``.
            n_time_points: длина временной сетки после ресемплинга.
            erode_px: эрозия bbox зоны при построении `cell_index`.
        """
        self.data_root = Path(data_root)
        self.n_time_points = n_time_points
        self.examples: list[dict] = []

        for name in video_names:
            self.examples.append(self._load_one(name, erode_px))

    def _load_one(self, video_name: str, erode_px: int) -> dict:
        meta_path = self.data_root / "honeycomb" / "metadata" / "videos" / f"{video_name}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        h, w = meta["image_shape"]
        crop = _union_crop(meta["cells"], REFERENCE_BBOX, (h, w))
        crop_y1, crop_y2, crop_x1, crop_x2 = crop

        frames_dir = self._resolve_frames_dir(video_name)
        cube = _load_cube(frames_dir, video_name, meta["frame_count"], crop)

        crop_shape = (crop_y2 - crop_y1, crop_x2 - crop_x1)
        if cube.shape[:2] != crop_shape:
            raise ValueError(f"{video_name}: shape куба {cube.shape[:2]} != окна кропа {crop_shape}")

        # Разметка задана в координатах исходного кадра — сдвигаем в кроп.
        shifted_cells = [
            {
                **cell,
                "bbox_xyxy": [
                    cell["bbox_xyxy"][0] - crop_x1,
                    cell["bbox_xyxy"][1] - crop_y1,
                    cell["bbox_xyxy"][2] - crop_x1,
                    cell["bbox_xyxy"][3] - crop_y1,
                ],
            }
            for cell in meta["cells"]
        ]

        ref_mask = np.zeros(crop_shape, dtype=bool)
        ref_x1, ref_y1, ref_x2, ref_y2 = REFERENCE_BBOX
        ref_mask[ref_y1 - crop_y1 : ref_y2 - crop_y1, ref_x1 - crop_x1 : ref_x2 - crop_x1] = True

        tensor, t_grid = build_input_tensor(cube, ref_mask, fps=FPS, n_points=self.n_time_points)
        del cube  # 372 МБ; освобождаем до того, как поднимется следующая запись

        cell_index, id_to_class = build_cell_index(shifted_cells, crop_shape, erode_px)
        n_cells = len(meta["cells"])

        substances = np.zeros(n_cells, dtype=np.int64)
        grades = np.full(n_cells, IGNORE_GRADE, dtype=np.int64)
        for cell_id, class_name in id_to_class.items():
            label = zone_name_to_label(class_name)
            substances[cell_id - 1] = label.substance
            grades[cell_id - 1] = fill_to_grade(label.fill_fraction)

        return {
            "video_name": video_name,
            "x": torch.from_numpy(tensor),
            "cell_index": torch.from_numpy(cell_index),
            "n_cells": n_cells,
            "substance": torch.from_numpy(substances),
            "water_grade": torch.from_numpy(grades),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]

    def _resolve_frames_dir(self, video_name: str) -> Path:
        """Находит директорию с кадрами по полному или сокращённому имени.

        Разные записи в этом наборе распаковывались под разными именами
        (``water1`` -> сокращено до ``w1`` при первичном извлечении архива,
        ``water2``/``water4`` сохранили полное имя) — оба варианта валидны.
        """
        candidates = [video_name, f"w{video_name[len('water'):]}" if video_name.startswith("water") else None]
        for candidate in candidates:
            if candidate is None:
                continue
            path = self.data_root / candidate / "honeycomb" / "images" / "train"
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"Не найдена директория с кадрами для {video_name!r} среди {candidates} в {self.data_root}"
        )
