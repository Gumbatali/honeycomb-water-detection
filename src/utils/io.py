"""Загрузка термограмм из .mat файлов (v5 и v7.3/HDF5) и из .7z без полной распаковки."""
from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Частота регистрации по умолчанию (Гц) — методика контроля из отчёта.
# Нужна, т.к. в части файлов поле FPS записано нулём.
DEFAULT_FPS = 10.0

# 7-Zip: ищем в системе, в т.ч. в поставке Unity (частый случай на macOS без brew).
SEVENZIP_CANDIDATES = (
    "7z",
    "7za",
    "7zz",
    "/opt/homebrew/bin/7z",
    "/usr/local/bin/7z",
    "/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/Helpers/7za",
)


@dataclass(frozen=True)
class Thermogram:
    """Куб термограмм.

    Attributes:
        data: (H, W, N_frames) float32 — температурное поле во времени.
        fps: частота регистрации кадров, Гц.
        name: идентификатор источника (имя файла).
    """

    data: np.ndarray
    fps: float
    name: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def duration_s(self) -> float:
        return self.data.shape[2] / self.fps if self.fps else 0.0


def find_7zip() -> str | None:
    """Возвращает путь к рабочему бинарю 7-Zip или None."""
    for candidate in SEVENZIP_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        try:
            subprocess.run([candidate], capture_output=True, timeout=10)
            return candidate
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    return None


def _load_mat_v5(path: Path) -> tuple[np.ndarray, float]:
    import scipy.io as sio

    raw = sio.loadmat(str(path))
    return raw["data"], float(raw["FPS"][0, 0])


def _load_mat_hdf5(path: Path) -> tuple[np.ndarray, float]:
    """MATLAB v7.3 хранится как HDF5 с транспонированными осями."""
    import h5py

    with h5py.File(path, "r") as handle:
        data = np.array(handle["data"]).T
        fps = float(np.array(handle["FPS"]).ravel()[0])
    return data, fps


def is_hdf5(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(8) == b"\x89HDF\r\n\x1a\n"


def fps_from_filename(name: str) -> float | None:
    """Извлекает частоту кадров из имени файла вида '... 100мс ...'.

    Часть записей имеет FPS=0 в самом .mat, но период кадра указан
    в имени файла ('100мс' -> 10 Гц).
    """
    match = re.search(r"(\d+)\s*мс", name)
    if match:
        period_ms = float(match.group(1))
        if period_ms > 0:
            return 1000.0 / period_ms
    return None


def resolve_fps(stored_fps: float, name: str, default: float = DEFAULT_FPS) -> float:
    """Определяет рабочую частоту кадров: файл -> имя -> значение по умолчанию."""
    if stored_fps and stored_fps > 0:
        return float(stored_fps)
    from_name = fps_from_filename(name)
    return from_name if from_name else default


def load_thermogram(path: str | Path, fps: float | None = None) -> Thermogram:
    """Загружает куб термограмм из .mat (формат v5 или v7.3/HDF5).

    Если поле FPS в файле нулевое (встречается в части записей), частота
    берётся из имени файла или из `DEFAULT_FPS`.

    Внимание: файл целиком читается в память. Для больших кубов
    используйте `stream_frames` или `extract_features_streaming`.
    """
    path = Path(path)
    data, stored_fps = _load_mat_hdf5(path) if is_hdf5(path) else _load_mat_v5(path)
    effective_fps = float(fps) if fps else resolve_fps(stored_fps, path.name)
    return Thermogram(
        data=np.asarray(data, dtype=np.float32), fps=effective_fps, name=path.name
    )


@contextmanager
def extracted_from_7z(archive: str | Path, workdir: str | Path) -> Iterator[Path]:
    """Распаковывает единственный .mat из .7z во временную папку и удаляет после выхода.

    LZMA2 solid-блок не допускает частичного чтения, поэтому архив
    распаковывается целиком, но результат живёт только внутри контекста.
    """
    archive, workdir = Path(archive), Path(workdir)
    binary = find_7zip()
    if binary is None:
        raise RuntimeError("7-Zip не найден: установите p7zip или укажите путь в SEVENZIP_CANDIDATES")

    workdir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        subprocess.run(
            [binary, "x", str(archive), f"-o{workdir}", "-y"],
            check=True,
            capture_output=True,
        )
        extracted = sorted(workdir.rglob("*.mat"))
        if not extracted:
            raise RuntimeError(f"В архиве {archive.name} не найдено .mat файлов")
        yield extracted[0]
    finally:
        for mat in extracted:
            mat.unlink(missing_ok=True)


def stream_frames(path: str | Path, chunk: int = 100) -> Iterator[tuple[int, np.ndarray]]:
    """Читает HDF5-.mat порциями кадров, не загружая куб целиком.

    Yields:
        (start_index, block) где block имеет форму (H, W, n) c n <= chunk.
    """
    import h5py

    path = Path(path)
    if not is_hdf5(path):
        cube = load_thermogram(path).data
        for start in range(0, cube.shape[2], chunk):
            yield start, cube[:, :, start : start + chunk]
        return

    with h5py.File(path, "r") as handle:
        dataset = handle["data"]  # в HDF5 оси лежат как (N, W, H)
        total = dataset.shape[0]
        for start in range(0, total, chunk):
            block = np.array(dataset[start : start + chunk]).T
            yield start, np.asarray(block, dtype=np.float32)
