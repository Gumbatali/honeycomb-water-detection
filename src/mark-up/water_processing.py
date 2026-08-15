"""Потоковая обработка больших MATLAB v5 thermal-файлов."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np


def _tag(data: bytes, offset: int) -> tuple[int, int, int]:
    first, second = struct.unpack_from("<II", data, offset)
    if first >> 16:
        return first & 0xFFFF, first >> 16, offset + 4
    return first, second, offset + 8


def open_thermal_data(path: str | Path) -> np.memmap:
    """Открывает data лениво, без загрузки всего .mat в оперативную память."""
    path = Path(path)
    with path.open("rb") as file:
        file.seek(128)
        header = file.read(4096)

    matrix_type, _, offset = _tag(header, 0)
    if matrix_type != 14:  # miMATRIX
        raise ValueError("Ожидался несжатый MATLAB v5 файл с матрицей data")

    _, flags_size, flags_payload = _tag(header, offset)  # array flags
    offset = flags_payload + flags_size
    dims_type, dims_size, dims_payload = _tag(header, offset)
    if dims_type != 5 or dims_size % 4:
        raise ValueError("Не удалось прочитать размеры матрицы data")
    dims = struct.unpack_from("<" + "i" * (dims_size // 4), header, dims_payload)
    offset = dims_payload + dims_size
    offset += (-offset) % 8

    _, name_size, name_payload = _tag(header, offset)
    name = header[name_payload:name_payload + name_size].decode("ascii", errors="ignore")
    if name != "data":
        raise ValueError(f"Первой матрицей является {name!r}, а не data")
    offset = name_payload + name_size
    offset += (-offset) % 8

    data_type, data_size, data_payload = _tag(header, offset)
    dtype = {7: np.dtype("<f4"), 9: np.dtype("<f8")}.get(data_type)
    if dtype is None:
        raise ValueError(f"Неподдерживаемый тип MATLAB: {data_type}")
    expected = int(np.prod(dims)) * dtype.itemsize
    if data_size < expected:
        raise ValueError("Файл обрезан: размер data меньше ожидаемого")

    return np.memmap(path, mode="r", dtype=dtype, offset=128 + data_payload,
                     shape=tuple(dims), order="F")


def _frames(count: int, start: int, stop: int | None):
    end = count if stop is None else min(stop, count)
    if not 0 <= start < end:
        raise ValueError(f"Диапазон должен быть внутри 0…{count - 1}")
    return range(start, end)


def render_video_stream(input_path, output_path, fps=25.0,
                        frame_start=0, frame_stop=None):
    import matplotlib.pyplot as plt

    thermal = open_thermal_data(input_path)
    height, width, count = thermal.shape
    frames = _frames(count, frame_start, frame_stop)
    indices = np.linspace(frames.start, frames.stop - 1, min(100, len(frames)), dtype=int)
    sample = np.stack([thermal[:, :, i] for i in indices])
    low, high = np.nanpercentile(sample, [1.0, 99.5]).astype(float)
    scale = max(high - low, 1e-6)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Не удалось открыть кодек mp4v")
    inferno = plt.colormaps["inferno"]
    try:
        for n, i in enumerate(frames, 1):
            frame = np.asarray(thermal[:, :, i], dtype=np.float32)
            normalized = np.clip(np.nan_to_num((frame - low) / scale,
                                                nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            heat = (inferno(normalized)[..., :3] * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
            if n % 100 == 0:
                print(f"Обработано кадров: {n}/{len(frames)}")
    finally:
        writer.release()
    return Path(output_path)


def compare_gamma_correction_stream(input_path, output_path, gamma=3.0,
                                    fps=25.0, frame_start=0, frame_stop=None):
    if gamma <= 0:
        raise ValueError("gamma должен быть больше нуля")
    thermal = open_thermal_data(input_path)
    height, width, count = thermal.shape
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError("Не удалось открыть видео для записи")
    try:
        for i in _frames(count, frame_start, frame_stop):
            frame = np.asarray(thermal[:, :, i], dtype=np.float32)
            normalized = cv2.normalize(frame, None, 0.0, 1.0, cv2.NORM_MINMAX)
            original = (normalized * 255).astype(np.uint8)
            corrected = (np.power(normalized, gamma) * 255).astype(np.uint8)
            combined = np.concatenate((cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)), axis=1)
            writer.write(combined)
    finally:
        writer.release()
    return Path(output_path)


def grad_video_stream(input_path, output_path, fps=25.0, gamma=0.5,
                      frame_start=0, frame_stop=None):
    if gamma <= 0:
        raise ValueError("gamma должен быть больше нуля")
    thermal = open_thermal_data(input_path)
    height, width, count = thermal.shape
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Не удалось открыть видео для записи")
    try:
        for i in _frames(count, frame_start, frame_stop):
            frame = np.asarray(thermal[:, :, i], dtype=np.float32)
            sobelx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.normalize(cv2.magnitude(sobelx, sobely), None, 0.0, 1.0,
                                 cv2.NORM_MINMAX)
            gray = (np.power(grad, gamma) * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    return Path(output_path)
