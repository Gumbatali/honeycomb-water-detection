#!/usr/bin/env python3
"""Utilities for loading MATLAB ``.mat`` files.

Small files can be loaded with :func:`load_mat`.  For large, uncompressed
MATLAB v5 files, :func:`load_mat_variable` and :func:`load_video_memmap`
return a lazy ``numpy.memmap`` instead of copying the whole array to RAM.

MATLAB v7.3 files are HDF5 containers and are intentionally not handled by
the lazy reader.  Use an HDF5 reader such as ``h5py`` for those files.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MI_MATRIX = 14
MI_SINGLE = 7
MI_DOUBLE = 9
_NUMERIC_DTYPES = {
    MI_SINGLE: np.dtype("<f4"),
    MI_DOUBLE: np.dtype("<f8"),
}


@dataclass(frozen=True)
class MatArray:
    """Location and type of a numeric array in an uncompressed v5 file."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    offset: int


def _element(stream, offset: int, endian: str) -> tuple[int, int, int, int]:
    """Return element type, size, payload offset, and next element offset."""
    stream.seek(offset)
    tag = stream.read(8)
    if len(tag) != 8:
        raise ValueError("Unexpected end of MAT file")
    first, second = struct.unpack(f"{endian}II", tag)
    small_type, small_size = struct.unpack(f"{endian}HH", tag[:4])
    if small_size and small_size <= 4:
        return small_type, small_size, offset + 4, offset + 8
    next_offset = offset + 8 + ((second + 7) // 8) * 8
    return first, second, offset + 8, next_offset


def find_mat_arrays(path: str | Path) -> list[MatArray]:
    """Find numeric single/double arrays in an uncompressed MATLAB v5 file."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(128)
        if len(header) != 128 or header[126:128] not in (b"IM", b"MI"):
            raise ValueError("Expected an uncompressed MATLAB v5 MAT file")
        endian = "<" if header[126:128] == b"IM" else ">"
        file_size = path.stat().st_size
        cursor = 128
        arrays: list[MatArray] = []

        while cursor + 8 <= file_size:
            stream.seek(cursor)
            raw = stream.read(8)
            element_type, element_size = struct.unpack(f"{endian}II", raw)
            next_matrix = cursor + 8 + ((element_size + 7) // 8) * 8
            if element_type != MI_MATRIX:
                cursor = next_matrix
                continue

            inner = cursor + 8
            _, _, _, inner = _element(stream, inner, endian)  # flags
            _, dims_size, dims_offset, inner = _element(stream, inner, endian)
            if dims_size % 4:
                raise ValueError("Invalid dimensions element in MAT file")
            stream.seek(dims_offset)
            shape = tuple(struct.unpack(f"{endian}{dims_size // 4}i", stream.read(dims_size)))
            _, name_size, name_offset, inner = _element(stream, inner, endian)
            stream.seek(name_offset)
            name = stream.read(name_size).decode("ascii", errors="strict")
            data_type, data_size, data_offset, _ = _element(stream, inner, endian)
            dtype = _NUMERIC_DTYPES.get(data_type)
            if dtype is not None and data_size == int(np.prod(shape)) * dtype.itemsize:
                arrays.append(MatArray(name, shape, dtype.newbyteorder(endian), data_offset))
            cursor = next_matrix
    return arrays


def load_mat(path: str | Path, *, variable_names: list[str] | None = None,
             squeeze_me: bool = False) -> dict[str, Any]:
    """Load a MATLAB file with SciPy and return its named variables.

    This is the convenient API for normal-sized files.  MATLAB metadata keys
    beginning with ``__`` are removed from the returned mapping.
    """
    from scipy import io as scipy_io

    raw = scipy_io.loadmat(str(path), variable_names=variable_names, squeeze_me=squeeze_me)
    return {name: value for name, value in raw.items() if not name.startswith("__")}


def load_mat_variable(path: str | Path, key: str, *, mmap: bool = True) -> np.ndarray:
    """Load one numeric variable, lazily when possible.

    ``mmap=True`` requires an uncompressed MATLAB v5 file and returns a
    read-only memmap.  If ``mmap=False``, SciPy loads the selected variable.
    """
    path = Path(path)
    if mmap:
        selected = next((item for item in find_mat_arrays(path) if item.name == key), None)
        if selected is None:
            available = ", ".join(item.name for item in find_mat_arrays(path)) or "none"
            raise ValueError(f"Numeric variable {key!r} was not found; available: {available}")
        return np.memmap(path, mode="r", dtype=selected.dtype, offset=selected.offset,
                         shape=selected.shape, order="F")

    variables = load_mat(path, variable_names=[key])
    if key not in variables:
        raise ValueError(f"Variable {key!r} was not found in {path}")
    value = variables[key]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Variable {key!r} is not a numeric ndarray")
    return value


def load_video_memmap(path: str | Path, key: str = "data") -> tuple[np.memmap, float]:
    """Load a H×W×T video and its optional scalar ``FPS`` metadata."""
    video = load_mat_variable(path, key, mmap=True)
    if video.ndim != 3:
        raise ValueError(f"Variable {key!r} must be 3-D, got shape {video.shape}")
    fps = 0.0
    for item in find_mat_arrays(path):
        if item.name.lower() == "fps" and item.shape == (1, 1):
            fps = float(np.memmap(path, mode="r", dtype=item.dtype, offset=item.offset,
                                  shape=(1,))[0])
            break
    return video, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="MATLAB .mat file")
    parser.add_argument("--key", help="Variable to load; omit to list variables")
    parser.add_argument("--no-mmap", action="store_true", help="Load through SciPy instead of memmap")
    parser.add_argument("--output", type=Path, help="Save the selected variable as .npy")
    args = parser.parse_args()

    if args.key is None:
        for item in find_mat_arrays(args.input):
            print(f"{item.name}: shape={item.shape}, dtype={item.dtype}")
        return

    value = load_mat_variable(args.input, args.key, mmap=not args.no_mmap)
    print(f"{args.key}: shape={value.shape}, dtype={value.dtype}")
    if args.output:
        np.save(args.output, np.asarray(value))


if __name__ == "__main__":
    main()
