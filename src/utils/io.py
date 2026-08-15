"""Загрузка термограмм из .mat файлов."""
from pathlib import Path

import numpy as np
import scipy.io as sio


def load_thermogram(path: str | Path) -> tuple[np.ndarray, float]:
    """Загружает куб термограмм из .mat.

    Returns:
        data: ndarray (H, W, N_frames), float32
        fps: float
    """
    raw = sio.loadmat(str(path))
    data = raw["data"]
    fps = float(raw["FPS"][0, 0])
    return data, fps
