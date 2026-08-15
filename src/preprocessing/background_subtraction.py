"""Вычитание фонового кадра — baseline-предобработка термограмм."""
import numpy as np


def subtract_background(data: np.ndarray, background_frame_idx: int = 0) -> np.ndarray:
    """Вычитает фоновый (первый) кадр из каждого кадра куба термограмм.

    Args:
        data: (H, W, N_frames)
        background_frame_idx: индекс кадра, принимаемого за фон

    Returns:
        (H, W, N_frames) с вычтенным фоном
    """
    background = data[:, :, background_frame_idx : background_frame_idx + 1]
    return data - background
