"""MixUp-аугментация для видеоклипов."""

from __future__ import annotations

import tempfile
from pathlib import Path
import cv2

import numpy as np

try:
    import torch
except ImportError:  # PyTorch нужен только для работы с torch.Tensor.
    torch = None


class AlbumentationsVideoClipAug:
    """
    Применяет аугментации ко всем кадрам клипа,
    используя одинаковые случайные параметры для всего клипа.
    """
    
    

class VideoMixUp:
    """
    MixUp для видео.
    Смешивает клипы как взвешенное среднее.
    """
    
    def __init__(self, alpha: float = 0.2):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha должен быть неотрицательным конечным числом")
        self.alpha = float(alpha)


    def sample_lambda(self) -> float:
        if self.alpha <= 0:
            return 1

        lam = np.random.beta(self.alpha, self.alpha)
        return float(lam)


    def __call__(
        self,
        clip1: torch.Tensor | np.ndarray,
        clip2: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if clip1.shape != clip2.shape:
            raise ValueError(
                f"Клипы должны иметь одну и ту же форму: "
                f"получено {clip1.shape} и {clip2.shape}"
            )

        # Смешивание NumPy и PyTorch объектов не даёт предсказуемого результата.
        clip1_is_torch = torch is not None and isinstance(clip1, torch.Tensor)
        clip2_is_torch = torch is not None and isinstance(clip2, torch.Tensor)
        if clip1_is_torch != clip2_is_torch:
            raise TypeError("Оба клипа должны быть либо NumPy-массивами, либо torch.Tensor")
        
        lam = self.sample_lambda()
        
        mixed_clip = lam * clip1 + (1.0 - lam) * clip2

        return mixed_clip

    def mixup_to_memmap(
        self,
        clip1: np.ndarray,
        clip2: np.ndarray,
        output_path: str | Path,
        *,
        chunk_frames: int = 16,
    ) -> np.memmap:
        """Смешать большие клипы, записывая результат блоками на диск.

        Ожидается формат ``H x W x T``. Исходные клипы желательно передавать
        как ``numpy.memmap`` из :func:`load_video_memmap`: тогда в RAM не
        загружается весь 3.5-гигабайтный файл.
        """
        if clip1.shape != clip2.shape:
            raise ValueError(
                f"Клипы должны иметь одну и ту же форму: "
                f"получено {clip1.shape} и {clip2.shape}"
            )
        if not isinstance(chunk_frames, int) or chunk_frames <= 0:
            raise ValueError("chunk_frames должен быть положительным целым числом")
        if clip1.ndim != 3:
            raise ValueError(f"Ожидается клип формата H x W x T, получено {clip1.ndim} измерений")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dtype = np.result_type(clip1.dtype, clip2.dtype, np.float32)
        result = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=dtype, shape=clip1.shape
        )
        lam = self.sample_lambda()

        try:
            for start in range(0, clip1.shape[2], chunk_frames):
                stop = min(start + chunk_frames, clip1.shape[2])
                result[:, :, start:stop] = (
                    lam * clip1[:, :, start:stop]
                    + (1.0 - lam) * clip2[:, :, start:stop]
                )
            result.flush()
        except Exception:
            del result
            output_path.unlink(missing_ok=True)
            raise

        return result

    def mixup_to_video(
        self,
        clip1: np.ndarray,
        clip2: np.ndarray,
        output_path: str | Path,
        *,
        fps: float = 10.0,
        chunk_frames: int = 16,
    ) -> Path:
        """Смешать клипы и записать их в обычный grayscale MP4."""
        if clip1.shape != clip2.shape:
            raise ValueError(
                f"Клипы должны иметь одну и ту же форму: "
                f"получено {clip1.shape} и {clip2.shape}"
            )
        if clip1.ndim != 3:
            raise ValueError(f"Ожидается клип формата H x W x T, получено {clip1.ndim} измерений")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps должен быть положительным конечным числом")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width, frame_count = clip1.shape
        import cv2

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
            (width, height), isColor=False,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось открыть VideoWriter: {output_path}")

        lam = self.sample_lambda()
        try:
            for start in range(0, frame_count, chunk_frames):
                stop = min(start + chunk_frames, frame_count)
                mixed = (
                    lam * clip1[:, :, start:stop]
                    + (1.0 - lam) * clip2[:, :, start:stop]
                )
                for frame in np.moveaxis(mixed, 2, 0):
                    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
                    low, high = np.percentile(frame, (1, 99))
                    if high <= low:
                        frame_u8 = np.zeros(frame.shape, dtype=np.uint8)
                    else:
                        frame_u8 = np.clip(
                            (frame - low) * 255.0 / (high - low), 0, 255
                        ).astype(np.uint8)
                    writer.write(frame_u8)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        finally:
            writer.release()

        return output_path


def load_clip(path: str | Path, temp_dir: str | Path) -> tuple[np.ndarray, float]:
    """Загрузить MAT, NPY или MP4 как клип формата ``H x W x T``.

    Для MP4 создаётся временный ``.npy``-memmap, поэтому декодирование не
    требует выделения RAM под весь ролик.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        from src.utils.analyze_water_cooling import load_video_memmap

        return load_video_memmap(path, "data")

    if suffix == ".npy":
        clip = np.load(path, mmap_mode="r")
        if clip.ndim != 3:
            raise ValueError(f"Ожидается NPY-клип H x W x T, получено {clip.shape}")
        return clip, 0.0

    if suffix != ".mp4":
        raise ValueError(f"Неподдерживаемый формат входа: {suffix or '<без расширения>'}")

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Не удалось определить размер видео: {path}")

    # Для некоторых кодеков frame_count ненадёжен, поэтому при нём читаем
    # последовательно и увеличиваем временный файл при необходимости нельзя.
    # В обычных MP4 OpenCV возвращает корректное количество кадров.
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Не удалось определить количество кадров: {path}")

    temp_path = Path(temp_dir) / f"{path.stem}_{id(path)}.npy"
    clip = np.lib.format.open_memmap(
        temp_path, mode="w+", dtype=np.float32, shape=(height, width, frame_count)
    )
    actual_count = 0
    try:
        while actual_count < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clip[:, :, actual_count] = frame
            actual_count += 1
    finally:
        capture.release()

    if actual_count != frame_count:
        del clip
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Видео {path} содержит {actual_count} кадров вместо ожидаемых {frame_count}"
        )
    clip.flush()
    return clip, (fps if fps > 0 else 0.0)
    

if __name__ == "__main__":
    import argparse

    try:
        from src.utils.analyze_water_cooling import load_video_memmap
    except ModuleNotFoundError:  # Запуск напрямую: python src/augmentation/mixup.py
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.utils.analyze_water_cooling import load_video_memmap

    parser = argparse.ArgumentParser(description="Потоковый MixUp больших MAT-клипов")
    parser.add_argument("clip1", type=Path)
    parser.add_argument("clip2", type=Path)
    parser.add_argument("output", type=Path, help="Путь для выходного .npy или .mp4 файла")
    parser.add_argument(
        "--format", choices=("auto", "npy", "mp4"), default="auto",
        help="Формат результата; auto выбирает его по расширению",
    )
    parser.add_argument("--chunk-frames", type=int, default=16)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mixup-") as temp_dir:
        clip1, fps1 = load_clip(args.clip1, temp_dir)
        clip2, fps2 = load_clip(args.clip2, temp_dir)
        if fps1 and fps2 and fps1 != fps2:
            raise ValueError(f"Частота кадров не совпадает: {fps1} и {fps2}")

        mixup = VideoMixUp()
        output_format = args.format
        if output_format == "auto":
            output_format = args.output.suffix.lower().lstrip(".")
        if output_format == "npy":
            mixup.mixup_to_memmap(clip1, clip2, args.output, chunk_frames=args.chunk_frames)
        elif output_format == "mp4":
            output_fps = fps1 or fps2 or 10.0
            new_height, new_width, _ = clip2.shape
            clip1 = np.stack([
                cv2.resize(clip1[:, :, i], (new_width, new_height))
                for i in range(clip1.shape[2])
            ], axis=2)
            mixup.mixup_to_video(
                clip1, clip2, args.output, fps=output_fps,
                chunk_frames=args.chunk_frames,
            )
        else:
            raise ValueError("Для --format auto расширение output должно быть .npy или .mp4")
        print(f"Результат записан в {args.output}")
    
