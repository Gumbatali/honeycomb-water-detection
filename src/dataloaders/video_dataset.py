import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class ThermographyVideoDataset(Dataset):
    """
    Dataset для 3D-видеокубов.
    Читает .npy файлы и переводит их в формат PyTorch (C, T, H, W).
    """
    def __init__(self, data_dir: str, time_downsample: int = 10):
        self.data_dir = Path(data_dir)
        self.file_paths = list(self.data_dir.glob("patch_*.npy"))
        self.time_downsample = time_downsample
        
        if not self.file_paths:
            raise FileNotFoundError(f"В папке {self.data_dir} не найдено .npy файлов.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # Загрузка массива: размерность (H, W, T)
        video_3d = np.load(path)
        
        # Уменьшаем временное разрешение (например, 3000 кадров -> 300 кадров)
        # Это спасет VRAM видеокарты, при этом физика графика сохранится
        video_sampled = video_3d[:, :, ::self.time_downsample]
        
        # Перевод в формат PyTorch: (Channels, Time, Height, Width) -> (1, T, 86, 86)
        # Транспонируем (H, W, T) -> (T, H, W)
        video_tensor = torch.FloatTensor(video_sampled).permute(2, 0, 1)
        video_tensor = video_tensor.unsqueeze(0) # Добавляем канал (1)
        
        # Извлекаем таргет из названия файла (например, patch_water1_target_0.6.npy -> 0.6)
        filename = path.stem
        target_str = filename.split('_target_')[-1]
        target = float(target_str)
        target_tensor = torch.FloatTensor([target])
        
        return video_tensor, target_tensor

def get_video_dataloader(data_dir, batch_size=4, time_downsample=10, shuffle=True):
    """Фабрика для получения DataLoader'а"""
    dataset = ThermographyVideoDataset(data_dir, time_downsample=time_downsample)
    # Используем num_workers для многопоточной загрузки данных с диска
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)