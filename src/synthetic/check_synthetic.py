import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

def get_project_root() -> Path:
    """
    Надежный поиск корня проекта отталкиваясь от текущего расположения файла.
    Скрипт лежит в src/synthetic/, значит корень на 2 уровня выше.
    Если запущен из Jupyter, ищет папку 'data'.
    """
    try:
        current_dir = Path(__file__).resolve().parent
        root = current_dir.parent.parent
        if (root / "data").exists():
            return root
    except NameError:
        pass # Fallback для Jupyter Notebook
        
    root = Path.cwd()
    while not (root / "data").exists() and root != root.parent:
        root = root.parent
    return root

# Определение путей
PROJECT_ROOT = get_project_root()
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

def visualize_synthetic_data(max_files: int = 30):
    """
    Универсальная визуализация синтетических данных ИК-термографии.
    """
    if not SYNTHETIC_DIR.exists():
        print(f"❌ Папка не найдена: {SYNTHETIC_DIR}")
        return

    # Собираем все поддерживаемые файлы
    files = list(SYNTHETIC_DIR.rglob("*.npy")) + list(SYNTHETIC_DIR.rglob("*.mat"))
    
    if not files:
        print("📭 В папке data/synthetic пока нет сгенерированных файлов.")
        return
        
    print(f"🔍 Найдено файлов: {len(files)}. Анализируем первые {max_files}...\n")
    
    for file_path in files[:max_files]:
        # Парсинг .npy
        if file_path.suffix == '.npy':
            data = np.load(file_path)
            target = "Неизвестно (NPY)"
            
        # Парсинг .mat
        elif file_path.suffix == '.mat':
            mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
            data = np.asarray(mat.get("data", []), dtype=np.float32)
            target = mat.get("target", "Неизвестно")
            if not data.any():
                continue
        
        print("-" * 55)
        print(f"📄 Файл: {file_path.name}")
        print(f"🎯 Целевой класс: {target} | Размерность: {data.shape}")
        
        # Отрисовка графиков
        plt.figure(figsize=(10, 4))
        
        # 1D: Температурный профиль (например, из центра дефектной зоны)
        if data.ndim == 1:
            plt.plot(data, color='blue', linewidth=2)
            plt.title(f"1D Синтетика (Таргет: {target})")
            plt.xlabel("Кадры (Время)")
            plt.ylabel("Нормализованная температура")
            plt.grid(True)
            
        # 2D: Синтетический кадр (термограмма)
        elif data.ndim == 2:
            im = plt.imshow(data, cmap='inferno')
            plt.title(f"2D Термограмма (Таргет: {target})")
            plt.axis('off')
            plt.colorbar(im, label="Дельта температур")
            
        # 3D: Синтетический куб (H, W, N_frames)
        elif data.ndim == 3:
            # Выводим профиль остывания центрального пикселя
            center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
            profile = data[center_y, center_x, :]
            plt.plot(profile, color='red', linewidth=2)
            plt.title(f"3D Куб: Динамика центрального пикселя (Таргет: {target})")
            plt.xlabel("Кадры (Время)")
            plt.ylabel("Температура")
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_synthetic_data()