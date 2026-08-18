import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# 1. Определение корня проекта
project_root = Path.cwd()
while (
    not (project_root / "data").exists() and project_root != project_root.parent
):
    project_root = project_root.parent

synthetic_dir = project_root / "data" / "synthetic"

def inspect_and_visualize():
    if not synthetic_dir.exists():
        print(f"Папка с синтетикой не найдена: {synthetic_dir.resolve()}")
        return

    # Ищем СТРОГО файлы чистой синтетики, сгенерированные нейросетью CVAE
    mat_files = sorted(list(synthetic_dir.glob("*_cvae.mat")))
    npy_files = sorted(list(synthetic_dir.glob("*_cvae.npy")))

    print("=" * 60)
    print(f"ДИРЕКТОРИЯ СИНТЕТИКИ: {synthetic_dir.resolve()}")
    print(f"Найдено .mat файлов CVAE: {len(mat_files)}")
    print(f"Найдено .npy файлов CVAE: {len(npy_files)}")
    print("=" * 60)

    if not mat_files and not npy_files:
        print("В папке data/synthetic не найдены файлы с суффиксом _cvae.")
        return

    # --- 1. Проверка .mat файлов ---
    for mat_path in mat_files:
        print(f"\n[MAT] Файл: {mat_path.name}")
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        for k, v in mat.items():
            if not k.startswith("__"):
                if hasattr(v, "shape"):
                    print(f"  Ключ '{k}': shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  Ключ '{k}': {v}")

        data = np.asarray(mat.get("data", None), dtype=np.float32)
        if data is None:
            continue

        # Для CVAE мы ожидаем 1D профили остывания (N_samples, N_frames)
        if data.ndim == 2:
            n_samples, n_frames = data.shape
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Считаем среднее и отклонение для красивой визуализации сгенерированного батча
            mean_profile = np.mean(data, axis=0)
            std_profile = np.std(data, axis=0)
            frames = np.arange(n_frames)
            
            # Рисуем "коридор" стандартного отклонения (показывает разнообразие генерации)
            ax.fill_between(frames, mean_profile - std_profile, mean_profile + std_profile, 
                            color="green", alpha=0.3, label="Разброс (±1 std)")
            
            # Рисуем среднюю кривую батча
            ax.plot(frames, mean_profile, label="Средний профиль", color="green", lw=2)
            
            # Добавляем пару индивидуальных сэмплов для наглядности
            ax.plot(frames, data[0], label="Сэмпл 1", color="black", lw=0.5, alpha=0.7, linestyle="--")
            ax.plot(frames, data[-1], label=f"Сэмпл {n_samples}", color="blue", lw=0.5, alpha=0.7, linestyle="--")
            
            target = mat.get("target", mat.get("labels", "N/A"))
            ax.set_title(f"Синтетика CVAE: {mat_path.name} (Запрошенный таргет: {target})")
            ax.set_xlabel("Кадры")
            ax.set_ylabel("Температура / Сигнал")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

    # --- 2. Проверка .npy файлов ---
    print("\n--- Проверка .npy файлов ---")
    for npy_path in npy_files[:3]:  # Показываем первые несколько для краткости
        arr = np.load(npy_path)
        print(f"[NPY] Файл: {npy_path.name} | shape={arr.shape}, dtype={arr.dtype}")

if __name__ == "__main__":
    inspect_and_visualize()