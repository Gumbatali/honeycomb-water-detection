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


def to_u8(frame):
  """Нормализация контраста для отображения термограммы."""
  low, high = np.nanpercentile(frame, [5, 95])
  norm = np.clip((frame - low) / max(high - low, 1e-6), 0.0, 1.0)
  return (norm * 255).astype(np.uint8)


def inspect_and_visualize():
  if not synthetic_dir.exists():
    print(f"Папка с синтетикой не найдена: {synthetic_dir.resolve()}")
    return

  mat_files = sorted(list(synthetic_dir.glob("*.mat")))
  npy_files = sorted(list(synthetic_dir.glob("*.npy")))

  print("=" * 60)
  print(f"ДИРЕКТОРИЯ СИНТЕТИКИ: {synthetic_dir.resolve()}")
  print(f"Найдено .mat файлов: {len(mat_files)}")
  print(f"Найдено .npy файлов: {len(npy_files)}")
  print("=" * 60)

  if not mat_files and not npy_files:
    print("В папке data/synthetic пока нет файлов.")
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

    # Если в файле 1D профили (N_samples, N_frames)
    if data.ndim == 2:
      n_samples, n_frames = data.shape
      fig, ax = plt.subplots(figsize=(10, 4))
      for i in range(min(5, n_samples)):
        ax.plot(data[i], label=f"Сэмпл {i+1}")
      target = mat.get("target", mat.get("labels", "N/A"))
      ax.set_title(
          f"1D Профили остывания: {mat_path.name} (Цель/Метка: {target})"
      )
      ax.set_xlabel("Кадры")
      ax.set_ylabel("Температура / Сигнал")
      ax.legend()
      plt.tight_layout()
      plt.show()

    # Если в файле 3D патч (H, W, N_frames)
    elif data.ndim == 3:
      h, w, n_frames = data.shape
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

      # Отрисовка кадра в момент остывания
      mid_frame = min(50, n_frames - 1)
      ax1.imshow(to_u8(data[:, :, mid_frame]), cmap="inferno")
      ax1.set_title(f"Кадр {mid_frame} ({mat_path.name})")
      ax1.axis("off")

      # Отрисовка центрального профиля остывания
      cy, cx = h // 2, w // 2
      ax2.plot(data[cy, cx, :], color="crimson", lw=1.5)
      target = mat.get("target", "N/A")
      ax2.set_title(f"Профиль центра [{cx}, {cy}] (Target: {target})")
      ax2.set_xlabel("Кадры")
      ax2.set_ylabel("Температура")
      ax2.grid(True)
      plt.tight_layout()
      plt.show()

  # --- 2. Проверка .npy файлов ---
  for npy_path in npy_files[:3]:  # Показываем первые несколько для краткости
    arr = np.load(npy_path)
    print(f"\n[NPY] Файл: {npy_path.name} | shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
  inspect_and_visualize()