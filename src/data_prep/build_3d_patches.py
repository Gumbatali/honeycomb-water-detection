import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

def main():
    # Определяем корень проекта
    project_root = Path.cwd()
    while not (project_root / "data").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    # Пути
    input_dir = project_root / "data" / "processed" / "images" / "train"
    output_dir = project_root / "data" / "processed_3d" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Координаты дефектных зон (86x86 пикселей)
    zones_info = {
        'epoxy':     ([210, 70, 295, 155], 1.2),
        'water_100': ([300, 70, 385, 155], 1.0),
        'water_80':  ([390, 70, 475, 155], 0.8),
        'water_60':  ([207, 220, 292, 305], 0.6),
        'water_40':  ([300, 220, 385, 305], 0.4),
        'water_20':  ([393, 220, 478, 305], 0.2)
    }

    print("🔍 Поиск покадровых данных...")
    all_files = list(input_dir.glob("*.npy"))
    if not all_files:
        print(f"❌ Кадры не найдены в {input_dir}")
        return

    # Извлекаем префиксы (например, 'water1', 'water4')
    prefixes = sorted(list(set([f.name.split('_frame_')[0] for f in all_files])))
    print(f"📦 Найдено образцов для сборки: {len(prefixes)} -> {prefixes}")

    for prefix in prefixes:
        frame_files = sorted(input_dir.glob(f"{prefix}_frame_*.npy"))
        if not frame_files:
            continue
            
        print(f"\n🎬 Сборка видео-куба для {prefix} ({len(frame_files)} кадров)...")
        
        # 1. Читаем все кадры образца
        frames_list = [np.load(f) for f in tqdm(frame_files, desc="Загрузка кадров")]
        
        # 2. Собираем в 3D (H, W, Time)
        full_video = np.stack(frames_list, axis=-1)
        
        # 3. Вычитаем фон (первый кадр)
        bg_frame = full_video[..., 0:1]
        delta_video = full_video - bg_frame
        
        # 4. Нарезаем на зоны и сохраняем
        for label, (bbox, target) in zones_info.items():
            x1, y1, x2, y2 = bbox
            patch_3d = delta_video[y1:y2+1, x1:x2+1, :]
            
            # Сохраняем таргет в названии файла для DataLoader-а
            out_filename = f"patch_{prefix}_target_{target}.npy"
            np.save(output_dir / out_filename, patch_3d.astype(np.float32))
            
    print(f"\n✅ Все 3D-патчи оптимизированы для GPU и сохранены в {output_dir}")

if __name__ == "__main__":
    main()