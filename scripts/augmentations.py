import cv2
import numpy as np


class HoneycombSpatialAugmenter:
  """Пространственная аугментация, сохраняющая физическую геометрию сотовой решетки.

  Поддерживает одиночные кадры (H, W) и временные профили (H, W, N_frames).
  """

  @staticmethod
  def flip(patch: np.ndarray, axis: str = 'horizontal') -> np.ndarray:
    """Зеркальное отражение патча."""
    if axis == 'horizontal':
      return np.fliplr(patch)
    elif axis == 'vertical':
      return np.flipud(patch)
    raise ValueError("Ось должна быть 'horizontal' или 'vertical'")

  @staticmethod
  def rotate_hex(patch: np.ndarray, angle: int) -> np.ndarray:
    """Безопасный поворот патча. Угол строго кратен 60°.

    Используется cv2.BORDER_REFLECT для заполнения возможных краевых пустот без
    резких перепадов.
    """
    if angle % 60 != 0:
      raise ValueError(f'Угол {angle}° недопустим. Разрешены только кратные 60°.')

    h, w = patch.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    if patch.ndim == 2:
      return cv2.warpAffine(
          patch,
          rotation_matrix,
          (w, h),
          flags=cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_REFLECT,
      )

    rotated_patch = np.zeros_like(patch)
    n_frames = patch.shape[2]

    for i in range(n_frames):
      rotated_patch[:, :, i] = cv2.warpAffine(
          patch[:, :, i],
          rotation_matrix,
          (w, h),
          flags=cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_REFLECT,
      )

    return rotated_patch


class HoneycombTemporalAugmenter:
  """Временная аугментация термограмм (H, W, N_frames)."""

  @staticmethod
  def drop_frames(
      thermal_cube: np.ndarray, drop_fraction: float = 0.05
  ) -> np.ndarray:
    """Случайное удаление кадров.

    Имитирует пропуски кадров или плавающий FPS.
    """
    n_frames = thermal_cube.shape[2]
    keep_count = int(n_frames * (1 - drop_fraction))
    indices = np.sort(np.random.choice(n_frames, keep_count, replace=False))
    return thermal_cube[:, :, indices]

  @staticmethod
  def temporal_crop(
      thermal_cube: np.ndarray, crop_frames: int = 50
  ) -> np.ndarray:
    """Обрезка временного ряда с конца (хвоста остывания)."""
    return thermal_cube[:, :, :-crop_frames]


class HoneycombRadiometricAugmenter:
  """Радиометрическая (физическая) аугментация температурных значений."""

  @staticmethod
  def add_sensor_noise(
      thermal_cube: np.ndarray, std: float = 0.05
  ) -> np.ndarray:
    """Добавление гауссовского шума (имитация шума болометрической матрицы)."""
    noise = np.random.normal(
        loc=0.0, scale=std, size=thermal_cube.shape
    ).astype(np.float32)
    return thermal_cube + noise

  @staticmethod
  def scale_heating_amplitude(
      thermal_cube: np.ndarray, variance: float = 0.05
  ) -> np.ndarray:
    """Вариация мощности галогенного нагревателя (+/- variance)."""
    scale = np.random.uniform(1.0 - variance, 1.0 + variance)
    return thermal_cube * scale

class AdvancedHoneycombAugmenter:
    """
    Продвинутые безопасные аугментации для сотовых композитов.
    Учитывают физику процесса и матричную структуру сот.
    """

    @staticmethod
    def tile_dropout(thermal_cube: np.ndarray, tile_size: int = 10, num_tiles: int = 3) -> np.ndarray:
        """
        Случайное зануление (Dropout) квадратных областей.
        Учит сеть не опираться на один конкретный локальный признак дефекта, 
        заставляя "смотреть" на всю зону (например, на все 72 ячейки).
        """
        augmented = thermal_cube.copy()
        h, w = augmented.shape[:2]

        for _ in range(num_tiles):
            y = np.random.randint(0, h - tile_size)
            x = np.random.randint(0, w - tile_size)
            # Зануляем область синхронно во всех кадрах временного ряда
            augmented[y:y+tile_size, x:x+tile_size, :] = 0

        return augmented

    @staticmethod
    def random_shift(thermal_cube: np.ndarray, max_shift: int = 5) -> np.ndarray:
        """
        Аккуратный пространственный сдвиг (в пределах шага соты).
        Используется отражение краев, чтобы не вносить артефакты нулевых границ.
        """
        h, w = thermal_cube.shape[:2]
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)

        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = np.zeros_like(thermal_cube)

        for i in range(thermal_cube.shape[2]):
            shifted[:, :, i] = cv2.warpAffine(
                thermal_cube[:, :, i],
                translation_matrix,
                (w, h),
                borderMode=cv2.BORDER_REFLECT
            )
        return shifted

    @staticmethod
    def mixup(cube_a: np.ndarray, cube_b: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Смешивание двух температурных профилей (MixUp).
        cube_a и cube_b должны быть одного размера (например, вырезанные зоны).
        В идеале применять к образцам с близким % заполнения для синтеза промежуточных значений.
        """
        # Генерируем коэффициент смешивания из Beta-распределения
        lam = np.random.beta(alpha, alpha)
        # Ограничиваем lam, чтобы один образец оставался доминирующим
        lam = max(lam, 1 - lam)

        return (lam * cube_a + (1 - lam) * cube_b).astype(np.float32)

    @staticmethod
    def contrast_scaling(thermal_cube: np.ndarray, variance: float = 0.05) -> np.ndarray:
        """
        Глобальное масштабирование контраста.
        Компенсирует микроколебания в мощности 500 Вт нагревателя или погрешности калибровки.
        """
        scale = np.random.uniform(1.0 - variance, 1.0 + variance)
        return (thermal_cube * scale).astype(np.float32)