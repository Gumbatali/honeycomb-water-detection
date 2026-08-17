import random

import cv2
import numpy as np

try:
    import albumentations as A
except ModuleNotFoundError:
    class _ImageOnlyTransform:
        def __init__(self, p=0.5):
            self.p = p
            self.py_random = random.Random()

        @property
        def targets_as_params(self):
            return []

        def get_params_dependent_on_data(self, params, data):
            return {}

        def apply(self, image, **params):
            return image

        def __call__(self, **data):
            if self.py_random.random() >= self.p:
                return data

            params = {}
            if self.targets_as_params:
                params = self.get_params_dependent_on_data({}, data) or {}

            data = dict(data)
            data["image"] = self.apply(image=data["image"], **params)
            return data

    class _Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms)

        def __call__(self, **data):
            for transform in self.transforms:
                data = transform(**data)
            return data

    class _AlbumentationsShim:
        ImageOnlyTransform = _ImageOnlyTransform
        Compose = _Compose

    A = _AlbumentationsShim()


def _fit_to_tile_grid(array, tile_size):
    h, w = array.shape[:2]
    h -= h % tile_size
    w -= w % tile_size
    if h <= 0 or w <= 0:
        raise ValueError("tile_size is larger than the input image")
    return array[:h, :w]


def _boundary_role(top, right, bottom, left):
    role = tuple(name for flag, name in (
        (top, "top"),
        (right, "right"),
        (bottom, "bottom"),
        (left, "left"),
    ) if flag)
    return role or ("interior",)


def build_tiles(image, tile_size=10):
    image = _fit_to_tile_grid(image, tile_size)
    h, w = image.shape[:2]
    grid_h = h // tile_size
    grid_w = w // tile_size

    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            role = _boundary_role(
                y == 0,
                x + tile_size >= w,
                y + tile_size >= h,
                x == 0,
            )
            tiles.append({
                "tile": tile.copy(),
                "role": role,
                "grid_pos": (y // tile_size, x // tile_size),
                "grid_size": (grid_h, grid_w),
            })
    return tiles


def extract_layout(candidate_mask, tile_size=10):
    if candidate_mask.ndim == 3:
        candidate_mask = cv2.cvtColor(candidate_mask, cv2.COLOR_BGR2GRAY)
    candidate_mask = _fit_to_tile_grid(candidate_mask, tile_size)
    h, w = candidate_mask.shape

    layout = []
    for y in range(0, h, tile_size):
        row = []
        for x in range(0, w, tile_size):
            block = candidate_mask[y:y + tile_size, x:x + tile_size]
            row.append(bool(block.any()))
        layout.append(row)
    return layout


def layout_signature(layout):
    return tuple(tuple(row) for row in layout)


def required_boundary(layout):
    h = len(layout)
    w = len(layout[0])
    req = set()

    for y in range(h):
        for x in range(w):
            if not layout[y][x]:
                continue
            if y == 0 or not layout[y - 1][x]:
                req.add(("top", y, x))
            if x == w - 1 or not layout[y][x + 1]:
                req.add(("right", y, x))
            if y == h - 1 or not layout[y + 1][x]:
                req.add(("bottom", y, x))
            if x == 0 or not layout[y][x - 1]:
                req.add(("left", y, x))
    return req


def _role_fallbacks(role):
    if role == ("interior",):
        return [role]

    fallbacks = [role]
    if len(role) > 1:
        fallbacks.extend((edge,) for edge in role)
    fallbacks.append(("interior",))
    return fallbacks


def place_tiles(source_image, target_layout, tile_size=10, rng=None):
    rng = rng or random.Random()
    tiles = build_tiles(source_image, tile_size=tile_size)
    pools = {}
    all_tiles = []
    for item in tiles:
        pools.setdefault(item["role"], []).append(item["tile"])
        all_tiles.append(item["tile"])

    h = len(target_layout)
    w = len(target_layout[0])

    out = np.zeros((h * tile_size, w * tile_size) + source_image.shape[2:], dtype=source_image.dtype)

    occupied = [(y, x) for y in range(h) for x in range(w) if target_layout[y][x]]
    if not occupied:
        return out

    # Наивное сопоставление:
    # берем тайл с той же ролью: угол, край или внутренняя часть.
    for y, x in occupied:
        role = _boundary_role(
            y == 0,
            x == w - 1,
            y == h - 1,
            x == 0,
        )

        found = None
        for candidate_role in _role_fallbacks(role):
            bucket = pools.get(candidate_role)
            if bucket:
                found = rng.choice(bucket)
                break

        if found is None:
            found = rng.choice(all_tiles)

        out[y * tile_size:(y + 1) * tile_size, x * tile_size:(x + 1) * tile_size] = found

    return out


def build_tile_plan(source_image, target_layout, tile_size=10, rng=None):
    rng = rng or random.Random()
    tiles = build_tiles(source_image, tile_size=tile_size)
    pools = {}
    all_tiles = []
    for item in tiles:
        pools.setdefault(item["role"], []).append(item)
        all_tiles.append(item)

    h = len(target_layout)
    w = len(target_layout[0])
    occupied = [(y, x) for y in range(h) for x in range(w) if target_layout[y][x]]

    plan = []
    for y, x in occupied:
        role = _boundary_role(
            y == 0,
            x == w - 1,
            y == h - 1,
            x == 0,
        )

        found = None
        for candidate_role in _role_fallbacks(role):
            bucket = pools.get(candidate_role)
            if bucket:
                found = rng.choice(bucket)
                break

        if found is None:
            found = rng.choice(all_tiles)

        plan.append({
            "target_pos": (y, x),
            "source_pos": found["grid_pos"],
        })

    return plan


def apply_tile_plan(source_image, target_layout, plan, tile_size=10):
    source_image = _fit_to_tile_grid(source_image, tile_size)
    h = len(target_layout)
    w = len(target_layout[0])
    out = np.zeros((h * tile_size, w * tile_size) + source_image.shape[2:], dtype=source_image.dtype)

    for item in plan:
        dst_y, dst_x = item["target_pos"]
        src_y, src_x = item["source_pos"]
        out[
            dst_y * tile_size:(dst_y + 1) * tile_size,
            dst_x * tile_size:(dst_x + 1) * tile_size,
        ] = source_image[
            src_y * tile_size:(src_y + 1) * tile_size,
            src_x * tile_size:(src_x + 1) * tile_size,
        ]

    return out


class ConstrainedTileMix(A.ImageOnlyTransform):
    def __init__(self, tile_size=10, p=0.5):
        super().__init__(p=p)
        self.tile_size = tile_size

    @property
    def targets_as_params(self):
        return ["tile_mix_metadata"]

    def get_params_dependent_on_data(self, params, data):
        candidates = data["tile_mix_metadata"]
        idx = self.py_random.randrange(len(candidates))
        candidate = candidates[idx]
        return {
            "source_image": candidate["image"],
            "layout_mask": candidate.get("layout_mask"),
        }

    def get_params_dependent_on_targets(self, params):
        candidates = params["tile_mix_metadata"]
        idx = self.py_random.randrange(len(candidates))
        candidate = candidates[idx]
        return {
            "source_image": candidate["image"],
            "layout_mask": candidate.get("layout_mask"),
        }

    def apply(self, image, source_image=None, layout_mask=None, **params):
        if source_image is None:
            return image

        if layout_mask is None:
            h, w = image.shape[:2]
            target_layout = [[True for _ in range(w // self.tile_size)] for _ in range(h // self.tile_size)]
        else:
            target_layout = extract_layout(layout_mask, tile_size=self.tile_size)

        return place_tiles(source_image, target_layout, tile_size=self.tile_size, rng=self.py_random)


class FixedConstrainedTileMix(ConstrainedTileMix):
    def __init__(self, tile_size=10, p=0.5, seed=None):
        super().__init__(tile_size=tile_size, p=p)
        self.plan_rng = random.Random(seed)
        self._plan_key = None
        self._target_layout = None
        self._tile_plan = None

    def apply(self, image, source_image=None, layout_mask=None, **params):
        if source_image is None:
            return image

        if layout_mask is None:
            h, w = image.shape[:2]
            target_layout = [[True for _ in range(w // self.tile_size)] for _ in range(h // self.tile_size)]
        else:
            target_layout = extract_layout(layout_mask, tile_size=self.tile_size)

        key = (
            layout_signature(target_layout),
            source_image.shape[:2],
            source_image.shape[2:],
        )
        if self._tile_plan is None or key != self._plan_key:
            self._target_layout = target_layout
            self._tile_plan = build_tile_plan(
                source_image,
                target_layout,
                tile_size=self.tile_size,
                rng=self.plan_rng,
            )
            self._plan_key = key

        return apply_tile_plan(
            source_image,
            self._target_layout,
            self._tile_plan,
            tile_size=self.tile_size,
        )
