"""Каталогизация всех .mat файлов в data/raw: форма, dtype, FPS.

Usage: python scripts/catalog_data.py data/raw
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_thermogram  # noqa: E402


def main(raw_dir: str) -> None:
    for path in sorted(Path(raw_dir).glob("*.mat")):
        data, fps = load_thermogram(path)
        print(f"{path.name}: shape={data.shape} dtype={data.dtype} fps={fps}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
