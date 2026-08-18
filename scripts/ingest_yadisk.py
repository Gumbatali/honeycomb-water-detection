"""Загрузка данных с Яндекс.Диска с немедленным сжатием в карты признаков.

Каждый файл скачивается, обрабатывается и удаляется — на диске никогда
не лежит больше одного сырого образца. Итоговый датасет занимает
единицы мегабайт вместо десятков гигабайт.

Usage:
    python scripts/ingest_yadisk.py <public_url> [--limit N] [--out data/processed]
"""
from __future__ import annotations

import argparse
import json
import sys

import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.dataset import process_7z, process_mat  # noqa: E402

API = "https://cloud-api.yandex.net/v1/disk/public/resources"
CHUNK = 1 << 20


def list_public(public_key: str, limit: int = 100) -> list[dict]:
    """Возвращает список файлов публичной папки."""
    url = f"{API}?public_key={urllib.parse.quote(public_key, safe='')}&limit={limit}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    return payload.get("_embedded", {}).get("items", [])


def download_url(public_key: str, path: str) -> str:
    """Получает временную прямую ссылку на файл внутри публичной папки."""
    url = (
        f"{API}/download?public_key={urllib.parse.quote(public_key, safe='')}"
        f"&path={urllib.parse.quote(path, safe='')}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["href"]


def download(
    href: str,
    target: Path,
    expected_size: int = 0,
    attempts: int = 6,
    timeout: int = 300,
) -> Path:
    """Скачивает файл потоком с докачкой после обрыва.

    Канал Яндекс.Диска отдаёт ~0.5 МБ/с и периодически рвёт соединение
    на файлах в сотни мегабайт, поэтому докачка по Range обязательна.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        done = target.stat().st_size if target.exists() else 0
        if expected_size and done >= expected_size:
            print()
            return target

        request = urllib.request.Request(href)
        if done:
            request.add_header("Range", f"bytes={done}-")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                mode = "ab" if done and response.status == 206 else "wb"
                if mode == "wb":
                    done = 0
                with open(target, mode) as out:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if expected_size:
                            pct = 100 * done / expected_size
                            print(
                                f"\r    {done/1e6:7.0f}/{expected_size/1e6:.0f} MB ({pct:4.1f}%)",
                                end="",
                                flush=True,
                            )
            print()
            return target
        except Exception as error:  # noqa: BLE001 - сеть рвётся, продолжаем докачку
            if attempt == attempts:
                raise
            print(f"\n    обрыв ({error}); докачка {attempt}/{attempts}...", flush=True)

    return target


def ingest(
    public_key: str,
    out_dir: Path,
    limit: int,
    staging_dir: Path,
    skip_existing: bool = True,
) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    items = list_public(public_key, limit=200)
    targets = [i for i in items if i["name"].lower().endswith((".mat", ".7z"))][:limit]
    print(f"Найдено {len(targets)} файлов для обработки\n")

    for index, item in enumerate(targets, 1):
        name = item["name"]
        stem = Path(name).stem
        result = out_dir / f"{stem}.npz"
        if skip_existing and result.exists():
            print(f"[{index}/{len(targets)}] {name[:55]} — уже обработан, пропуск")
            continue

        print(
            f"[{index}/{len(targets)}] {name[:55]}  ({item.get('size',0)/1e6:.0f} MB)",
            flush=True,
        )
        # Стейджинг вне временной папки: частично скачанный файл переживает
        # обрыв и повторный запуск скрипта, докачиваясь по Range.
        raw = staging_dir / name
        try:
            download(download_url(public_key, item["path"]), raw, item.get("size", 0))
            processed = (
                process_7z(raw, out_dir)
                if name.lower().endswith(".7z")
                else process_mat(raw, out_dir)
            )
            print(
                f"    -> {processed.name} ({processed.stat().st_size/1e6:.2f} MB)\n",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - продолжаем остальные файлы
            print(f"    ОШИБКА: {error}\n", flush=True)
        finally:
            # Сырьё удаляем сразу: 614 МБ на образец не должны копиться.
            raw.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_url", help="Публичная ссылка на папку Яндекс.Диска")
    parser.add_argument("--limit", type=int, default=100, help="Максимум файлов")
    parser.add_argument("--out", default="data/processed", help="Каталог для .npz")
    parser.add_argument(
        "--staging", default="data/raw/_staging", help="Каталог для временных загрузок"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ingest(args.public_url, out_dir, args.limit, Path(args.staging))


if __name__ == "__main__":
    main()
