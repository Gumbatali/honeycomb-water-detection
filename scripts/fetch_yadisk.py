"""
Скачивание публичной папки Яндекс.Диска на сервер (браузер не нужен).

    python scripts/fetch_yadisk.py "https://disk.yandex.ru/d/XXXX" --out D:\data
    python scripts/fetch_yadisk.py "https://disk.yandex.ru/d/XXXX" --out D:\data --list

Докачка поддерживается: при повторном запуске уже скачанные куски не тянутся заново.
Файлы по 3 ГБ — это надолго, лучше запускать в screen/tmux или как отдельный процесс.

ВАЖНО: скрипт написан по публичному API Яндекс.Диска, но на конкретной ссылке
не проверялся. Сначала запустите с --list и убедитесь, что структура читается.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://cloud-api.yandex.net/v1/disk/public/resources"
CHUNK = 1 << 20


def list_dir(public_key: str, path: str = "/", limit: int = 200) -> list[dict]:
    """Рекурсивный обход публичной папки. Возвращает список файлов."""
    items, offset = [], 0
    while True:
        r = requests.get(API, params={"public_key": public_key, "path": path,
                                      "limit": limit, "offset": offset}, timeout=60)
        r.raise_for_status()
        emb = r.json().get("_embedded")
        if emb is None:
            return [r.json()]
        batch = emb.get("items", [])
        for it in batch:
            if it["type"] == "dir":
                items += list_dir(public_key, it["path"], limit)
            else:
                items.append(it)
        offset += len(batch)
        if len(batch) < limit or offset >= emb.get("total", 0):
            break
    return items


def download(public_key: str, path: str, dst: Path, retries: int = 5) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            meta = requests.get(f"{API}/download",
                                params={"public_key": public_key, "path": path},
                                timeout=60)
            meta.raise_for_status()
            href = meta.json()["href"]

            with requests.get(href, headers=headers, stream=True, timeout=300) as r:
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                done = have
                t0 = time.time()
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            sp = done / max(time.time() - t0, 1e-6) / 1e6
                            pct = 100 * done / total
                            print(f"\r  {dst.name}: {pct:5.1f}%  "
                                  f"{done/1e9:.2f}/{total/1e9:.2f} ГБ  {sp:5.1f} МБ/с",
                                  end="", flush=True)
            print()
            part.replace(dst)
            return dst
        except Exception as e:
            print(f"\n  попытка {attempt}/{retries} не удалась: {type(e).__name__}: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_key", help="публичная ссылка на папку или файл")
    ap.add_argument("--out", default="data")
    ap.add_argument("--list", action="store_true", help="только показать содержимое")
    ap.add_argument("--filter", default="", help="скачивать только совпадающие имена")
    a = ap.parse_args()

    out = Path(a.out)
    files = list_dir(a.public_key)
    files = [f for f in files if a.filter.lower() in f["name"].lower()]
    total = sum(f.get("size", 0) for f in files)
    print(f"файлов: {len(files)}, суммарно {total/1e9:.2f} ГБ\n")
    for f in files:
        print(f"  {f.get('size',0)/1e9:8.2f} ГБ  {f['path']}")
    if a.list:
        return

    print()
    for f in files:
        rel = f["path"].lstrip("/")
        dst = out / rel
        if dst.exists() and dst.stat().st_size == f.get("size", -1):
            print(f"  уже есть: {dst.name}")
            continue
        download(a.public_key, f["path"], dst)


if __name__ == "__main__":
    sys.exit(main())
