"""Работа с хешами файлов."""
import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    """MD5 хеш файла."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_hashes(meta_file: Path) -> dict:
    """Загрузить сохранённые хеши.
    Возвращает dict: {path: {hash, mtime, size}}
    Поддерживает миграцию со старого формата {path: hash}.
    """
    if meta_file.exists():
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            # Migration check
            normalized = {}
            for k, v in data.items():
                if isinstance(v, str):
                    normalized[k] = {"hash": v, "mtime": 0, "size": 0}
                else:
                    normalized[k] = v
            return normalized
        except Exception:
            return {}
    return {}


def save_hashes(meta_file: Path, hashes: dict):
    """Сохранить хеши."""
    meta_file.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")

