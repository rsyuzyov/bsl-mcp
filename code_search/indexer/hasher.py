"""Работа с хешами файлов."""
import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    """MD5 хеш файла."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_hashes(meta_file: Path) -> dict[str, str]:
    """Загрузить сохранённые хеши."""
    if meta_file.exists():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {}


def save_hashes(meta_file: Path, hashes: dict[str, str]):
    """Сохранить хеши."""
    meta_file.write_text(json.dumps(hashes, ensure_ascii=False), encoding="utf-8")
