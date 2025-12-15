"""Утилиты форматирования."""
from datetime import datetime


def format_time(ts: float) -> str:
    """Форматирует timestamp в HH:MM:SS."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def format_duration(sec: float) -> str:
    """Форматирует секунды в MM:SS или HH:MM:SS."""
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60}:{sec % 60:02d}"
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
