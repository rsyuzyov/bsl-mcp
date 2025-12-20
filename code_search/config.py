"""Конфигурация и константы."""
from dataclasses import dataclass, field
import time


VECTOR_SIZE = 312
# COLLECTION_NAME, CHUNK_SIZE, etc. kept if needed, but globals moved to instance mgmt
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
# CHECK_INTERVAL = 300  # Moved to config logic, kept here as default if imported?
# Usually better to remove if not used.

@dataclass
class IndexingStatus:
    """Состояние индексации."""
    running: bool = False
    mode: str | None = None  # "full" или "incremental"
    started_at: float | None = None
    files_to_index: int = 0
    files_indexed: int = 0
    chunks_to_index_est: int = 0  # расчётное общее количество (total_bytes / 900)
    chunks_indexed: int = 0  # сколько чанков уже нарезали
    chunks_in_db_start: int = 0  # было в базе на начало индексации
    chunks_in_db: int = 0  # реально записано в Qdrant
    chunks_speed: float = 0
    elapsed: float = 0
    eta: float | None = None
    eta_time: float | None = None
    error: str | None = None
    last_file: str = ""
    status_detail: str = ""

    def reset(self, mode: str):
        """Сброс для новой индексации."""
        self.running = True
        self.mode = mode
        self.started_at = time.time()
        self.files_to_index = 0
        self.files_indexed = 0
        self.chunks_to_index_est = 0
        self.chunks_indexed = 0
        self.chunks_in_db_start = 0
        self.chunks_in_db = 0
        self.chunks_speed = 0
        self.elapsed = 0
        self.eta = None
        self.eta_time = None
        self.error = None
        self.last_file = ""
        self.status_detail = ""

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "mode": self.mode,
            "started_at": self.started_at,
            "files_to_index": self.files_to_index,
            "files_indexed": self.files_indexed,
            "chunks_to_index_est": self.chunks_to_index_est,
            "chunks_indexed": self.chunks_indexed,
            "chunks_in_db_start": self.chunks_in_db_start,
            "chunks_in_db": self.chunks_in_db,
            "chunks_awaiting_insert": self.chunks_indexed - self.chunks_in_db,
            "chunks_speed": self.chunks_speed,
            "elapsed": self.elapsed,
            "eta": self.eta,
            "eta_time": self.eta_time,
            "error": self.error,
            "last_file": self.last_file,
            "status_detail": self.status_detail,
        }

    def format_progress(self, collection_count: int = 0) -> dict:
        """Форматированные данные прогресса для консоли и веба."""
        from .utils import format_time, format_duration
        
        elapsed_sec = time.time() - self.started_at if self.started_at else 0
        pct = round(self.files_indexed / self.files_to_index * 100) if self.files_to_index > 0 else 0
        chunks_speed = self.chunks_in_db / elapsed_sec if elapsed_sec > 0 else 0
        
        return {
            "pct": pct,
            "files_indexed": self.files_indexed,
            "files_to_index": self.files_to_index,
            "collection_count": collection_count,
            "indexed": self.chunks_in_db,
            "chunks_speed": round(chunks_speed, 1),
            "started": format_time(self.started_at) if self.started_at else "?",
            "elapsed": format_duration(elapsed_sec),
            "eta_time": format_time(self.eta_time) if self.eta_time else "...",
            "last_file": self.last_file,
            "status_detail": self.status_detail,
            "mode": self.mode,
            "running": self.running,
            "error": self.error,
        }

    def format_console(self, collection_count: int = 0) -> str:
        """Строка прогресса для консоли."""
        p = self.format_progress(collection_count)
        return f"[{p['pct']}%] {p['files_indexed']}/{p['files_to_index']} | в базе: {p['collection_count']} | проиндексировано: {p['indexed']} | {p['chunks_speed']}/с | начало: {p['started']} | прошло: {p['elapsed']} | конец: {p['eta_time']}"
