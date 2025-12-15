"""Конфигурация и константы."""
from dataclasses import dataclass, field
import time


VECTOR_SIZE = 384
COLLECTION_NAME = "1c_code"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
CHECK_INTERVAL = 300  # 5 минут


@dataclass
class IndexingStatus:
    """Состояние индексации."""
    running: bool = False
    mode: str | None = None  # "full" или "incremental"
    started_at: float | None = None
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    speed: float = 0
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
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.speed = 0
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
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "total_chunks": self.total_chunks,
            "speed": self.speed,
            "elapsed": self.elapsed,
            "eta": self.eta,
            "eta_time": self.eta_time,
            "error": self.error,
            "last_file": self.last_file,
            "status_detail": self.status_detail,
        }


@dataclass
class ModelState:
    """Состояние модели эмбеддингов."""
    model: object = None
    loading: bool = True
    error: str | None = None


# Глобальные состояния
indexing_status = IndexingStatus()
model_state = ModelState()
