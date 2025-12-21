"""Контекст приложения и управление состоянием ИБ."""
import threading
import time
from dataclasses import dataclass, field
from queue import Queue

from .config_manager import IBConfig, ConfigManager
from .indexer.engine import IndexEngine
from .search.hybrid import HybridSearch
from .config import IndexingStatus
from .logger import get_logger

logger = get_logger("app.context")

@dataclass
class IBContext:
    """Контекст одной информационной базы."""
    config: IBConfig
    engine: IndexEngine
    searcher: HybridSearch
    status: IndexingStatus = field(default_factory=IndexingStatus)
    
    # Управление фоновыми задачами
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: threading.Thread | None = None

    def start_maintenance(self):
        """Запуск фонового обслуживания (индексации)."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.worker_thread.start()

    def stop_maintenance(self):
        """Остановка фонового обслуживания."""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def _maintenance_loop(self):
        """Цикл проверки изменений."""
        log = get_logger(f"ib.{self.config.name}")
        while not self.stop_event.is_set():
            if not self.status.running:
                try:
                    # Проверка изменений
                    has_changes, added, changed, deleted = self.engine.quick_check_changes()
                    if has_changes:
                        log.info(f"Обнаружены изменения, запуск индексации...")
                        self.engine.incremental_reindex(self.status)
                except Exception as e:
                    log.error(f"Ошибка в цикле обслуживания: {e}", exc_info=True)
            
            # Ждем 5 минут или сигнала остановки
            if self.stop_event.wait(300):
                break


class IBManager:
    """Менеджер всех активных ИБ."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.contexts: dict[str, IBContext] = {}
        self.is_initializing: bool = False
        self._init_thread: threading.Thread | None = None

    def initialize(self):
        """Синхронная инициализация всех ИБ из конфига."""
        self.is_initializing = True
        try:
            for ib_conf in self.config_manager.config.ibs:
                self._init_ib(ib_conf)
        finally:
            self.is_initializing = False

    def initialize_async(self):
        """Асинхронная инициализация в фоновом потоке."""
        if self._init_thread and self._init_thread.is_alive():
            return
        self._init_thread = threading.Thread(target=self.initialize, daemon=True)
        self._init_thread.start()

    def _init_ib(self, ib_conf: IBConfig):
        """Инициализация одной ИБ."""
        try:
            logger.info(f"Инициализация ИБ: {ib_conf.name}...")
            engine = IndexEngine(
                source_dir=ib_conf.source_dir,
                index_dir=ib_conf.index_dir,
                collection_name=f"code_{ib_conf.name}",
                embedding_model_name=ib_conf.embedding_model,
                embedding_device=ib_conf.embedding_device,
                vector_db_type=ib_conf.vector_db
            )
            # Searcher будет инициализирован с engine.db
            searcher = HybridSearch(engine.db, collection_name=f"code_{ib_conf.name}", model_name=ib_conf.embedding_model, embedding_device=ib_conf.embedding_device)
            
            ctx = IBContext(config=ib_conf, engine=engine, searcher=searcher)
            self.contexts[ib_conf.name] = ctx
            
            # Запуск обслуживания
            ctx.start_maintenance()
            logger.info(f"ИБ {ib_conf.name} инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации ИБ {ib_conf.name}: {e}", exc_info=True)

    def add_ib(self, ib_conf: IBConfig, overwrite: bool = False):
        """Добавление или обновление ИБ."""
        if overwrite and ib_conf.name in self.contexts:
            # Stop existing
            logger.info(f"Остановка ИБ {ib_conf.name} перед обновлением...")
            self.remove_ib(ib_conf.name, remove_config=False)
            
        self.config_manager.add_ib(ib_conf, overwrite=overwrite)
        self._init_ib(ib_conf)

    def remove_ib(self, name: str, remove_config: bool = True, with_data: bool = False):
        """Удаление ИБ."""
        if name in self.contexts:
            ctx = self.contexts[name]
            ctx.stop_maintenance()
            ctx.engine.close()
            del self.contexts[name]
            
            if with_data:
                try:
                    import shutil
                    from pathlib import Path
                    p = Path(ctx.config.index_dir)
                    if p.exists() and p.is_dir():
                        shutil.rmtree(p)
                        logger.info(f"Удалена папка индекса: {p}")
                except Exception as e:
                    logger.error(f"Ошибка удаления данных {name}: {e}")
        
        if remove_config:
            self.config_manager.remove_ib(name)

    def get_context(self, name: str) -> IBContext | None:
        return self.contexts.get(name)

    def get_all_contexts(self) -> list[IBContext]:
        return list(self.contexts.values())
