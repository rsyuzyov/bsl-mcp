"""Контекст приложения и управление состоянием ИБ."""
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from typing import Optional

from .config_manager import IBConfig, ConfigManager

from .config import IndexingStatus
from .logger import get_logger
from .utils import check_index_lock

logger = get_logger("app.context")

@dataclass
class IBContext:
    """Контекст одной информационной базы."""
    config: IBConfig
    engine: 'IndexEngine'
    searcher: 'HybridSearch'
    status: IndexingStatus = field(default_factory=IndexingStatus)
    error: Optional[str] = None  # Сообщение об ошибке (если есть)
    locking_pid: Optional[int] = None  # PID процесса-блокировщика
    
    # Управление фоновыми задачами
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: threading.Thread | None = None
    
    @property
    def is_error(self) -> bool:
        """Контекст в состоянии ошибки?"""
        return self.error is not None

    @property
    def model_error(self) -> Optional[str]:
        """Ошибка загрузки модели (если есть)."""
        from .model_manager import ModelManager
        model_mgr = ModelManager()
        model_info = model_mgr.get_model(self.config.embedding_model, self.config.embedding_device)
        return model_info.error

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
            # Ждём до 5 секунд — индексация может быть в процессе flush
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning(f"Поток обслуживания {self.config.name} не остановился за 5 сек")

    def _maintenance_loop(self):
        """Цикл проверки изменений."""
        log = get_logger(f"ib.{self.config.name}")
        while not self.stop_event.is_set():
            # Проверяем что индексация не запущена (в т.ч. через API)
            if not self.status.running:
                try:
                    # Проверка изменений
                    has_changes, added, changed, deleted = self.engine.quick_check_changes()
                    
                    # Проверка выполнена
                    self.status.initial_check_pending = False

                    # Ещё раз проверяем — за время quick_check могли запустить индексацию через API
                    if has_changes and not self.status.running and not self.stop_event.is_set():
                        log.info(f"Обнаружены изменения, запуск индексации...")
                        self.engine.incremental_reindex(self.status, stop_event=self.stop_event)
                except Exception as e:
                    log.error(f"Ошибка в цикле обслуживания: {e}", exc_info=True)
            
            # Ждем 5 минут или сигнала остановки
            # После пробуждения от stop_event — проверим is_set в начале цикла
            self.stop_event.wait(300)
            # Если stop_event был set и потом clear (перезапуск индексации) — продолжаем цикл
            # Если stop_event остался set (реальная остановка) — выйдем в начале цикла


@dataclass 
class ErrorIBContext:
    """Контекст ИБ в состоянии ошибки (не инициализирована)."""
    config: IBConfig
    error: str
    locking_pid: Optional[int] = None
    
    @property
    def is_error(self) -> bool:
        return True
    
    @property
    def model_error(self) -> Optional[str]:
        """Для ErrorIBContext всегда None (ошибка уже в error)."""
        return None
    
    @property
    def status(self) -> IndexingStatus:
        """Фиктивный статус для совместимости с UI."""
        return IndexingStatus()


class IBManager:
    """Менеджер всех активных ИБ."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.contexts: Dict[str, IBContext] = {}
        self.error_contexts: Dict[str, ErrorIBContext] = {} # IBs that failed to load
        self.is_initializing: bool = False
        self.current_init_ib: Optional[str] = None # Name of the IB currently being initialized
        self.current_init_stage: str = "" # Detaled initialization stage
        self._init_thread: threading.Thread | None = None # Kept for compatibility with existing async_init if not fully replaced
        self._last_ibs_file = Path(".last_ibs.json")  # Файл со списком ИБ предыдущего запуска

    def initialize(self):
        """Синхронная инициализация всех ИБ из конфига (запускается в потоке)."""
        self.is_initializing = True
        self.current_init_ib = None
        self.current_init_stage = ""
        self.contexts.clear()
        self.error_contexts.clear()
        
        try:
            # Очистка данных удалённых ИБ
            self._cleanup_removed_ibs()
            
            # Сначала проверяем, есть ли конфигурации
            if not self.config_manager.config.ibs:
                logger.info("Нет конфигураций для инициализации")
                return

            logger.info(f"Запуск инициализации {len(self.config_manager.config.ibs)} ИБ...")
            start_total = time.time()
            for ib_conf in self.config_manager.config.ibs:
                self.current_init_ib = ib_conf.name
                self._set_init_stage("Начало загрузки")
                
                logger.info(f"--- Начало инициализации ИБ: {ib_conf.name} ---")
                start_ib = time.time()
                self._init_ib(ib_conf)
                duration_ib = time.time() - start_ib
                logger.info(f"--- Завершено ИБ {ib_conf.name}: {duration_ib:.2f} сек ---")
                
            total_duration = time.time() - start_total
            logger.info(f"Инициализация всех ИБ завершена за {total_duration:.2f} сек")
            
            # Сохраняем список ИБ для следующего запуска
            self._save_last_ibs()
        finally:
            self.current_init_ib = None
            self.current_init_stage = ""
            self.is_initializing = False
            logger.info("Инициализация завершена")

    def initialize_async(self):
        """Асинхронная инициализация в фоновом потоке."""
        if self._init_thread and self._init_thread.is_alive():
            return
        self._init_thread = threading.Thread(target=self.initialize, daemon=True)
        self._init_thread.start()

    def _save_last_ibs(self):
        """Сохранить список ИБ для отслеживания удалённых."""
        import json
        ibs_data = {
            ib.name: ib.index_dir 
            for ib in self.config_manager.config.ibs
        }
        try:
            with open(self._last_ibs_file, "w", encoding="utf-8") as f:
                json.dump(ibs_data, f)
        except Exception as e:
            logger.warning(f"Не удалось сохранить список ИБ: {e}")

    def _cleanup_removed_ibs(self):
        """Удалить index_dir для ИБ, удалённых из конфига."""
        import json
        import shutil
        
        if not self._last_ibs_file.exists():
            return
        
        try:
            with open(self._last_ibs_file, "r", encoding="utf-8") as f:
                last_ibs = json.load(f)
        except Exception:
            return
        
        current_names = {ib.name for ib in self.config_manager.config.ibs}
        
        for name, index_dir in last_ibs.items():
            if name not in current_names:
                index_path = Path(index_dir)
                if index_path.exists() and index_path.is_dir():
                    logger.info(f"ИБ '{name}' удалена из конфига, удаляю каталог: {index_dir}")
                    try:
                        shutil.rmtree(index_path)
                        logger.info(f"Каталог {index_dir} удалён")
                    except Exception as e:
                        logger.error(f"Не удалось удалить каталог {index_dir}: {e}")

    def _set_init_stage(self, stage: str):
        """Установить текущий этап инициализации и записать в лог."""
        self.current_init_stage = stage
        if self.current_init_ib:
            logger.info(f"[{self.current_init_ib}] Этап: {stage}")

    def _log_ib_status(self, ctx: IBContext):
        """Логирование детального статуса ИБ после инициализации."""
        from .model_manager import ModelManager
        
        ib_name = ctx.config.name
        model_name = ctx.config.embedding_model
        device = ctx.config.embedding_device
        
        # Проверяем статус модели
        model_mgr = ModelManager()
        model_info = model_mgr.get_model(model_name, device)
        
        if model_info.error:
            model_status = f"ОШИБКА: {model_info.error}"
        elif model_info.loading:
            model_status = "загружается..."
        else:
            model_status = "OK"
        
        # Количество документов в индексе
        try:
            doc_count = ctx.engine.get_collection_count()
        except Exception:
            doc_count = "?"
        
        logger.info(
            f"[{ib_name}] Статус: модель={model_status}, "
            f"устройство={device.upper()}, документов={doc_count}"
        )

    def _init_ib(self, ib_conf: IBConfig):
        """Инициализация одной ИБ."""
        # Сначала проверяем блокировку папки индекса
        self._set_init_stage("Проверка блокировок (FS Lock)")
        is_locked, pid, lock_message = check_index_lock(ib_conf.index_dir)
        if is_locked:
            logger.error(f"ИБ {ib_conf.name}: папка индекса заблокирована! {lock_message}")
            self.error_contexts[ib_conf.name] = ErrorIBContext(
                config=ib_conf,
                error=lock_message or "Папка индекса заблокирована другим процессом",
                locking_pid=pid
            )
            return
        
        try:
            logger.info(f"Инициализация ИБ: {ib_conf.name}...")
            
            self._set_init_stage("Инициализация IndexEngine (Загрузка индекса...)")
            from .indexer.engine import IndexEngine
            from .search.hybrid import HybridSearch
            
            engine = IndexEngine(
                source_dir=ib_conf.source_dir,
                index_dir=ib_conf.index_dir,
                collection_name=f"code_{ib_conf.name}",
                embedding_model_name=ib_conf.embedding_model,
                embedding_device=ib_conf.embedding_device,
                vector_db_type=ib_conf.vector_db,
                scan_workers=self.config_manager.config.scan_workers,
                embedding_mode=ib_conf.embedding_mode
            )
            
            self._set_init_stage("Инициализация HybridSearch (подготовка поиска)")
            # Searcher будет инициализирован с engine.db
            searcher = HybridSearch(engine.db, collection_name=f"code_{ib_conf.name}", model_name=ib_conf.embedding_model, embedding_device=ib_conf.embedding_device)
            
            ctx = IBContext(config=ib_conf, engine=engine, searcher=searcher)
            self.contexts[ib_conf.name] = ctx
            
            # Запуск обслуживания
            self._set_init_stage("Запуск фонового обслуживания")
            ctx.start_maintenance()
            
            # Логируем детальный статус после инициализации
            self._log_ib_status(ctx)
            
            self._set_init_stage("Готово")
            logger.info(f"ИБ {ib_conf.name} инициализирована")
        except Exception as e:
            error_msg = str(e)
            # Проверяем, не ошибка ли это блокировки от Qdrant
            if "already accessed" in error_msg.lower():
                pid = None  # Не смогли определить
                error_msg = f"Папка индекса заблокирована: {error_msg}"
            logger.error(f"Ошибка инициализации ИБ {ib_conf.name}: {e}", exc_info=True)
            self.error_contexts[ib_conf.name] = ErrorIBContext(
                config=ib_conf,
                error=error_msg,
                locking_pid=pid if 'pid' in dir() else None
            )

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

    def get_context(self, name: str) -> IBContext | ErrorIBContext | None:
        """Получить контекст ИБ (может быть в состоянии ошибки)."""
        if name in self.contexts:
            return self.contexts[name]
        return self.error_contexts.get(name)

    def get_all_contexts(self) -> list[IBContext | ErrorIBContext]:
        """Все контексты, включая ошибочные."""
        result = list(self.contexts.values())
        result.extend(self.error_contexts.values())
        return result
    
    def get_working_contexts(self) -> list[IBContext]:
        """Только рабочие контексты (без ошибок)."""
        return list(self.contexts.values())

    def shutdown(self):
        """Корректное завершение работы."""
        logger.info("Остановка IBManager...")
        self.is_initializing = False
        
        for name in list(self.contexts.keys()):
            try:
                self.remove_ib(name, remove_config=False)
            except Exception as e:
                logger.error(f"Ошибка при остановке {name}: {e}")
        logger.info("IBManager остановлен")
