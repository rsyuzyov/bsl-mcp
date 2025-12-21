import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Создаем папку для логов, если её нет
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Формат логов
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _cleanup_old_logs():
    """Удаление логов старше 30 дней."""
    import time
    cutoff = time.time() - (30 * 24 * 60 * 60)
    for path in LOG_DIR.glob("*.log"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass

def setup_logging(log_level_str: str = "INFO"):
    """Настройка глобального логгера."""
    _cleanup_old_logs()
    
    # Преобразуем строку уровня в константу logging
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Корневой логгер
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 1. Вывод в консоль - ОТКЛЮЧЕНО ПО ТРЕБОВАНИЮ
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    # console_handler.setLevel(level)
    # logger.addHandler(console_handler)
    
    # 2. Вывод в файл
    log_file = LOG_DIR / "app.log"
    if log_file.exists():
        import datetime
        import re
        # Читаем первую строку и извлекаем timestamp начала сессии
        timestamp = None
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                # Формат: "2025-12-21 18:04:33 - ..."
                match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", first_line)
                if match:
                    dt = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
        except Exception:
            pass
        
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        new_name = LOG_DIR / f"app_{timestamp}.log"
        try:
            log_file.rename(new_name)
        except OSError:
            pass # Если файл занят, просто пишем в него дальше или он перезапишется ротацией

    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024, 
        backupCount=5, 
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    # Отключаем (или понижаем) шумные логгеры библиотек
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return logger

def get_logger(name: str):
    """Получение логгера для модуля."""
    return logging.getLogger(name)
