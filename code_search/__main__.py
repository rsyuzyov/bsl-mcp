#!/usr/bin/env python3
"""
1C RAG + MCP сервер. Индексирует выгрузку 1С XML, CPU-only.
Поддержка нескольких ИБ.
Usage: python -m code_search
"""
import socket
import threading
import sys
import os

# Отключаем телеметрию Qdrant ДО всех импортов
os.environ["QDRANT_TELEMETRY_DISABLED"] = "1"

import uvicorn
from pathlib import Path

from .config_manager import ConfigManager, IBConfig
from .app_context import IBManager
from .web import create_app
from .cli import parse_args
from .logger import setup_logging, get_logger


def is_port_in_use(port: int) -> bool:
    """Проверка занятости порта."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    # 1. Настройка логирования - ПЕРВЫМ ДЕЛОМ (ротация старого лога)
    logger = setup_logging()  # Сначала с уровнем по умолчанию
    logger.info("Запуск Multi-IB Code Search...")

    args = parse_args()
    
    # 2. Загрузка конфигурации
    config_mgr = ConfigManager()
    config = config_mgr.load()
    
    # Переопределение порта из аргументов
    if args.port:
        config.port = args.port

    # Применяем уровень логирования из конфига
    import logging
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    # 3. Проверка занятости порта ДО тяжёлой инициализации
    if is_port_in_use(config.port):
        logger.error(f"Порт {config.port} уже занят. Возможно, приложение уже запущено.")
        sys.exit(1)

    # 4. Инициализация менеджера ИБ (фоновая загрузка моделей)
    ib_manager = IBManager(config_mgr)
    
    # 5. Создание приложения (сразу доступен веб-интерфейс)
    app = create_app(ib_manager)
    
    logger.info(f"Сервер доступен на http://localhost:{config.port}")
    if not config.ibs:
        logger.warning("Список ИБ пуст. Добавьте ИБ через веб-интерфейс.")

    # 6. Запуск сервера
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.port, log_level=config.log_level.lower())
    except KeyboardInterrupt:
        logger.info("Остановка по KeyboardInterrupt...")
        # Stop background threads
        for ctx in ib_manager.get_working_contexts():
            ctx.stop_maintenance()


if __name__ == "__main__":
    main()
