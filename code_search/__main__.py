#!/usr/bin/env python3
"""
1C RAG + MCP сервер. Индексирует выгрузку 1С XML, CPU-only.
Поддержка нескольких ИБ.
Usage: python -m code_search
"""
import threading
import sys
import uvicorn
from pathlib import Path

from .config_manager import ConfigManager, IBConfig
from .app_context import IBManager
from .web import create_app
from .cli import parse_args
from .logger import setup_logging, get_logger

def main():
    args = parse_args()
    
    # 1. Загрузка конфигурации
    config_mgr = ConfigManager()
    config = config_mgr.load()

    # 2. Настройка логирования
    logger = setup_logging(config.log_level)
    logger.info("Запуск Multi-IB Code Search...")
    
    # 3. Переопределение порта из аргументов
    if args.port:
        config.port = args.port

    # 4. Инициализация менеджера ИБ
    ib_manager = IBManager(config_mgr)
    ib_manager.initialize()
    
    # 5. Создание приложения
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
        for ctx in ib_manager.get_all_contexts():
            ctx.stop_maintenance()


if __name__ == "__main__":
    main()
