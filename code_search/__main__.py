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

def main():
    args = parse_args()
    print("Запуск Multi-IB Code Search...")
    
    # 1. Загрузка конфигурации
    config_mgr = ConfigManager() # loads config.yaml by default or from args (if we extended cli)
    config = config_mgr.load()
    
    # 2. Переопределение порта из аргументов
    if args.port:
        config.port = args.port

    # 3. Инициализация менеджера ИБ
    ib_manager = IBManager(config_mgr)
    ib_manager.initialize()
    
    # 4. Создание приложения
    app = create_app(ib_manager)
    
    print(f"Сервер доступен на http://localhost:{config.port}")
    if not config.ibs:
        print("Внимание: Список ИБ пуст. Добавьте ИБ через веб-интерфейс.")

    # 5. Запуск сервера
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="warning")
    except KeyboardInterrupt:
        print("Остановка...")
        # Stop background threads
        for ctx in ib_manager.get_all_contexts():
            ctx.stop_maintenance()


if __name__ == "__main__":
    main()
