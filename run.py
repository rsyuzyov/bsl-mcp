#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запуск BSL MCP сервера из виртуального окружения."""
import sys
import subprocess
import platform
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    venv_dir = base_dir / ".venv"
    
    # Путь к python в venv
    if platform.system() == "Windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    # Проверка наличия venv
    if not venv_python.exists():
        print(f"Ошибка: виртуальное окружение не найдено.")
        print(f"Сначала выполните: python install.py")
        pass
        sys.exit(1)

    # Запуск приложения
    print("--- [START] Запуск BSL MCP ---")
    
    # Отключаем телеметрию Qdrant (убирает задержку 25с при старте)
    import os
    os.environ["QDRANT_TELEMETRY_DISABLED"] = "1"

    try:
        cmd = [str(venv_python), "-m", "code_search"] + sys.argv[1:]
        # Явно передаем environment с установленной переменной
        env = os.environ.copy()
        env["QDRANT_TELEMETRY_DISABLED"] = "1"
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    except Exception as e:
        print(f"Ошибка запуска: {e}")
        pass

if __name__ == "__main__":
    main()
