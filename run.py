#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import platform

def main():
    # Определяем корневую директорию проекта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base_dir, ".venv")
    requirements_file = os.path.join(base_dir, "requirements.txt")
    
    # Определяем пути к исполняемым файлам в venv
    if platform.system() == "Windows":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        # venv_pip можно не определять отдельно, вызовем через python -m pip
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    # 1. Создание виртуального окружения
    # Если папки нет или нет интерпретатора внутри
    need_install = False
    if not os.path.exists(venv_dir) or not os.path.exists(venv_python):
        print(f"--- [START] Создаю виртуальное окружение в {venv_dir} ---")
        try:
            # Создаем venv
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
            need_install = True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при создании venv: {e}")
            sys.exit(1)
    
    # Если только создали, ставим зависимости
    if need_install:
        print("--- [START] Устанавливаю зависимости ---")
        try:
            subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
            if os.path.exists(requirements_file):
                subprocess.check_call([venv_python, "-m", "pip", "install", "-r", requirements_file])
            else:
                print(f"Внимание: файл {requirements_file} не найден.")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка установки зависимостей: {e}")
            if platform.system() == "Windows":
                input("Нажмите Enter...")
            sys.exit(1)

    # 2. Запуск приложения
    print("--- [START] Запуск BSL MCP ---")
    
    try:
        # Запускаем целевой модуль через python из venv
        cmd = [venv_python, "-m", "code_search"] + sys.argv[1:]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    except Exception as e:
        print(f"Ошибка запуска: {e}")
        if platform.system() == "Windows":
            input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()
