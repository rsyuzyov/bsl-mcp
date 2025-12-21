#!/usr/bin/env python3
"""Установка зависимостей для code-search.py в локальное виртуальное окружение .venv"""
import subprocess
import sys
import os
import platform
import msvcrt
import time
from pathlib import Path

# Константы
VENV_DIR = Path(".venv")

def input_with_timeout(prompt, timeout, default):
    print(prompt, end='', flush=True)
    start_time = time.time()
    input_str = ''
    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            if char == '\r' or char == '\n':
                print()
                return input_str if input_str else default
            input_str += char
        
        if time.time() - start_time > timeout:
            print(f"\nВремя истекло. Выбрано по умолчанию: {default}")
            return default
        
        time.sleep(0.1)

def get_venv_python():
    """Возвращает путь к python внутри venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"

def create_venv():
    """Создает виртуальное окружение, если оно не существует"""
    if VENV_DIR.exists():
        print(f"Виртуальное окружение {VENV_DIR} уже существует.")
        return

    print(f"Создание виртуального окружения в {VENV_DIR}...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("Виртуальное окружение создано успешно.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при создании venv: {e}")
        sys.exit(1)

def install_requirements(python_exe, requirements_file):
    """Устанавливает зависимости из указанного файла требований"""
    if not os.path.exists(requirements_file):
        print(f"ОШИБКА: Файл {requirements_file} не найден!")
        sys.exit(1)

    print(f"Установка зависимостей из {requirements_file}...")
    cmd = [str(python_exe), "-m", "pip", "install", "-r", requirements_file]
    print(f"Выполнение: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("=== Установка зависимостей Code Search (в .venv) ===")
    
    # 1. Создаем venv
    create_venv()
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print(f"ОШИБКА: Интерпретатор не найден по пути {venv_python}")
        sys.exit(1)

    # 2. Обновляем pip внутри venv
    print("\nОбновление pip...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("\nВыберите режим ускорения:")
    print("1. DirectML (рекомендуется для Windows: AMD, Intel, NVIDIA)")
    print("2. NVIDIA CUDA (только для мощных карт NVIDIA)")
    print("Если не знаете, что выбрать - выбирайте 1.")
    
    choice = input_with_timeout("Ваш выбор [1] (10 сек): ", 10, "1").strip()
    
    requirements_file = "requirements-dml.txt" # Default/DirectML
    
    if choice == "2":
        print("Выбран режим NVIDIA CUDA")
        requirements_file = "requirements-cuda.txt"
    else:
        # DirectML (default)
        print("Выбран режим DirectML (или по умолчанию)")
        requirements_file = "requirements-dml.txt"

    try:
        install_requirements(venv_python, requirements_file)
            
        print("\nГотово! Теперь все пакеты установлены в изолированное окружение .venv")
        print("Для запуска используйте: python run.py")
            
    except subprocess.CalledProcessError as e:
        print(f"\nОшибка установки: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
