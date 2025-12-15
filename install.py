#!/usr/bin/env python3
"""Установка зависимостей для code-search.py"""
import subprocess
import sys

PACKAGES = [
    "qdrant-client",
    "sentence-transformers",
    "fastapi",
    "uvicorn",
    "pymorphy3",
    "tqdm",
]

def main():
    print("Устанавливаю зависимости...")
    cmd = [sys.executable, "-m", "pip", "install"] + PACKAGES
    subprocess.run(cmd, check=True)
    print("\nГотово! Запуск: python code-search.py --source ./путь-к-выгрузке-1с")

if __name__ == "__main__":
    main()
