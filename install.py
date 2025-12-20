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
    "optimum[onnxruntime]",
    "PyYAML",
    "python-multipart",
]

def main():
    print("Устанавливаю зависимости...")
    cmd = [sys.executable, "-m", "pip", "install"] + PACKAGES
    subprocess.run(cmd, check=True)
    print("\nГотово! Запуск: python -m code_search")

if __name__ == "__main__":
    main()
