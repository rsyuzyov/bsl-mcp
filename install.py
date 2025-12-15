#!/usr/bin/env python3
"""Установка зависимостей для code-search.py"""
import subprocess
import sys

PACKAGES = [
    "llama-index-core",
    "llama-index-embeddings-huggingface",
    "llama-index-vector-stores-faiss",
    "faiss-cpu",
    "fastapi",
    "uvicorn",
    "sentence-transformers",
    "pymorphy3",
]

def main():
    print("Устанавливаю зависимости...")
    cmd = [sys.executable, "-m", "pip", "install"] + PACKAGES
    subprocess.run(cmd, check=True)
    print("\nГотово! Запуск: python code-search.py --source ./путь-к-выгрузке-1с")

if __name__ == "__main__":
    main()
