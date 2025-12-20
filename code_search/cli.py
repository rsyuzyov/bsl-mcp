"""CLI парсер аргументов."""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="1C RAG + MCP сервер")
    parser.add_argument("--port", type=int, default=None, help="MCP порт (переопределяет конфиг)")

    return parser.parse_args()
