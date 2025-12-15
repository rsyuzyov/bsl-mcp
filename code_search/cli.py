"""CLI парсер аргументов."""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="1C RAG + MCP сервер")
    parser.add_argument("--source", required=True, help="Каталог выгрузки 1С (с Configuration.xml)")
    parser.add_argument("--port", type=int, default=8000, help="MCP порт (default: 8000)")
    parser.add_argument("--index", default="./1c-index", help="Каталог индекса (default: ./1c-index)")
    parser.add_argument("--name", default=None, help="Название ИБ (если не указано, берётся из Configuration.xml)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()
