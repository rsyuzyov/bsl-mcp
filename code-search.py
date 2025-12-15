#!/usr/bin/env python3
"""
1C RAG + MCP сервер. Индексирует выгрузку 1С XML, CPU-only.
Usage: python code-search.py --source ./1c-dump
       python -m code_search --source ./1c-dump
"""
from code_search.__main__ import main

if __name__ == "__main__":
    main()
