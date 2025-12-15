#!/usr/bin/env python3
"""
1C RAG + MCP сервер. Индексирует выгрузку 1С XML, CPU-only.
Использует Qdrant для инкрементальной индексации.
Usage: python -m code_search --source ./1c-dump
"""
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .cli import parse_args
from .config import model_state, indexing_status, CHECK_INTERVAL
from .indexer import IndexEngine
from .indexer.hasher import load_hashes
from .search import HybridSearch
from .web import create_app


def get_config_name(source_dir: str) -> str:
    """Получить название конфигурации из Configuration.xml."""
    config_file = Path(source_dir) / "Configuration.xml"
    if not config_file.exists():
        return Path(source_dir).name
    try:
        tree = ET.parse(config_file)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag.endswith("}Synonym") or elem.tag == "Synonym":
                for item in elem.iter():
                    if item.tag.endswith("}content") or item.tag == "content":
                        if item.text:
                            return item.text
        return Path(source_dir).name
    except Exception:
        return Path(source_dir).name


def load_model_background():
    """Загрузка модели в фоне."""
    try:
        print("Загрузка модели эмбеддингов...")
        from sentence_transformers import SentenceTransformer
        model_state.model = SentenceTransformer("intfloat/multilingual-e5-small")
        model_state.loading = False
        print("Модель загружена")
    except Exception as e:
        model_state.error = str(e)
        model_state.loading = False
        print(f"Ошибка загрузки модели: {e}")


def main():
    args = parse_args()
    print("Загрузка библиотек...")

    if not os.path.exists(args.source):
        print(f"Ошибка: каталог {args.source} не найден", file=sys.stderr)
        sys.exit(1)

    if not any(Path(args.source).rglob("*.xml")):
        print(f"Ошибка: в {args.source} нет XML файлов", file=sys.stderr)
        sys.exit(1)

    config_name = args.name if args.name else get_config_name(args.source)
    
    # Загрузка модели в фоне
    threading.Thread(target=load_model_background, daemon=True).start()

    # Инициализация движка
    engine = IndexEngine(args.source, args.index)
    searcher = HybridSearch(engine.client)
    app = create_app(engine, searcher, config_name)


    def startup_indexing():
        """Ждём загрузки модели, потом проверяем индекс."""
        while model_state.loading:
            time.sleep(0.5)
        if model_state.error:
            return
        
        count = engine.get_collection_count()
        old_hashes = load_hashes(engine.meta_file)
        
        if count == 0 or not old_hashes:
            print("Индекс пуст, запускаю полную индексацию...")
            engine.full_reindex()
        else:
            print(f"Индекс загружен: {count} чанков")
            print("Проверка изменений...")
            has_changes, added, changed, deleted = engine.quick_check_changes()
            if has_changes:
                print(f"Обнаружены изменения: +{added} ~{changed} -{deleted}, запускаю обновление...")
                engine.incremental_reindex()
            else:
                print("Изменений нет")

    def periodic_check():
        """Проверка изменений каждые 5 минут."""
        while True:
            time.sleep(CHECK_INTERVAL)
            if indexing_status.running or model_state.loading or model_state.error:
                continue
            has_changes, added, changed, deleted = engine.quick_check_changes()
            if has_changes:
                print(f"[auto] Обнаружены изменения: +{added} ~{changed} -{deleted}, запускаю обновление...")
                threading.Thread(target=engine.incremental_reindex, daemon=True).start()

    # Запуск индексации после загрузки модели
    threading.Thread(target=startup_indexing, daemon=True).start()

    # Фоновая проверка каждые 5 минут
    threading.Thread(target=periodic_check, daemon=True).start()

    print(f"Сервер на http://localhost:{args.port}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
