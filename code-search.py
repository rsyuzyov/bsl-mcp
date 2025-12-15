#!/usr/bin/env python3
"""
1C RAG + MCP сервер. Индексирует выгрузку 1С XML, CPU-only.
Usage: python code-search.py --source ./1c-dump
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="1C RAG + MCP сервер")
    parser.add_argument("--source", required=True, help="Каталог выгрузки 1С (с Configuration.xml)")
    parser.add_argument("--port", type=int, default=8000, help="MCP порт (default: 8000)")
    parser.add_argument("--index", default="./1c-index", help="Каталог индекса (default: ./1c-index)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def main():
    args = parse_args()

    # Тяжёлые импорты только после проверки аргументов
    import os
    from pathlib import Path
    from typing import Any

    import faiss
    import uvicorn
    from fastapi import FastAPI
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore

    # CPU embeddings
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Проверяем source
    if not os.path.exists(args.source):
        print(f"Ошибка: каталог {args.source} не найден", file=sys.stderr)
        sys.exit(1)

    if not any(f.endswith('.xml') for f in Path(args.source).rglob('*.xml')):
        print(f"Ошибка: в {args.source} нет XML файлов", file=sys.stderr)
        sys.exit(1)

    Path(args.index).mkdir(exist_ok=True)

    def build_index(source_dir: str, index_dir: str):
        print(f"Индексирую {source_dir}...")
        docs = SimpleDirectoryReader(source_dir, file_extractor={".xml": "text"}).load_data()
        print(f"Загружено {len(docs)} файлов")

        vector_store = FaissVectorStore(faiss.IndexFlatL2(384))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
        index.storage_context.persist(index_dir)
        print(f"Индекс сохранён: {index_dir}")
        return index

    # Загрузка/создание индекса
    try:
        index = VectorStoreIndex.from_persist_dir(args.index)
        print("Индекс загружен")
    except:
        index = build_index(args.source, args.index)

    query_engine = index.as_query_engine(similarity_top_k=5)

    # MCP сервер
    app = FastAPI()

    @app.post("/mcp/tools/1c_search")
    async def mcp_search(request: dict) -> dict[str, Any]:
        query = request["params"]["query"]
        result = query_engine.query(query)
        return {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "source": args.source}

    print(f"MCP сервер на http://0.0.0.0:{args.port}")
    print(f"Kiro: добавь в ~/.kiro/settings/mcp.json: \"1c-rag\": {{\"url\": \"http://localhost:{args.port}\"}}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
