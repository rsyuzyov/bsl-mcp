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
    from fastapi.responses import HTMLResponse
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore

    # CPU embeddings — мультиязычная модель для русского и 1С
    print("Загрузка модели эмбеддингов...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        query_instruction="query: ",  # e5 требует префикс для запросов
    )
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    # Проверяем source
    if not os.path.exists(args.source):
        print(f"Ошибка: каталог {args.source} не найден", file=sys.stderr)
        sys.exit(1)

    if not any(Path(args.source).rglob('*.xml')):
        print(f"Ошибка: в {args.source} нет XML файлов", file=sys.stderr)
        sys.exit(1)

    Path(args.index).mkdir(exist_ok=True)

    def build_index(source_dir: str, index_dir: str):
        print(f"Чтение XML файлов из {source_dir}...")
        docs = SimpleDirectoryReader(
            source_dir,
            recursive=True,
            required_exts=[".xml", ".bsl"]
        ).load_data()
        print(f"Загружено {len(docs)} файлов")
        print("Создание векторного индекса (может занять несколько минут)...")

        vector_store = FaissVectorStore(faiss.IndexFlatIP(384))  # cosine similarity
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
        index.storage_context.persist(index_dir)
        print(f"Индекс сохранён: {index_dir}")
        return index

    # Загрузка/создание индекса
    try:
        print("Загрузка индекса с диска...")
        index = VectorStoreIndex.from_persist_dir(args.index)
        print("Индекс загружен")
    except:
        index = build_index(args.source, args.index)

    retriever = index.as_retriever(similarity_top_k=5)

    # MCP сервер
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>1C Code Search</title>
<style>body{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px}
input{width:70%;padding:10px;font-size:16px}button{padding:10px 20px;font-size:16px}
.result{border:1px solid #ddd;margin:10px 0;padding:15px;border-radius:5px}
.file{color:#666;font-size:12px}.score{color:#090;font-weight:bold}
pre{background:#f5f5f5;padding:10px;overflow-x:auto;font-size:13px}</style></head>
<body><h1>🔍 1C Code Search</h1>
<form onsubmit="search();return false"><input id="q" placeholder="Поиск по коду 1С..." autofocus>
<button>Найти</button></form><div id="results"></div>
<script>async function search(){const q=document.getElementById('q').value;
const r=document.getElementById('results');r.innerHTML='Поиск...';
const res=await fetch('/search?q='+encodeURIComponent(q));const data=await res.json();
r.innerHTML=data.map(d=>`<div class="result"><span class="score">${d.score.toFixed(3)}</span>
<span class="file">${d.file}</span><pre>${d.text.replace(/</g,'&lt;')}</pre></div>`).join('')||'Ничего не найдено';}</script>
</body></html>"""

    @app.post("/mcp/tools/1c_search")
    async def mcp_search(request: dict) -> dict[str, Any]:
        query = request["params"]["query"]
        nodes = retriever.retrieve(query)
        results = [{"score": n.score, "file": n.node.metadata.get("file_path", ""), "text": n.node.text[:500]} for n in nodes]
        return {
            "content": [{"type": "text", "text": str(results)}],
            "isError": False
        }

    @app.get("/search")
    async def search_get(q: str) -> list[dict[str, Any]]:
        """GET /search?q=запрос — для тестирования в браузере"""
        nodes = retriever.retrieve(q)
        return [{"score": n.score, "file": n.node.metadata.get("file_path", ""), "text": n.node.text[:500]} for n in nodes]

    @app.get("/health")
    async def health():
        return {"status": "ok", "source": args.source}

    print(f"MCP сервер на http://0.0.0.0:{args.port}")
    print(f"Kiro: добавь в ~/.kiro/settings/mcp.json: \"1c-rag\": {{\"url\": \"http://localhost:{args.port}\"}}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
