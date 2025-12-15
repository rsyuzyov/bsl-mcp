"""FastAPI приложение."""
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import indexing_status, model_state
from ..indexer import IndexEngine
from ..search import HybridSearch


def create_app(engine: IndexEngine, searcher: HybridSearch, config_name: str) -> FastAPI:
    """Создать FastAPI приложение."""
    app = FastAPI()
    
    template_path = Path(__file__).parent / "templates" / "index.html"
    template = template_path.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        return template.replace("{{ config_name }}", config_name)

    @app.get("/indexing-status")
    async def get_indexing_status():
        return indexing_status.to_dict()

    @app.get("/indexing-progress")
    async def get_indexing_progress():
        """Форматированный прогресс для веба."""
        return indexing_status.format_progress(engine.get_collection_count())

    @app.post("/mcp/tools/1c_search")
    async def mcp_search(request: dict) -> dict[str, Any]:
        query = request["params"]["query"]
        results = searcher.search(query)
        return {"content": [{"type": "text", "text": str(results)}], "isError": False}

    @app.get("/search")
    async def search_get(q: str) -> list[dict[str, Any]]:
        return searcher.search(q)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "source": str(engine.source_dir),
            "chunks_in_db": engine.get_collection_count(),
            "chunks_pending": indexing_status.total_chunks - indexing_status.chunks_in_db if indexing_status.running else 0,
            "indexing": indexing_status.running
        }

    @app.post("/reindex/full")
    async def reindex_full_endpoint():
        if model_state.loading:
            return {"error": "Модель ещё загружается"}
        if indexing_status.running:
            return {"error": "Индексация уже запущена"}
        threading.Thread(target=engine.full_reindex, daemon=True).start()
        return {"started": True, "mode": "full"}

    @app.post("/reindex/incremental")
    async def reindex_incremental_endpoint():
        if model_state.loading:
            return {"error": "Модель ещё загружается"}
        if indexing_status.running:
            return {"error": "Индексация уже запущена"}
        threading.Thread(target=engine.incremental_reindex, daemon=True).start()
        return {"started": True, "mode": "incremental"}

    return app
