"""FastAPI приложение."""
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..app_context import IBManager, IBConfig
from ..model_manager import ModelManager


def create_app(ib_manager: IBManager) -> FastAPI:
    """Создать FastAPI приложение."""
    app = FastAPI()
    
    # Simple templates using string replacement or basic Jinja2 if we add dependency?
    # Original used read_text().replace(). Let's strive for better UI.
    # We can use simple Jinja2 since FastAPI usually comes with it or we added it implicitly?
    # requirements.txt has fastapi, uvicorn. It doesn't list jinja2 explicitly but fastapi[all] includes it?
    # We listed `fastapi`. Let's assume we might need to add jinja2 or stick to simple formatting.
    # To follow requirements "WOW the user", we should use better templates. 
    # Let's add jinja2 safely by checking import or fallback.
    # Actually, let's just use string templates for simplicity to avoid new deps if possible, 
    # BUT "WOW" requires good HTML/CSS.
    # Let's use simple f-strings for HTML generation or simple helper.
    
    # We will create a better HTML structure in `web/templates`.
    
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    # Helper to load template
    def render(name: str, **kwargs):
        path = template_dir / name
        if not path.exists():
            return f"Template {name} not found"
        html = path.read_text(encoding="utf-8")
        for k, v in kwargs.items():
            html = html.replace(f"{{{{ {k} }}}}", str(v))
        return html

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        # List IBs
        ibs = ib_manager.get_all_contexts()
        # Generate list HTML
        ib_list_html = ""
        for ctx in ibs:
            ib_list_html += f"""
            <div class="ib-card" onclick="window.location='/ib/{ctx.config.name}'">
                <h3>{ctx.config.title}</h3>
                <p>{ctx.config.name}</p>
                <div class="status">
                    <span class="bad">●</span> {'Индексация' if ctx.status.running else 'Готов'}
                </div>
            </div>
            """
        
        # Load main template
        # We need to create 'list.html' or update 'index.html'
        return render("list.html", ib_list=ib_list_html)

    @app.get("/ib/{name}", response_class=HTMLResponse)
    async def ib_page(name: str):
        ctx = ib_manager.get_context(name)
        if not ctx:
            return HTMLResponse("ИБ не найдена", status_code=404)
        
        return render("ib.html", 
            ib_name=ctx.config.name, 
            ib_title=ctx.config.title,
            collection_count=ctx.engine.get_collection_count()
        )

    @app.post("/api/ib/add")
    async def add_ib(
        name: str = Form(...), 
        title: str = Form(""), 
        source_dir: str = Form(...),
        index_dir: str = Form(...),
        embedding_model: str = Form("cointegrated/rubert-tiny2")
    ):
        try:
            conf = IBConfig(
                name=name,
                title=title or name,
                source_dir=source_dir,
                index_dir=index_dir,
                embedding_model=embedding_model
            )
            ib_manager.add_ib(conf)
            return RedirectResponse("/", status_code=303)
        except Exception as e:
            return HTMLResponse(f"Ошибка: {e}", status_code=400)

    @app.post("/api/ib/{name}/delete")
    async def delete_ib(name: str):
        ib_manager.remove_ib(name)
        return RedirectResponse("/", status_code=303)

    # API endpoints for specific IB
    
    @app.get("/api/ib/{name}/status")
    async def get_ib_status(name: str):
        ctx = ib_manager.get_context(name)
        if not ctx:
            raise HTTPException(404, "IB not found")
        
        progress = ctx.status.format_progress(ctx.engine.get_collection_count())
        return progress

    @app.get("/api/ib/{name}/search")
    async def search_ib(name: str, q: str):
        ctx = ib_manager.get_context(name)
        if not ctx:
            raise HTTPException(404, "IB not found")
        return ctx.searcher.search(q)

    @app.post("/api/ib/{name}/reindex/full")
    async def reindex_full(name: str):
        ctx = ib_manager.get_context(name)
        if not ctx:
           raise HTTPException(404, "IB not found")
        
        if ctx.status.running:
             return {"error": "Индексация уже запущена"}
             
        threading.Thread(target=ctx.engine.full_reindex, args=(ctx.status,), daemon=True).start()
        return {"started": True}

    @app.post("/api/ib/{name}/reindex/incremental")
    async def reindex_incremental(name: str):
        ctx = ib_manager.get_context(name)
        if not ctx:
           raise HTTPException(404, "IB not found")
        
        if ctx.status.running:
             return {"error": "Индексация уже запущена"}
             
        threading.Thread(target=ctx.engine.incremental_reindex, args=(ctx.status,), daemon=True).start()
        return {"started": True}

    # MCP Tool endpoint (assuming it works for ANY IB? Or needs param?)
    @app.post("/mcp/tools/1c_search")
    async def mcp_search(request: dict) -> dict[str, Any]:
        # query should contain like "ib_name: query" or we search all? 
        # Or we assume default IB?
        # User requirement didn't specify MCP changes, but we should support it.
        # Let's search ALL IBs or parse query.
        
        query = request["params"]["query"]
        
        # Simple heuristic: "zup3: запрос"
        parts = query.split(":", 1)
        if len(parts) == 2:
            ib_name = parts[0].strip()
            q_text = parts[1].strip()
            ctx = ib_manager.get_context(ib_name)
            if ctx:
                 results = ctx.searcher.search(q_text)
                 return {"content": [{"type": "text", "text": str(results)}], "isError": False}
        
        # If no IB specified, search all?
        # For now, just search the first one or return help
        all_res = []
        for ctx in ib_manager.get_all_contexts():
             res = ctx.searcher.search(query, top_k=3)
             for r in res:
                 r["ib"] = ctx.config.name
                 all_res.append(r)
        
        all_res.sort(key=lambda x: x["score"], reverse=True)
        return {"content": [{"type": "text", "text": str(all_res[:5])}], "isError": False}
        
    return app
