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
    
    
    # Setup templates and static files
    base_dir = Path(__file__).parent
    template_dir = base_dir / "templates"
    static_dir = base_dir / "static"
    
    template_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(template_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        # List IBs
        ibs = ib_manager.get_all_contexts()
        
        # Calculate system paths for instructions
        import sys
        repo_path = str(Path(__file__).parents[2].resolve())
        python_path = sys.executable

        return templates.TemplateResponse("list.html", {
            "request": request,
            "ibs": ibs,
            "active_page": "config",
            "repo_path": repo_path,
            "python_path": python_path
        })
    
    @app.get("/search", response_class=HTMLResponse)
    async def search_page(request: Request):
        ibs = ib_manager.get_all_contexts()
        return templates.TemplateResponse("search.html", {
            "request": request,
            "ibs": ibs,  # For dropdown if needed
            "active_page": "search"
        })

    @app.get("/info", response_class=HTMLResponse)
    async def info_page(request: Request):
        return templates.TemplateResponse("info.html", {
            "request": request,
            "active_page": "info"
        })

    @app.get("/ib/{name}", response_class=HTMLResponse)
    async def ib_page(request: Request, name: str):
        ctx = ib_manager.get_context(name)
        if not ctx:
            return HTMLResponse("ИБ не найдена", status_code=404)
        
        # Reuse list template or separate? 
        # For now maybe just redirect to home or show detailed view?
        # User asked for "Sections: Config, Search, Info". 
        # Detailed view is probably part of Config logic or Search.
        # Let's keep it simple for now, maybe render a specific template.
        return templates.TemplateResponse("ib.html", {
            "request": request,
            "ctx": ctx,
            "active_page": "config"
        })

    @app.post("/api/ib/add")
    async def add_ib(
        name: str = Form(...), 
        title: str = Form(""), 
        source_dir: str = Form(...),
        index_dir: str = Form(...),
        embedding_model: str = Form("cointegrated/rubert-tiny2"),
        engine: str = Form("qdrant")
    ):
        try:
            conf = IBConfig(
                name=name,
                title=title or name,
                source_dir=source_dir,
                index_dir=index_dir,
                embedding_model=embedding_model,
                vector_db=engine
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
        
    @app.post("/api/system/fs")
    async def list_fs(path: str = Form(".")):
        """List directories in path."""
        try:
            p = Path(path).resolve()
            if not p.exists():
                p = Path(".").resolve()
            
            # Windows drives check if parent is same (root)
            is_root = p.anchor == str(p)
            
            dirs = []
            
            # If windows and we want to list drives? 
            # Path("/") on windows goes to current drive root.
            # To list drives we need specific logic.
            # Simple workaround: if path is empty or special, show drives.
            # But python pathlib is tricky with drives.
            # Let's just list child directories of current path.
            
            # Up dir
            if not is_root:
                dirs.append({"name": "..", "path": str(p.parent)})
                
            for item in p.iterdir():
                try:
                    if item.is_dir() and not item.name.startswith("."):
                        dirs.append({"name": item.name, "path": str(item)})
                except PermissionError:
                    continue
            
            dirs.sort(key=lambda x: x["name"])
            
            return {"current": str(p), "dirs": dirs}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/system/mkdir")
    async def make_dir(path: str = Form(...), name: str = Form(...)):
        """Создать папку."""
        try:
            p = Path(path).resolve() / name
            if p.exists():
                 return {"error": "Папка уже существует"}
            p.mkdir(parents=False, exist_ok=False)
            return {"success": True, "path": str(p)}
        except Exception as e:
             return {"error": str(e)}

    # --- MCP Integration ---
    from ..mcp_server import create_mcp_server
    from mcp.server.sse import SseServerTransport

    mcp_server = create_mcp_server(ib_manager)
    sse_transport = SseServerTransport("/messages")

    @app.get("/sse")
    async def handle_sse(request: Request):
        try:
             async with sse_transport.connect_sse(
                 request.scope, request.receive, request._send
             ) as streams:
                 await mcp_server.run(
                     streams[0], streams[1], 
                     sse_transport.initialization_options
                 )
        except Exception:
             # Client disconnected or error
             pass

    @app.post("/messages")
    async def handle_messages(request: Request):
        await sse_transport.handle_post_message(request.scope, request.receive, request._send)

    return app

