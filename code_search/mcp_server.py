import json
from typing import Any, Dict, List

from mcp.server import Server
import mcp.types as types
from .app_context import IBManager

def create_mcp_server(ib_manager: IBManager) -> Server:
    server = Server("1c-code-search")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_code",
                description="Поиск кода в информационных базах 1С. Поддерживает семантический поиск.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "Текст запроса для поиска (например: 'как создать справочник' или 'функция расчета НДС')"
                        },
                        "ib": {
                            "type": "string", 
                            "description": "Имя информационной базы для поиска (необязательно). Если не указано, поиск идет по всем."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Количество результатов (по умолчанию 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="list_ibs",
                description="Получить список доступных информационных баз 1С.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if not arguments:
            arguments = {}

        if name == "list_ibs":
            ibs = ib_manager.get_all_contexts()
            
            # Формируем красиво
            lines = ["Доступные базы:"]
            for ib in ibs:
                lines.append(f"- **{ib['name']}**: {ib['title']} (Статус: {ib['status']['status']})")
            
            return [types.TextContent(type="text", text="\n".join(lines))]
        
        if name == "search_code":
            query = arguments.get("query")
            ib_name = arguments.get("ib")
            limit = arguments.get("limit", 5)
            
            if not query:
                return [types.TextContent(type="text", text="Ошибка: Пустой запрос")]

            results_text = []
            
            # Если указана база
            if ib_name:
                ctx = ib_manager.get_context(ib_name)
                if not ctx:
                    return [types.TextContent(type="text", text=f"Ошибка: База '{ib_name}' не найдена")]
                
                res = ctx.searcher.search(query, top_k=limit)
                results_text.append(f"Результаты поиска в **{ib_name}**:")
                results_text.append(_format_results(res))
            else:
                # По всем базам
                contexts = ib_manager.get_all_contexts()
                if not contexts:
                    return [types.TextContent(type="text", text="Нет подключенных баз.")]

                all_res = []
                for ctx_info in contexts:
                    # ib_manager.get_all_contexts returns dicts, we need objects
                    # But wait, get_all_contexts returns list of dicts with status.
                    # We need the actual context object to search.
                    # IBManager should have a method to get real contexts or we use get_context in loop.
                    name = ctx_info['name']
                    ctx = ib_manager.get_context(name)
                    if ctx:
                        res = ctx.searcher.search(query, top_k=limit) # Search locally top_k
                        for r in res:
                            r['ib'] = name
                            all_res.append(r)
                
                # Sort by score
                all_res.sort(key=lambda x: x['score'], reverse=True)
                top_res = all_res[:limit]
                
                results_text.append(f"Результаты поиска по всем базам:")
                results_text.append(_format_results(top_res))

            return [types.TextContent(type="text", text="\n".join(results_text))]

        raise ValueError(f"Unknown tool: {name}")

    return server

def _format_results(results: List[Dict[str, Any]]) -> str:
    """Helper to format search results."""
    if not results:
        return "Ничего не найдено."
    
    out = []
    for i, r in enumerate(results, 1):
        score = f"{r['score']:.3f}"
        ib_info = f"[{r.get('ib', 'IB')}] " if 'ib' in r else ""
        file_path = r['payload'].get('file_path', 'unknown')
        code_snippet = r['payload'].get('text', '').strip()
        
        # Truncate code if too long
        if len(code_snippet) > 500:
             code_snippet = code_snippet[:500] + "..."

        out.append(f"\n{i}. {ib_info}**{file_path}** (Score: {score})")
        out.append(f"```bsl\n{code_snippet}\n```")
    
    return "\n".join(out)
