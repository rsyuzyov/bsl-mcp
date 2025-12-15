#!/usr/bin/env python3
"""
1C RAG + MCP сервер. Индексирует выгрузку 1С XML, CPU-only.
Использует Qdrant для инкрементальной индексации.
Usage: python code-search.py --source ./1c-dump
"""
import argparse
import hashlib
import json
import sys
import threading
import time


def parse_args():
    parser = argparse.ArgumentParser(description="1C RAG + MCP сервер")
    parser.add_argument("--source", required=True, help="Каталог выгрузки 1С (с Configuration.xml)")
    parser.add_argument("--port", type=int, default=8000, help="MCP порт (default: 8000)")
    parser.add_argument("--index", default="./1c-index", help="Каталог индекса (default: ./1c-index)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


# Глобальное состояние прогресса индексации
indexing_status = {
    "running": False,
    "mode": None,  # "full" или "incremental"
    "total_files": 0,
    "processed_files": 0,
    "total_chunks": 0,
    "speed": 0,
    "elapsed": 0,
    "eta": None,
    "error": None,
    "last_file": "",
}


def main():
    args = parse_args()

    print("Загрузка библиотек...")

    import os
    import re
    from pathlib import Path
    from typing import Any

    import pymorphy3
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer

    morph = pymorphy3.MorphAnalyzer()

    print("Загрузка модели эмбеддингов...")
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    VECTOR_SIZE = 384

    if not os.path.exists(args.source):
        print(f"Ошибка: каталог {args.source} не найден", file=sys.stderr)
        sys.exit(1)

    if not any(Path(args.source).rglob("*.xml")):
        print(f"Ошибка: в {args.source} нет XML файлов", file=sys.stderr)
        sys.exit(1)

    Path(args.index).mkdir(exist_ok=True)
    meta_file = Path(args.index) / "file_hashes.json"
    qdrant_path = Path(args.index) / "qdrant"

    client = QdrantClient(path=str(qdrant_path))
    COLLECTION_NAME = "1c_code"

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    def file_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def load_hashes() -> dict[str, str]:
        if meta_file.exists():
            return json.loads(meta_file.read_text(encoding="utf-8"))
        return {}

    def save_hashes(hashes: dict[str, str]):
        meta_file.write_text(json.dumps(hashes, ensure_ascii=False), encoding="utf-8")

    def get_all_files(source_dir: str) -> list[Path]:
        source_path = Path(source_dir)
        return list(source_path.rglob("*.xml")) + list(source_path.rglob("*.bsl"))

    def embed_texts(texts: list[str]) -> list[list[float]]:
        prefixed = ["passage: " + t for t in texts]
        embeddings = embed_model.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 100) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def get_collection_count() -> int:
        try:
            info = client.get_collection(COLLECTION_NAME)
            return info.points_count
        except:
            return 0

    def full_reindex(source_dir: str) -> dict:
        """Полная переиндексация в фоне"""
        global indexing_status
        
        indexing_status = {
            "running": True, "mode": "full", "total_files": 0, "processed_files": 0,
            "total_chunks": 0, "speed": 0, "elapsed": 0, "eta": None, "error": None, "last_file": ""
        }
        
        try:
            client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            
            files = get_all_files(source_dir)
            indexing_status["total_files"] = len(files)
            
            hashes = {}
            total_chunks = 0
            batch_points = []
            batch_size = 100
            point_id = 0
            start_time = time.time()
            
            for i, file_path in enumerate(files):
                try:
                    content = file_path.read_text(encoding="utf-8-sig", errors="ignore")
                    rel_path = str(file_path.relative_to(source_dir))
                    hashes[rel_path] = file_hash(file_path)
                    indexing_status["last_file"] = rel_path
                    
                    chunks = chunk_text(content)
                    for j, chunk in enumerate(chunks):
                        batch_points.append({
                            "id": point_id, "text": chunk,
                            "payload": {"file_path": rel_path, "chunk": j}
                        })
                        point_id += 1
                        total_chunks += 1
                    
                    if len(batch_points) >= batch_size:
                        texts = [p["text"] for p in batch_points]
                        embeddings = embed_texts(texts)
                        points = [
                            PointStruct(id=p["id"], vector=emb, payload={**p["payload"], "text": p["text"]})
                            for p, emb in zip(batch_points, embeddings)
                        ]
                        client.upsert(collection_name=COLLECTION_NAME, points=points)
                        batch_points = []
                    
                    # Обновляем статус
                    elapsed = time.time() - start_time
                    indexing_status.update({
                        "processed_files": i + 1,
                        "total_chunks": total_chunks,
                        "elapsed": round(elapsed, 1),
                        "speed": round(total_chunks / elapsed, 1) if elapsed > 0 else 0,
                        "eta": round((len(files) - i - 1) * elapsed / (i + 1), 0) if i > 0 else None
                    })
                except Exception as e:
                    print(f"Ошибка {file_path}: {e}")
            
            # Остаток
            if batch_points:
                texts = [p["text"] for p in batch_points]
                embeddings = embed_texts(texts)
                points = [
                    PointStruct(id=p["id"], vector=emb, payload={**p["payload"], "text": p["text"]})
                    for p, emb in zip(batch_points, embeddings)
                ]
                client.upsert(collection_name=COLLECTION_NAME, points=points)
            
            save_hashes(hashes)
            indexing_status["running"] = False
            return {"files": len(files), "chunks": total_chunks, "time_sec": round(time.time() - start_time, 1)}
        
        except Exception as e:
            indexing_status["running"] = False
            indexing_status["error"] = str(e)
            raise

    def incremental_reindex(source_dir: str) -> dict:
        """Инкрементальная индексация в фоне"""
        global indexing_status
        
        indexing_status = {
            "running": True, "mode": "incremental", "total_files": 0, "processed_files": 0,
            "total_chunks": 0, "speed": 0, "elapsed": 0, "eta": None, "error": None, "last_file": ""
        }
        
        try:
            files = get_all_files(source_dir)
            old_hashes = load_hashes()
            new_hashes = {}
            
            added, updated, deleted = 0, 0, 0
            current_files = set()
            start_time = time.time()
            max_id = get_collection_count()
            
            # Считаем изменённые файлы
            to_process = []
            for file_path in files:
                rel_path = str(file_path.relative_to(source_dir))
                current_files.add(rel_path)
                h = file_hash(file_path)
                new_hashes[rel_path] = h
                if rel_path not in old_hashes:
                    to_process.append((file_path, rel_path, "new"))
                elif old_hashes[rel_path] != h:
                    to_process.append((file_path, rel_path, "changed"))
            
            indexing_status["total_files"] = len(to_process)
            
            for i, (file_path, rel_path, status) in enumerate(to_process):
                try:
                    indexing_status["last_file"] = rel_path
                    
                    if status == "changed":
                        client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=Filter(
                                must=[FieldCondition(key="file_path", match=MatchValue(value=rel_path))]
                            )
                        )
                    
                    content = file_path.read_text(encoding="utf-8-sig", errors="ignore")
                    chunks = chunk_text(content)
                    embeddings = embed_texts(chunks)
                    points = [
                        PointStruct(id=max_id + j, vector=emb,
                                    payload={"file_path": rel_path, "chunk": j, "text": chunk})
                        for j, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ]
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    max_id += len(chunks)
                    
                    if status == "new":
                        added += 1
                    else:
                        updated += 1
                    
                    elapsed = time.time() - start_time
                    indexing_status.update({
                        "processed_files": i + 1,
                        "total_chunks": added + updated,
                        "elapsed": round(elapsed, 1),
                        "speed": round((i + 1) / elapsed, 1) if elapsed > 0 else 0,
                        "eta": round((len(to_process) - i - 1) * elapsed / (i + 1), 0) if i > 0 else None
                    })
                except Exception as e:
                    print(f"Ошибка {rel_path}: {e}")
            
            # Удалённые файлы
            for rel_path in old_hashes:
                if rel_path not in current_files:
                    try:
                        client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=Filter(
                                must=[FieldCondition(key="file_path", match=MatchValue(value=rel_path))]
                            )
                        )
                        deleted += 1
                    except Exception as e:
                        print(f"Ошибка удаления {rel_path}: {e}")
            
            save_hashes(new_hashes)
            indexing_status["running"] = False
            return {"added": added, "updated": updated, "deleted": deleted, "time_sec": round(time.time() - start_time, 1)}
        
        except Exception as e:
            indexing_status["running"] = False
            indexing_status["error"] = str(e)
            raise

    def normalize(text: str) -> str:
        words = re.findall(r"[а-яёa-z0-9]+", text.lower())
        return " ".join(morph.parse(w)[0].normal_form for w in words)

    def hybrid_search(query: str, top_k: int = 5):
        query_lower = query.lower()
        query_normalized = normalize(query)
        
        query_embedding = embed_model.encode(["query: " + query]).tolist()[0]
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k * 2
        ).points
        
        results = []
        seen_files = set()
        
        for hit in search_results:
            file_path = hit.payload["file_path"]
            text = hit.payload.get("text", "")
            score = hit.score
            
            if query_lower in file_path.lower():
                score += 1.0
                match_type = "filename"
            elif query_lower in text.lower():
                score += 0.5
                match_type = "text"
            elif query_normalized in normalize(file_path):
                score += 0.8
                match_type = "filename_stem"
            elif query_normalized in normalize(text):
                score += 0.3
                match_type = "text_stem"
            else:
                match_type = "semantic"
            
            if file_path not in seen_files:
                seen_files.add(file_path)
                results.append({"file": file_path, "text": text[:500], "score": score, "match": match_type})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # FastAPI
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>1C Code Search</title>
<style>
body{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px}
input{width:60%;padding:10px;font-size:16px}
button{padding:10px 15px;font-size:14px;margin:2px;cursor:pointer}
.result{border:1px solid #ddd;margin:10px 0;padding:15px;border-radius:5px}
.file{color:#666;font-size:12px}.score{color:#090;font-weight:bold}
pre{background:#f5f5f5;padding:10px;overflow-x:auto;font-size:13px}
.btn-full{background:#dc3545;color:#fff;border:none}
.btn-inc{background:#28a745;color:#fff;border:none}
#status{margin:10px 0;padding:15px;border-radius:5px;display:none}
.progress-bar{background:#e9ecef;border-radius:5px;height:20px;margin:10px 0}
.progress-fill{background:#007bff;height:100%;border-radius:5px;transition:width 0.3s}
.stats{font-size:13px;color:#666;margin-top:5px}
</style></head>
<body><h1>🔍 1C Code Search</h1>
<form onsubmit="search();return false"><input id="q" placeholder="Поиск по коду 1С..." autofocus>
<button>Найти</button></form>
<div style="margin-top:15px">
<button class="btn-inc" onclick="reindex('incremental')">🔄 Обновить изменённые</button>
<button class="btn-full" onclick="reindex('full')">⚠️ Полная переиндексация</button>
</div>
<div id="status"></div>
<div id="results"></div>
<script>
let pollInterval = null;

async function search(){
    const q=document.getElementById('q').value;
    const r=document.getElementById('results');
    r.innerHTML='Поиск...';
    const res=await fetch('/search?q='+encodeURIComponent(q));
    const data=await res.json();
    r.innerHTML=data.map(d=>`<div class="result"><span class="score">${d.score.toFixed(3)}</span> <span class="match">[${d.match}]</span>
<div class="file">${d.file}</div><pre>${d.text.replace(/</g,'&lt;')}</pre></div>`).join('')||'Ничего не найдено';
}

async function reindex(mode){
    const msg=mode==='full'?'Полная переиндексация удалит весь индекс. Продолжить?':'Обновить изменённые файлы?';
    if(!confirm(msg))return;
    
    fetch('/reindex/'+mode,{method:'POST'});
    startPolling();
}

function startPolling(){
    const s=document.getElementById('status');
    s.style.display='block';
    if(pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(updateProgress, 500);
    updateProgress();
}

async function updateProgress(){
    const s=document.getElementById('status');
    try {
        const res = await fetch('/indexing-status');
        const data = await res.json();
        
        if(!data.running && data.total_files === 0){
            s.style.display='none';
            if(pollInterval){clearInterval(pollInterval);pollInterval=null;}
            return;
        }
        
        const pct = data.total_files > 0 ? Math.round(data.processed_files / data.total_files * 100) : 0;
        const mode = data.mode === 'full' ? 'Полная индексация' : 'Обновление';
        const eta = data.eta ? `~${Math.round(data.eta)}с` : '...';
        
        if(data.running){
            s.style.background='#fff3cd';
            s.innerHTML = `<b>⏳ ${mode}</b>
<div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
<div class="stats">
    Файлов: ${data.processed_files}/${data.total_files} (${pct}%) | 
    Чанков: ${data.total_chunks} | 
    Скорость: ${data.speed}/с | 
    Осталось: ${eta}
</div>
<div class="stats" style="font-size:11px;color:#999">${data.last_file}</div>`;
        } else if(data.error){
            s.style.background='#f8d7da';
            s.innerHTML=`❌ Ошибка: ${data.error}`;
            if(pollInterval){clearInterval(pollInterval);pollInterval=null;}
        } else {
            s.style.background='#d4edda';
            s.innerHTML=`✅ Готово! Файлов: ${data.processed_files}, чанков: ${data.total_chunks}, время: ${data.elapsed}с`;
            if(pollInterval){clearInterval(pollInterval);pollInterval=null;}
        }
    } catch(e) {
        console.error(e);
    }
}

// Проверяем статус при загрузке страницы
updateProgress();
</script></body></html>"""

    @app.get("/indexing-status")
    async def get_indexing_status():
        return indexing_status

    @app.post("/mcp/tools/1c_search")
    async def mcp_search(request: dict) -> dict[str, Any]:
        query = request["params"]["query"]
        results = hybrid_search(query)
        return {"content": [{"type": "text", "text": str(results)}], "isError": False}

    @app.get("/search")
    async def search_get(q: str) -> list[dict[str, Any]]:
        return hybrid_search(q)

    @app.get("/health")
    async def health():
        return {"status": "ok", "source": args.source, "chunks": get_collection_count(), "indexing": indexing_status["running"]}

    @app.post("/reindex/full")
    async def reindex_full_endpoint():
        if indexing_status["running"]:
            return {"error": "Индексация уже запущена"}
        threading.Thread(target=full_reindex, args=(args.source,), daemon=True).start()
        return {"started": True, "mode": "full"}

    @app.post("/reindex/incremental")
    async def reindex_incremental_endpoint():
        if indexing_status["running"]:
            return {"error": "Индексация уже запущена"}
        threading.Thread(target=incremental_reindex, args=(args.source,), daemon=True).start()
        return {"started": True, "mode": "incremental"}

    # Запуск
    count = get_collection_count()
    if count == 0:
        print("Индекс пуст, запускаю фоновую индексацию...")
        threading.Thread(target=full_reindex, args=(args.source,), daemon=True).start()
    else:
        print(f"Индекс загружен: {count} чанков")

    print(f"Сервер на http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
