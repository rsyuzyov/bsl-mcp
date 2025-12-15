"""Движок индексации."""
import threading
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from ..config import VECTOR_SIZE, COLLECTION_NAME, indexing_status, model_state
from ..utils import format_time, format_duration
from .hasher import file_hash, load_hashes, save_hashes
from .chunker import chunk_text


class IndexEngine:
    """Движок индексации 1С выгрузки."""

    def __init__(self, source_dir: str, index_dir: str):
        self.source_dir = Path(source_dir)
        self.index_dir = Path(index_dir)
        self.meta_file = self.index_dir / "file_hashes.json"
        self.qdrant_path = self.index_dir / "qdrant"
        
        self.index_dir.mkdir(exist_ok=True)
        self.client = QdrantClient(path=str(self.qdrant_path))
        self._ensure_collection()

    def _ensure_collection(self):
        """Создать коллекцию если не существует."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )

    def get_all_files(self) -> list[Path]:
        """Получить все XML и BSL файлы."""
        return list(self.source_dir.rglob("*.xml")) + list(self.source_dir.rglob("*.bsl"))

    def get_collection_count(self) -> int:
        """Количество точек в коллекции."""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return info.points_count
        except:
            return 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги текстов."""
        if model_state.model is None:
            raise RuntimeError("Модель ещё загружается")
        prefixed = ["passage: " + t for t in texts]
        embeddings = model_state.model.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    def quick_check_changes(self) -> tuple[bool, int, int, int]:
        """Быстрая проверка изменений по mtime+size."""
        old_hashes = load_hashes(self.meta_file)
        if not old_hashes:
            return False, 0, 0, 0
        
        files = self.get_all_files()
        current_files = {}
        added, changed = 0, 0
        
        for file_path in files:
            rel_path = str(file_path.relative_to(self.source_dir))
            stat = file_path.stat()
            quick_key = f"{stat.st_mtime_ns}:{stat.st_size}"
            current_files[rel_path] = quick_key
            
            if rel_path not in old_hashes:
                added += 1
            elif not old_hashes[rel_path].startswith(quick_key.split(":")[0][:10]):
                changed += 1
        
        deleted = len(set(old_hashes.keys()) - set(current_files.keys()))
        has_changes = added > 0 or changed > 0 or deleted > 0
        return has_changes, added, changed, deleted


    def _print_progress(self):
        """Вывод прогресса в консоль."""
        s = indexing_status
        if s.running and s.total_files > 0:
            pct = round(s.processed_files / s.total_files * 100) if s.total_files > 0 else 0
            started = format_time(s.started_at) if s.started_at else "?"
            elapsed_sec = time.time() - s.started_at if s.started_at else 0
            elapsed = format_duration(elapsed_sec)
            eta_time = format_time(s.eta_time) if s.eta_time else "..."
            speed = s.total_chunks / elapsed_sec if elapsed_sec > 0 else 0
            print(f"[{pct}%] {s.processed_files}/{s.total_files} | {s.total_chunks} чанков | {speed:.1f}/с | начало: {started} | прошло: {elapsed} | конец: {eta_time}", flush=True)

    def full_reindex(self) -> dict:
        """Полная переиндексация."""
        start_time = time.time()
        indexing_status.reset("full")
        
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            
            files = self.get_all_files()
            indexing_status.total_files = len(files)
            print(f"[Индексация] Начало: {format_time(start_time)} | Файлов: {len(files)}", flush=True)
            
            hashes = {}
            total_chunks = 0
            batch_points = []
            batch_size = 100
            point_id = 0
            
            progress_stop = threading.Event()
            def progress_printer():
                while not progress_stop.is_set():
                    time.sleep(5)
                    if indexing_status.running:
                        self._print_progress()
            progress_thread = threading.Thread(target=progress_printer, daemon=True)
            progress_thread.start()
            
            def flush_batch():
                nonlocal batch_points
                if not batch_points:
                    return
                texts = [p["text"] for p in batch_points]
                embeddings = self.embed_texts(texts)
                points = [
                    PointStruct(id=p["id"], vector=emb, payload={**p["payload"], "text": p["text"]})
                    for p, emb in zip(batch_points, embeddings)
                ]
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                batch_points = []

            for i, file_path in enumerate(files):
                try:
                    content = file_path.read_text(encoding="utf-8-sig", errors="ignore")
                    rel_path = str(file_path.relative_to(self.source_dir))
                    hashes[rel_path] = file_hash(file_path)
                    indexing_status.last_file = rel_path
                    
                    chunks = chunk_text(content)
                    for j, chunk in enumerate(chunks):
                        batch_points.append({
                            "id": point_id, "text": chunk,
                            "payload": {"file_path": rel_path, "chunk": j}
                        })
                        point_id += 1
                        total_chunks += 1
                        indexing_status.total_chunks = total_chunks
                        
                        if len(batch_points) >= batch_size:
                            indexing_status.status_detail = "embedding..."
                            flush_batch()
                            indexing_status.status_detail = "reading..."
                    
                    elapsed = time.time() - start_time
                    eta_sec = round((len(files) - i - 1) * elapsed / (i + 1), 0) if i > 0 else None
                    indexing_status.processed_files = i + 1
                    indexing_status.elapsed = round(elapsed, 1)
                    indexing_status.speed = round(total_chunks / elapsed, 1) if elapsed > 0 else 0
                    indexing_status.eta = eta_sec
                    indexing_status.eta_time = time.time() + eta_sec if eta_sec else None
                except Exception as e:
                    print(f"Ошибка {file_path}: {e}", flush=True)
            
            flush_batch()
            progress_stop.set()
            
            save_hashes(self.meta_file, hashes)
            elapsed_total = time.time() - start_time
            print(f"[Индексация] Готово: {format_time(time.time())} | Файлов: {len(files)} | Чанков: {total_chunks} | Время: {format_duration(elapsed_total)}", flush=True)
            indexing_status.running = False
            return {"files": len(files), "chunks": total_chunks, "time_sec": round(elapsed_total, 1)}
        
        except Exception as e:
            indexing_status.running = False
            indexing_status.error = str(e)
            raise


    def incremental_reindex(self) -> dict:
        """Инкрементальная индексация."""
        start_time = time.time()
        indexing_status.reset("incremental")
        
        try:
            files = self.get_all_files()
            old_hashes = load_hashes(self.meta_file)
            new_hashes = {}
            
            added, updated, deleted = 0, 0, 0
            current_files = set()
            max_id = self.get_collection_count()
            
            to_process = []
            for file_path in files:
                rel_path = str(file_path.relative_to(self.source_dir))
                current_files.add(rel_path)
                h = file_hash(file_path)
                new_hashes[rel_path] = h
                if rel_path not in old_hashes:
                    to_process.append((file_path, rel_path, "new"))
                elif old_hashes[rel_path] != h:
                    to_process.append((file_path, rel_path, "changed"))
            
            indexing_status.total_files = len(to_process)
            
            progress_stop = threading.Event()
            def progress_printer():
                while not progress_stop.is_set():
                    time.sleep(5)
                    if indexing_status.running:
                        self._print_progress()
            progress_thread = threading.Thread(target=progress_printer, daemon=True)
            progress_thread.start()
            
            if to_process:
                print(f"[Обновление] Начало: {format_time(start_time)} | Файлов к обработке: {len(to_process)}", flush=True)
            
            for i, (file_path, rel_path, status) in enumerate(to_process):
                try:
                    indexing_status.last_file = rel_path
                    
                    if status == "changed":
                        self.client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=Filter(
                                must=[FieldCondition(key="file_path", match=MatchValue(value=rel_path))]
                            )
                        )
                    
                    content = file_path.read_text(encoding="utf-8-sig", errors="ignore")
                    chunks = chunk_text(content)
                    embeddings = self.embed_texts(chunks)
                    points = [
                        PointStruct(id=max_id + j, vector=emb,
                                    payload={"file_path": rel_path, "chunk": j, "text": chunk})
                        for j, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ]
                    self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                    max_id += len(chunks)
                    
                    if status == "new":
                        added += 1
                    else:
                        updated += 1
                    
                    elapsed = time.time() - start_time
                    eta_sec = round((len(to_process) - i - 1) * elapsed / (i + 1), 0) if i > 0 else None
                    indexing_status.processed_files = i + 1
                    indexing_status.total_chunks = added + updated
                    indexing_status.elapsed = round(elapsed, 1)
                    indexing_status.speed = round((i + 1) / elapsed, 1) if elapsed > 0 else 0
                    indexing_status.eta = eta_sec
                    indexing_status.eta_time = time.time() + eta_sec if eta_sec else None
                except Exception as e:
                    print(f"Ошибка {rel_path}: {e}", flush=True)
            
            for rel_path in old_hashes:
                if rel_path not in current_files:
                    try:
                        self.client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=Filter(
                                must=[FieldCondition(key="file_path", match=MatchValue(value=rel_path))]
                            )
                        )
                        deleted += 1
                    except Exception as e:
                        print(f"Ошибка удаления {rel_path}: {e}")
            
            progress_stop.set()
            save_hashes(self.meta_file, new_hashes)
            elapsed_total = time.time() - start_time
            if added or updated or deleted:
                print(f"[Обновление] Готово: {format_time(time.time())} | +{added} ~{updated} -{deleted} | Время: {format_duration(elapsed_total)}", flush=True)
            indexing_status.running = False
            return {"added": added, "updated": updated, "deleted": deleted, "time_sec": round(elapsed_total, 1)}
        
        except Exception as e:
            indexing_status.running = False
            indexing_status.error = str(e)
            raise
