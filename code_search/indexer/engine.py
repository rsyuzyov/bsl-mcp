"""Движок индексации."""
import threading
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import uuid
from ..config import VECTOR_SIZE, IndexingStatus
from ..utils import format_time, format_duration
from .hasher import file_hash, load_hashes, save_hashes
from .chunker import chunk_text
from ..model_manager import ModelManager
from ..vector_db import get_vector_db, VectorPoint
from ..logger import get_logger


# Настройки производительности
BATCH_SIZE = 500  # Возвращаем нормальный размер батча
READ_WORKERS = 1  # 1 поток чтения, чтобы отдать CPU под ONNX
EMBED_BATCH_SIZE = 64


class IndexEngine:
    """Движок индексации 1С выгрузки для конкретной ИБ."""

    def __init__(self, source_dir: str, index_dir: str, collection_name: str, embedding_model_name: str, embedding_device: str = "cpu", vector_db_type: str = "qdrant", vector_size: int = VECTOR_SIZE):
        self.source_dir = Path(source_dir)
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.vector_db_type = vector_db_type
        self.vector_size = vector_size
        
        self.meta_file = self.index_dir / "file_hashes.json"
        
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = get_logger(f"idx.{collection_name}")
        
        t0 = time.time()
        self.logger.info(f"Подключение к vector_db ({vector_db_type})...")
        self.db = get_vector_db(self.vector_db_type, str(self.index_dir))
        dt = time.time() - t0
        self.logger.info(f"Подключено к БД за {dt:.2f} сек")
        if dt > 5.0:
             self.logger.warning("Долгое подключение к БД! Возможно, проблема с сетью или телеметрией Qdrant.")
        
        t1 = time.time()
        self._ensure_collection()
        self.logger.info(f"Коллекция проверена за {time.time() - t1:.2f} сек")
        
        self.model_manager = ModelManager()
        self.batch_counter = 0

    def _ensure_collection(self):
        """Создать коллекцию если не существует."""
        self.db.create_collection(self.collection_name, self.vector_size)

    def close(self):
        """Закрыть соединение с БД."""
        if self.db:
            self.db.close()

    def get_all_files(self) -> list[Path]:
        """Получить все XML и BSL файлы."""
        if not self.source_dir.exists():
            return []
        return list(self.source_dir.rglob("*.xml")) + list(self.source_dir.rglob("*.bsl"))

    def get_collection_count(self) -> int:
        """Количество точек в коллекции."""
        return self.db.count(self.collection_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги текстов."""
        model_info = self.model_manager.get_model(self.embedding_model_name, self.embedding_device)
        if model_info.loading:
            # Wait or raise? For background process it's better to wait a bit or raise error
            # But here we assume it's called when we permit indexing.
            # Let's simple wait for a moment or raise exception
            start_wait = time.time()
            while model_info.loading and time.time() - start_wait < 60:
                time.sleep(1)
            
        if model_info.loading:
             raise RuntimeError(f"Модель {self.embedding_model_name} ещё загружается")
        if model_info.error:
             raise RuntimeError(f"Ошибка модели: {model_info.error}")

        prefixed = ["passage: " + t for t in texts]
        embeddings = model_info.model.encode(
            prefixed, 
            show_progress_bar=False,
            batch_size=EMBED_BATCH_SIZE
        )
        return embeddings.tolist()

    def quick_check_changes(self) -> tuple[bool, int, int, int]:
        """Быстрая проверка изменений по mtime+size."""
        old_hashes = load_hashes(self.meta_file)
        # If no hashes, and no files in DB -> no changes (technically empty), but usually means full reindex needed.
        # If DB has data but no hashes -> inconsistency, treat as changes needed.
        # For simplicity, if no hashes, we say NO changes (unless we want to trigger initial index elsewhere).
        # But wait, helper 'startup_indexing' logic relied on this.
        # Let's say: if old_hashes is empty, check if we have files.
        
        files = self.get_all_files()
        if not files and not old_hashes:
             return False, 0, 0, 0
             
        current_files = {}
        added, changed = 0, 0
        
        for file_path in files:
            try:
                rel_path = str(file_path.relative_to(self.source_dir))
                stat = file_path.stat()
                quick_key = f"{stat.st_mtime_ns}:{stat.st_size}"
                current_files[rel_path] = quick_key
                
                if rel_path not in old_hashes:
                    added += 1
                elif not old_hashes[rel_path].startswith(quick_key.split(":")[0][:10]):
                    # Check comments in original file about logic issues.
                    pass
            except Exception:
                continue
                
        deleted = len(set(old_hashes.keys()) - set(current_files.keys()))
        
        return (added > 0 or deleted > 0), added, 0, deleted


    def _print_progress(self, status: IndexingStatus):
        """Вывод прогресса в консоль."""
        if status.running and status.files_to_index > 0:
            self.logger.info(status.format_console(self.get_collection_count()))

    def _read_file(self, file_path: Path) -> tuple[Path, str, str, list[str]] | None:
        """Читает файл и возвращает (path, rel_path, hash, chunks)."""
        try:
            content_bytes = file_path.read_bytes()
            h = hashlib.md5(content_bytes).hexdigest()
            content = content_bytes.decode("utf-8-sig", errors="ignore")
            
            rel_path = str(file_path.relative_to(self.source_dir))
            chunks = chunk_text(content)
            return (file_path, rel_path, h, chunks)
        except Exception as e:
            self.logger.error(f"Ошибка чтения {file_path}: {e}")
            return None

    def full_reindex(self, status: IndexingStatus = None) -> dict:
        """Полная переиндексация с параллельным чтением."""
        if status is None:
            status = IndexingStatus()
            
        start_time = time.time()
        status.reset("full")
        self.logger.info(f"Запуск полной переиндексации...")
        
        try:
            self.db.delete_collection(self.collection_name)
            self.db.create_collection(self.collection_name, self.vector_size)
            
            files = self.get_all_files()
            status.files_to_index = len(files)
            status.chunks_in_db_start = 0 
            
            total_bytes = sum(f.stat().st_size for f in files)
            status.chunks_to_index_est = total_bytes // 900
            
            self.logger.info(f"[Full Index] Start: {format_time(start_time)} | Files: {len(files)}")
            
            hashes = {}
            chunks_indexed = 0
            batch_points = []
            files_processed = 0
            
            progress_stop = threading.Event()
            def progress_printer():
                while not progress_stop.is_set():
                    time.sleep(60)
                    if status.running:
                        self._print_progress(status)
            progress_thread = threading.Thread(target=progress_printer, daemon=True)
            progress_thread.start()
            
            def flush_batch():
                nonlocal batch_points
                if not batch_points:
                    return
                
                t0 = time.time()
                status.status_detail = f"embedding {len(batch_points)}..."
                texts = [p["text"] for p in batch_points]
                
                t1 = time.time()
                embeddings = self.embed_texts(texts)
                t2 = time.time()
                
                points = [
                    VectorPoint(
                        id=str(uuid.uuid4()), 
                        vector=emb, 
                        payload={**p["payload"]}, 
                        text=p["text"]
                    )
                    for p, emb in zip(batch_points, embeddings)
                ]
                
                t3 = time.time()
                status.status_detail = f"upsert {len(points)}..."
                self.db.upsert(self.collection_name, points)
                t4 = time.time()
                
                status.chunks_in_db += len(batch_points)
                batch_points = []
                status.status_detail = "reading..."
                
                embed_time = t2 - t1
                upsert_time = t4 - t3
                total_time = t4 - t0
                per_item = (embed_time / len(texts) * 1000) if texts else 0
                
                self.logger.info(f"Batch {len(texts)}: Full={total_time:.2f}s | Embed={embed_time:.2f}s ({per_item:.1f}ms/item) | Upsert={upsert_time:.2f}s")

            with ThreadPoolExecutor(max_workers=READ_WORKERS) as executor:
                futures = {executor.submit(self._read_file, f): f for f in files}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        files_processed += 1
                        continue
                    
                    file_path, rel_path, h, chunks = result
                    hashes[rel_path] = h
                    status.last_file = rel_path
                    
                    for j, chunk in enumerate(chunks):
                        batch_points.append({
                            "text": chunk,
                            "payload": {"file_path": rel_path, "chunk": j}
                        })
                        chunks_indexed += 1
                    
                    status.chunks_indexed = chunks_indexed
                    
                    if len(batch_points) >= BATCH_SIZE:
                        flush_batch()
                    
                    files_processed += 1
                    elapsed = time.time() - start_time
                    chunks_speed = chunks_indexed / elapsed if elapsed > 0 else 0
                    eta_sec = round((status.chunks_to_index_est - chunks_indexed) / chunks_speed, 0) if chunks_speed > 0 else None
                    status.files_indexed = files_processed
                    status.elapsed = round(elapsed, 1)
                    status.chunks_speed = round(chunks_indexed / elapsed, 1) if elapsed > 0 else 0
                    status.eta = eta_sec
                    status.eta_time = time.time() + eta_sec if eta_sec else None
            
            flush_batch()
            progress_stop.set()
            
            save_hashes(self.meta_file, hashes)
            elapsed_total = time.time() - start_time
            self.logger.info(f"[Full Index] Done in {format_duration(elapsed_total)}")
            status.running = False
            return {"files": len(files), "chunks": chunks_indexed, "time_sec": round(elapsed_total, 1)}
        
        except Exception as e:
            status.running = False
            status.error = str(e)
            self.logger.error(f"Full Index Error: {e}", exc_info=True)
            raise


    def incremental_reindex(self, status: IndexingStatus = None) -> dict:
        """Инкрементальная индексация с батчингом."""
        if status is None:
            status = IndexingStatus()
            
        start_time = time.time()
        status.reset("incremental")
        
        try:
            files = self.get_all_files()
            old_hashes = load_hashes(self.meta_file)
            new_hashes = {}
            
            to_process = []
            
            # Identify changes: reading files is necessary for hash comparison
            # We can optimize by cheking mtime but that requires mtime storage.
            # Here we do full check: read all files, calculate hash, compare with old hash.
            # This is "heavy" incremental check.
            
            # Для progress reporting on discovery phase:
            status.status_detail = "Проверка модели..."
            
            # Предзагрузка модели, чтобы отобразить статус в вебе
            self.logger.info("[Incremental] Проверка модели эмбеддингов...")
            model_info = self.model_manager.get_model(self.embedding_model_name, self.embedding_device)
            if model_info.loading:
                 status.status_detail = "Загрузка нейросети (ML)..."
                 wait_start = time.time()
                 while model_info.loading and time.time() - wait_start < 120:
                     time.sleep(1)
            
            status.status_detail = "Сканирование файлов..."
            self.logger.info(f"[Incremental] Сканирование {len(files)} файлов для изменений...")

            current_files_set = set()
            processed_scan = 0
            
            # Use threading for faster IO/Hashing
            with ThreadPoolExecutor(max_workers=READ_WORKERS) as executor:
                # Helper to process one file
                def process_file_hash(f_path):
                    try:
                        r_path = str(f_path.relative_to(self.source_dir))
                        # Read bytes for hashing
                        content_bytes = f_path.read_bytes()
                        h = hashlib.md5(content_bytes).hexdigest()
                        return (f_path, r_path, h)
                    except Exception:
                        return None

                futures = {executor.submit(process_file_hash, f): f for f in files}
                
                # Monitor scanning progress
                last_print = time.time()
                
                for future in as_completed(futures):
                    res = future.result()
                    processed_scan += 1
                    
                    if time.time() - last_print > 5:
                        self.logger.info(f"Scanned {processed_scan}/{len(files)} files...")
                        last_print = time.time()
                    
                    if res is None:
                        continue
                        
                    f_p, r_p, f_h = res
                    current_files_set.add(r_p)
                    
                    new_hashes[r_p] = f_h
                    
                    if r_p not in old_hashes:
                        to_process.append((f_p, r_p, "new"))
                    elif old_hashes[r_p] != f_h:
                        to_process.append((f_p, r_p, "changed"))

            added = len([x for x in to_process if x[2] == "new"])
            updated = len([x for x in to_process if x[2] == "changed"])
            
            deleted_paths = set(old_hashes.keys()) - current_files_set
            deleted = len(deleted_paths)
            
            # Clear line after scanning done
            self.logger.info(f"Scanning done. Found: +{added} ~{updated} -{deleted}")
            
            if not added and not updated and not deleted:
                status.running = False
                return {"added": 0, "updated": 0, "deleted": 0}

            status.files_to_index = len(to_process) + len(deleted_paths) # rough count
            status.chunks_in_db_start = self.get_collection_count()
            
            # Use average estimate instead of reading all content
            # Avg module size ~10KB? Let's say 10 chunks per file conservatively
            status.chunks_to_index_est = len(to_process) * 10
            
            self.logger.info(f"[Incremental] Starting indexing...")

            progress_stop = threading.Event()
            def progress_printer():
                while not progress_stop.is_set():
                    time.sleep(60)
                    if status.running:
                        self._print_progress(status)
            progress_thread = threading.Thread(target=progress_printer, daemon=True)
            progress_thread.start()

            batch_points = []
            files_in_batch = []
            
            def flush_batch():
                nonlocal batch_points, files_in_batch, added, updated
                if not batch_points:
                    return
                
                t0 = time.time()
                status.status_detail = f"embedding {len(batch_points)}..."
                texts = [p["text"] for p in batch_points]
                
                t1 = time.time()
                embeddings = self.embed_texts(texts)
                t2 = time.time()
                
                points = [
                    VectorPoint(
                        id=str(uuid.uuid4()), 
                        vector=emb, 
                        payload={**p["payload"]},
                        text=p["text"]
                    )
                    for p, emb in zip(batch_points, embeddings)
                ]
                
                # Delete old chunks for changed files BEFORE upsert
                # (Only unique files in batch needed)
                changed_files_uniq = set(rel for rel, st, _ in files_in_batch if st == "changed")
                for rel_path in changed_files_uniq:
                    try:
                        self.db.delete_by_file_path(self.collection_name, rel_path)
                    except: 
                        pass # avoid crash on delete

                t3 = time.time()
                status.status_detail = f"upsert {len(points)}..."
                self.db.upsert(self.collection_name, points)
                t4 = time.time()
                
                status.chunks_in_db += len(batch_points)
                
                embed_time = t2 - t1
                upsert_time = t4 - t3
                total_time = t4 - t0
                per_item = (embed_time / len(texts) * 1000) if texts else 0
                
                self.batch_counter += 1
                if self.batch_counter % 3 == 0:
                    self.logger.info(f"Batch {len(texts)}: Full={total_time:.2f}s | Embed={embed_time:.2f}s ({per_item:.1f}ms/item) | Upsert={upsert_time:.2f}s")
                
                batch_points = []
                files_in_batch = []
                status.status_detail = "reading..."

            # Process new/updates
            try:
                for i, (file_path, rel_path, st) in enumerate(to_process):
                    if i % 1000 == 0:
                        self.logger.info(f"Processing file {i}: {rel_path}")

                    try:
                        content = file_path.read_text(encoding="utf-8-sig", errors="ignore")
                    except Exception as e:
                        self.logger.error(f"Error reading {rel_path}: {e}")
                        continue
                    
                    status.last_file = rel_path
                    chunks = chunk_text(content)
                    
                    for j, chunk in enumerate(chunks):
                        batch_points.append({
                            "text": chunk,
                            "payload": {"file_path": rel_path, "chunk": j}
                        })
                        if len(batch_points) >= BATCH_SIZE:
                            flush_batch()
                            
                    files_in_batch.append((rel_path, st, len(chunks)))
                    
                    status.chunks_indexed += len(chunks)
                    
                    status.files_indexed = i + 1
                    
                    # Estimate eta
                    elapsed = time.time() - start_time
                    if elapsed > 0 and status.files_indexed > 0:
                        files_speed = status.files_indexed / elapsed
                        files_left = len(to_process) - status.files_indexed
                        status.eta = round(files_left / files_speed, 0)
                        status.eta_time = time.time() + status.eta
                        status.chunks_speed = round(status.chunks_indexed / elapsed, 1)
                        status.elapsed = round(elapsed, 1)

                flush_batch()
                
                # Process deletes
                for rel_path in deleted_paths:
                    try:
                        self.db.delete_by_file_path(self.collection_name, rel_path)
                    except Exception:
                        pass
                
                save_hashes(self.meta_file, new_hashes)
                
            finally:
                progress_stop.set()
                progress_thread.join(timeout=1)
            
            status.running = False
            elapsed = time.time() - start_time
            self.logger.info(f"Indexing done in {round(elapsed, 1)}s")
            return {"added": added, "updated": updated, "deleted": deleted, "time_sec": round(elapsed, 1)}

        except Exception as e:
            status.running = False
            status.error = str(e)
            self.logger.error(f"Incremental Index Error: {e}", exc_info=True)
            raise
