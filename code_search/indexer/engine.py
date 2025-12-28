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
from ..metadata_utils import parse_1c_path, format_object_context


# Настройки производительности
BATCH_SIZE = 500  # Возвращаем нормальный размер батча
EMBED_BATCH_SIZE = 64


class IndexEngine:
    """Движок индексации 1С выгрузки для конкретной ИБ."""

    def __init__(self, source_dir: str, index_dir: str, collection_name: str, embedding_model_name: str, embedding_device: str = "cpu", vector_db_type: str = "qdrant", vector_size: int = VECTOR_SIZE, scan_workers: int = 4):
        self.source_dir = Path(source_dir)
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.vector_db_type = vector_db_type
        self.vector_size = vector_size
        self.scan_workers = scan_workers
        
        self.meta_file = self.index_dir / "file_hashes.json"
        self.reindex_flag = self.index_dir / "reindex_pending"
        
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = get_logger(f"idx.{collection_name}")
        
        # Проверяем флаг отложенного reindex
        if self.reindex_flag.exists():
            self.logger.info("Обнаружен флаг reindex_pending, очистка базы...")
            self._cleanup_qdrant_storage()
            self.reindex_flag.unlink(missing_ok=True)
            self.logger.info("Отложенный reindex выполнен")
        
        t0 = time.time()
        self.logger.info(f"Подключение к vector_db ({vector_db_type})...")
        self.db = get_vector_db(self.vector_db_type, str(self.index_dir))
        dt = time.time() - t0
        self.logger.info(f"Подключено к БД за {dt:.2f} сек")
        if dt > 5.0:
             self.logger.warning("Долгое подключение к БД! Возможно, проблема с сетью или телеметрией Qdrant.")
        
        t1 = time.time()
        self._ensure_collection()
        
        # Оптимизация WAL при старте — только если коллекция не пустая
        count = self.db.count(self.collection_name)
        if count > 0:
            self.db.optimize(self.collection_name)
            self.logger.info(f"Коллекция ({count} точек) оптимизирована за {time.time() - t1:.2f} сек")
        else:
            self.logger.info(f"Коллекция пустая, оптимизация пропущена")
        
        self.model_manager = ModelManager()
        self.batch_counter = 0

    def _ensure_collection(self):
        """Создать коллекцию если не существует."""
        self.db.create_collection(self.collection_name, self.vector_size)

    def _cleanup_qdrant_storage(self):
        """Очистка хранилища Qdrant (используется при отложенном reindex)."""
        import sqlite3
        qdrant_storage = self.index_dir / "qdrant"
        
        # Удаляем хэши
        if self.meta_file.exists():
            self.meta_file.unlink()
            self.logger.info("Хэши удалены")
        
        if not qdrant_storage.exists():
            return
        
        # Очищаем все sqlite файлы
        sqlite_files = list(qdrant_storage.rglob("*.sqlite"))
        for sqlite_file in sqlite_files:
            try:
                old_size = sqlite_file.stat().st_size
                conn = sqlite3.connect(str(sqlite_file))
                cursor = conn.cursor()
                
                # Получаем список таблиц
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # DROP всех таблиц (кроме системных)
                for table in tables:
                    if not table.startswith('sqlite_'):
                        try:
                            cursor.execute(f"DROP TABLE IF EXISTS {table}")
                        except Exception:
                            pass
                
                conn.commit()
                conn.execute("VACUUM")
                conn.close()
                
                new_size = sqlite_file.stat().st_size
                self.logger.info(f"Очистка {sqlite_file.name}: {old_size/1024/1024:.1f}MB -> {new_size/1024/1024:.1f}MB")
            except Exception as e:
                self.logger.warning(f"Ошибка очистки {sqlite_file}: {e}")

    def close(self):
        """Закрыть соединение с БД."""
        if self.db:
            self.db.close()

    def cleanup_orphans(self) -> int:
        """Удалить сироты — чанки, чьи файлы отсутствуют в file_hashes.json.
        
        Возвращает количество удалённых файлов.
        """
        hashes = load_hashes(self.meta_file)
        if not hashes:
            # Если хешей нет — сироты не определить (могут хранить легитимные данные)
            return 0
        
        known_files = set(hashes.keys())
        db_files = self.db.get_all_file_paths(self.collection_name)
        
        orphan_files = db_files - known_files
        if not orphan_files:
            return 0
        
        self.logger.info(f"Найдено {len(orphan_files)} сирот в БД, удаляю batch...")
        
        # Используем batch-удаление вместо поочерёдного — критически быстрее
        deleted = self.db.delete_by_file_paths(self.collection_name, orphan_files)
        
        self.logger.info(f"Удалено {deleted} точек-сирот из {len(orphan_files)} файлов")
        return len(orphan_files)

    def get_all_files(self) -> list[Path]:
        """Получить все BSL файлы (код). XML исключены из семантического поиска."""
        if not self.source_dir.exists():
            return []
        # Индексируем только .bsl файлы — это реальный код 1С
        # XML файлы содержат метаданные, их лучше обрабатывать отдельно
        return list(self.source_dir.rglob("*.bsl"))

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
        try:
            embeddings = model_info.model.encode(
                prefixed, 
                show_progress_bar=False,
                batch_size=EMBED_BATCH_SIZE
            )
        except RuntimeError as e:
            if "DirectML Native Crash" in str(e) and self.embedding_device != "cpu":
                self.logger.error("DirectML failed with native crash. Switching to CPU and retrying...")
                self.embedding_device = "cpu"
                return self.embed_texts(texts)
            raise e
        return embeddings.tolist()

    def quick_check_changes(self) -> tuple[bool, int, int, int]:
        """Быстрая проверка изменений по mtime+size."""
        old_hashes = load_hashes(self.meta_file)
        
        files = self.get_all_files()
        if not files and not old_hashes:
             return False, 0, 0, 0
             
        current_files = set()
        added, changed = 0, 0
        
        for file_path in files:
            try:
                rel_path = str(file_path.relative_to(self.source_dir))
                current_files.add(rel_path)
                
                if rel_path not in old_hashes:
                    added += 1
                else:
                    info = old_hashes[rel_path]
                    stat = file_path.stat()
                    # Check if mtime/size matches
                    if info.get("mtime") != stat.st_mtime_ns or info.get("size") != stat.st_size:
                         changed += 1
            except Exception:
                continue
                
        deleted = len(set(old_hashes.keys()) - current_files)
        
        return (added > 0 or deleted > 0 or changed > 0), added, changed, deleted


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

    def full_reindex(self, status: IndexingStatus = None, stop_event: threading.Event = None) -> dict:
        """Полная переиндексация с параллельным чтением."""
        if status is None:
            status = IndexingStatus()
            
        start_time = time.time()
        status.reset("full")
        self.logger.info(f"Запуск полной переиндексации...")
        
        try:
            # Создаём флаг ПЕРВЫМ — если что-то пойдёт не так, при следующем старте очистим
            self.reindex_flag.touch()
            self.logger.info("Флаг reindex_pending установлен")
            
            # Удаляем хэши
            if self.meta_file.exists():
                self.meta_file.unlink()
                self.logger.info("Хэши удалены")
            
            # Очистка коллекции и сжатие хранилища (VACUUM)
            t_clean = time.time()
            self.logger.info("Очистка и сжатие БД...")
            self.db.clear_and_compact(self.collection_name, self.vector_size)
            self.logger.info(f"БД очищена и сжата за {time.time() - t_clean:.2f} сек")
            
            # Флаг можно удалить — база очищена успешно
            self.reindex_flag.unlink(missing_ok=True)
            self.logger.info("Флаг reindex_pending снят")
            
            t_scan = time.time()
            self.logger.info("Сканирование файлов...")
            files = self.get_all_files()
            self.logger.info(f"Найдено {len(files)} файлов за {time.time() - t_scan:.2f} сек")
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

            with ThreadPoolExecutor(max_workers=self.scan_workers) as executor:
                futures = {executor.submit(self._read_file, f): f for f in files}
                
                for future in as_completed(futures):
                    if stop_event and stop_event.is_set():
                        for f in futures:
                             f.cancel()
                        break
                    
                    result = future.result()
                    if result is None:
                        files_processed += 1
                        continue
                    
                    file_path, rel_path, h, chunks = result
                    hashes[rel_path] = h
                    status.last_file = rel_path
                    
                    # Извлекаем метаданные объекта из пути
                    obj_meta = parse_1c_path(rel_path)
                    
                    for j, chunk in enumerate(chunks):
                        payload = {"file_path": rel_path, "chunk": j}
                        if obj_meta:
                            payload["object_type"] = obj_meta.object_type
                            payload["object_type_en"] = obj_meta.object_type_en
                            payload["object_name"] = obj_meta.object_name
                            payload["module_type"] = obj_meta.module_type
                            if obj_meta.form_name:
                                payload["form_name"] = obj_meta.form_name
                        
                        batch_points.append({
                            "text": chunk,
                            "payload": payload
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
            
            # Оптимизация после индексации
            self.logger.info("Запуск оптимизации (сжатие сегментов, очистка WAL)...")
            self.db.optimize(self.collection_name)
            
            elapsed_total = time.time() - start_time
            self.logger.info(f"[Full Index] Done in {format_duration(elapsed_total)}")
            status.running = False
            return {"files": len(files), "chunks": chunks_indexed, "time_sec": round(elapsed_total, 1)}
        
        except Exception as e:
            status.running = False
            status.error = str(e)
            self.logger.error(f"Full Index Error: {e}", exc_info=True)
            raise


    def incremental_reindex(self, status: IndexingStatus = None, stop_event: threading.Event = None) -> dict:
        """Инкрементальная индексация с батчингом."""
        if status is None:
            status = IndexingStatus()
            
        start_time = time.time()
        status.reset("incremental")
        
        try:
            # Очистка сирот перед началом индексации
            status.status_detail = "Очистка сирот..."
            orphans_deleted = self.cleanup_orphans()
            if orphans_deleted:
                self.logger.info(f"Очищено {orphans_deleted} файлов-сирот")
            
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
            with ThreadPoolExecutor(max_workers=self.scan_workers) as executor:
                # Helper to process one file
                def process_file_hash(f_path):
                    try:
                        r_path = str(f_path.relative_to(self.source_dir))
                        stat = f_path.stat()
                        mtime = stat.st_mtime_ns
                        size = stat.st_size
                        
                        # Optimization: check mtime+size
                        if r_path in old_hashes:
                            old = old_hashes[r_path]
                            # load_hashes normalizes to dict, safely check
                            if old.get("mtime") == mtime and old.get("size") == size:
                                return (f_path, r_path, old["hash"], mtime, size, False)

                        # Read bytes for hashing
                        content_bytes = f_path.read_bytes()
                        h = hashlib.md5(content_bytes).hexdigest()
                        return (f_path, r_path, h, mtime, size, True)
                    except Exception:
                        return None

                futures = {executor.submit(process_file_hash, f): f for f in files}
                
                # Monitor scanning progress
                last_print = time.time()
                
                for future in as_completed(futures):
                    if stop_event and stop_event.is_set():
                        for f in futures:
                             f.cancel()
                        break
                    
                    res = future.result()
                    processed_scan += 1
                    
                    if time.time() - last_print > 5:
                        self.logger.info(f"Scanned {processed_scan}/{len(files)} files...")
                        last_print = time.time()
                    
                    if res is None:
                        continue
                        
                    f_p, r_p, f_h, mtime, size, was_read = res
                    current_files_set.add(r_p)
                    
                    new_hashes[r_p] = {"hash": f_h, "mtime": mtime, "size": size}
                    
                    if r_p not in old_hashes:
                        to_process.append((f_p, r_p, "new"))
                    elif was_read:
                        # If read, check if hash changed
                        if old_hashes[r_p]["hash"] != f_h:
                            to_process.append((f_p, r_p, "changed"))

            added = len([x for x in to_process if x[2] == "new"])
            updated = len([x for x in to_process if x[2] == "changed"])
            
            deleted_paths = set(old_hashes.keys()) - current_files_set
            deleted = len(deleted_paths)
            
            # Clear line after scanning done
            self.logger.info(f"Scanning done. Found: +{added} ~{updated} -{deleted}")
            
            if stop_event and stop_event.is_set():
                 self.logger.info("Indexing cancelled.")
                 status.running = False
                 return {}
            
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
            indexed_hashes = dict(old_hashes)  # Начинаем с существующих хешей
            
            def flush_batch():
                nonlocal batch_points, files_in_batch, added, updated, indexed_hashes
                if not batch_points:
                    return
                # Не записываем если получен сигнал остановки
                if stop_event and stop_event.is_set():
                    batch_points = []
                    files_in_batch = []
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
                
                # Удаление чанков теперь происходит ПЕРЕД обработкой файла (см. ниже)

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
                
                # Сохраняем хеши успешно проиндексированных файлов
                for rel_path, st, _ in files_in_batch:
                    if rel_path in new_hashes:
                        indexed_hashes[rel_path] = new_hashes[rel_path]
                
                # Периодически сохраняем хеши — чтобы прерванная индексация продолжилась с места остановки
                save_hashes(self.meta_file, indexed_hashes)
                
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
                    if stop_event and stop_event.is_set():
                        break
                    
                    # Удаляем старые чанки ПЕРЕД добавлением новых — защита от дубликатов
                    try:
                        self.db.delete_by_file_path(self.collection_name, rel_path)
                    except Exception:
                        pass
                        
                    chunks = chunk_text(content)
                    
                    # Добавляем файл в батч ПЕРЕД обработкой чанков!
                    # Иначе при flush_batch хэш файла не сохранится
                    files_in_batch.append((rel_path, st, len(chunks)))
                    
                    # Извлекаем метаданные объекта из пути
                    obj_meta = parse_1c_path(rel_path)
                    
                    for j, chunk in enumerate(chunks):
                        payload = {"file_path": rel_path, "chunk": j}
                        if obj_meta:
                            payload["object_type"] = obj_meta.object_type
                            payload["object_type_en"] = obj_meta.object_type_en
                            payload["object_name"] = obj_meta.object_name
                            payload["module_type"] = obj_meta.module_type
                            if obj_meta.form_name:
                                payload["form_name"] = obj_meta.form_name
                        
                        batch_points.append({
                            "text": chunk,
                            "payload": payload
                        })
                        if len(batch_points) >= BATCH_SIZE:
                            flush_batch()
                    
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
                    # Удаляем хеш удалённого файла
                    indexed_hashes.pop(rel_path, None)
                
                # Сохраняем только если индексация не была прервана
                if not (stop_event and stop_event.is_set()):
                    save_hashes(self.meta_file, indexed_hashes)
                else:
                    self.logger.info("Индексация прервана, хеши не сохранены")
                
            finally:
                progress_stop.set()
                progress_thread.join(timeout=1)
            
            status.running = False
            elapsed = time.time() - start_time
            
            # Оптимизация после индексации (если были изменения)
            if added or updated or deleted:
                self.logger.info("Запуск оптимизации (сжатие сегментов, очистка WAL)...")
                self.db.optimize(self.collection_name)
            
            self.logger.info(f"Indexing done in {round(elapsed, 1)}s")
            return {"added": added, "updated": updated, "deleted": deleted, "time_sec": round(elapsed, 1)}

        except Exception as e:
            status.running = False
            status.error = str(e)
            self.logger.error(f"Incremental Index Error: {e}", exc_info=True)
            raise
