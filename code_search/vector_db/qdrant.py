from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .base import VectorDB, VectorPoint

class QdrantAdapter(VectorDB):
    def __init__(self, path: str):
        self.path = Path(path) / "qdrant"
        self.path.mkdir(parents=True, exist_ok=True)
        import os
        from ..logger import get_logger
        log = get_logger("idx.qdrant")
        telemetry = os.environ.get("QDRANT_TELEMETRY_DISABLED", "NOT_SET")
        log.info(f"Инициализация QdrantClient (telemetry={telemetry})...")
        try:
             # timeout=5.0 ограничивает время попыток подключения (если оно происходит)
             # check_compatibility=False отключает лишние проверки версий
             self.client = QdrantClient(path=str(self.path), timeout=5.0, check_compatibility=False)
        except TypeError:
             # Если версия старая и не поддерживает аргументы
             self.client = QdrantClient(path=str(self.path))

    def create_collection(self, name: str, vector_size: int):
        collections = [c.name for c in self.client.get_collections().collections]
        if name not in collections:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def delete_collection(self, name: str):
        self.client.delete_collection(name)

    def upsert(self, collection_name: str, points: List[VectorPoint]):
        qdrant_points = [
            PointStruct(
                id=p.id,
                vector=p.vector,
                payload={**p.payload, "text": p.text}
            )
            for p in points
        ]
        self.client.upsert(collection_name=collection_name, points=qdrant_points)

    def delete_by_file_path(self, collection_name: str, file_path: str):
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
                )
            )
        except Exception:
            pass

    def delete_by_file_paths(self, collection_name: str, file_paths: set[str]) -> int:
        """Batch-удаление точек по списку file_path. Возвращает количество удалённых."""
        if not file_paths:
            return 0
        
        from ..logger import get_logger
        log = get_logger("idx.qdrant")
        
        # Собираем все ID точек для удаления через scroll
        point_ids = []
        try:
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["file_path"],
                    with_vectors=False
                )
                for point in results:
                    fp = point.payload.get("file_path")
                    if fp in file_paths:
                        point_ids.append(point.id)
                if offset is None:
                    break
        except Exception as e:
            log.warning(f"Ошибка scroll при сборе сирот: {e}")
            return 0
        
        if not point_ids:
            return 0
        
        # Удаляем батчами по 1000 ID
        deleted = 0
        batch_size = 1000
        for i in range(0, len(point_ids), batch_size):
            batch = point_ids[i:i + batch_size]
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=batch
                )
                deleted += len(batch)
            except Exception as e:
                log.warning(f"Ошибка batch delete: {e}")
        
        return deleted

    def search(self, collection_name: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
        try:
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit
            ).points
            
            results = []
            for hit in search_results:
                res = {
                    "payload": hit.payload,
                    "score": hit.score
                }
                results.append(res)
            return results
        except Exception:
            return []

    def count(self, collection_name: str) -> int:
        try:
            info = self.client.get_collection(collection_name)
            return info.points_count
        except:
            return 0

    def optimize(self, collection_name: str):
        """Оптимизация коллекции: сжатие сегментов и очистка WAL."""
        try:
            from ..logger import get_logger
            log = get_logger("idx.qdrant")
            
            # Запускаем оптимизацию
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config={
                    "indexing_threshold": 0  # Форсирует индексацию
                }
            )
            
            # Ждём завершения оптимизации
            import time
            max_wait = 60
            start = time.time()
            while time.time() - start < max_wait:
                info = self.client.get_collection(collection_name)
                if info.status.name == "GREEN":
                    break
                time.sleep(1)
            
            log.info(f"Оптимизация {collection_name} завершена")
        except Exception as e:
            from ..logger import get_logger
            get_logger("idx.qdrant").warning(f"Ошибка оптимизации: {e}")

    def get_all_file_paths(self, collection_name: str) -> set[str]:
        """Получить все уникальные file_path из коллекции через scroll."""
        file_paths = set()
        try:
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["file_path"],
                    with_vectors=False
                )
                for point in results:
                    fp = point.payload.get("file_path")
                    if fp:
                        file_paths.add(fp)
                if offset is None:
                    break
        except Exception:
            pass
        return file_paths

    def clear_and_compact(self, collection_name: str, vector_size: int):
        """Удаляет все данные и сжимает SQLite хранилище."""
        from ..logger import get_logger
        log = get_logger("idx.qdrant")
        
        # 1. Удаляем коллекцию
        try:
            self.client.delete_collection(collection_name)
            log.info(f"Коллекция {collection_name} удалена")
        except Exception as e:
            log.warning(f"Ошибка удаления коллекции: {e}")
        
        # 2. Закрываем клиент чтобы освободить файлы
        self.close()
        
        # 3. DROP всех таблиц + VACUUM на sqlite файлах
        import sqlite3
        import time
        time.sleep(0.5)  # Даём время освободить файлы
        
        sqlite_files = list(self.path.rglob("*.sqlite"))
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
                log.info(f"Очистка {sqlite_file.name}: {old_size/1024/1024:.1f}MB -> {new_size/1024/1024:.1f}MB")
            except Exception as e:
                log.warning(f"Ошибка очистки {sqlite_file}: {e}")
        
        # 4. Переоткрываем клиент
        try:
            self.client = QdrantClient(path=str(self.path), timeout=5.0, check_compatibility=False)
        except TypeError:
            self.client = QdrantClient(path=str(self.path))
        
        # 5. Создаём пустую коллекцию
        self.create_collection(collection_name, vector_size)
        log.info(f"Коллекция {collection_name} пересоздана")

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()
