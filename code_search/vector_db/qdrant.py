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

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()
