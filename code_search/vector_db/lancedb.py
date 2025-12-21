import lancedb
import pyarrow as pa
from pathlib import Path
from typing import List, Dict, Any

from .base import VectorDB, VectorPoint

class LanceDBAdapter(VectorDB):
    def __init__(self, path: str):
        self.path = Path(path) / "lancedb"
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.path)

    def create_collection(self, name: str, vector_size: int):
        # Используем фиксированную схему
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
            pa.field("text", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("chunk", pa.int64())
        ])
        
        if name in self.db.table_names():
            # Если таблица существует, проверяем, не надо ли пересоздать?
            # В рамках данного интерфейса считаем, что если есть - ок.
            pass
        else:
            self.db.create_table(name, schema=schema)

    def delete_collection(self, name: str):
        if name in self.db.table_names():
            self.db.drop_table(name)

    def upsert(self, collection_name: str, points: List[VectorPoint]):
        if not points:
            return
        
        table = self.db.open_table(collection_name)
        data = []
        for p in points:
            row = {
                "id": str(p.id),
                "vector": p.vector,
                "text": p.text,
                "file_path": p.payload.get("file_path", ""),
                "chunk": p.payload.get("chunk", 0)
            }
            data.append(row)
        
        table.add(data)

    def delete_by_file_path(self, collection_name: str, file_path: str):
        if collection_name not in self.db.table_names():
            return
        table = self.db.open_table(collection_name)
        # Экранирование кавычек
        safe_path = file_path.replace("'", "''")
        table.delete(f"file_path = '{safe_path}'")

    def search(self, collection_name: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
        if collection_name not in self.db.table_names():
            return []
        
        table = self.db.open_table(collection_name)
        
        # Используем metric="cosine"
        results = table.search(vector).metric("cosine").limit(limit).to_list()
        
        out = []
        for r in results:
            # Для cosine в lancedb distance = 1 - cosine_similarity (обычно)
            # Но надо проверять. LanceDB возвращает distance.
            dist = r.get("_distance", 1.0)
            score = 1.0 - dist
            
            out.append({
                "payload": {
                    "text": r["text"],
                    "file_path": r["file_path"],
                    "chunk": r["chunk"]
                },
                "score": score
            })
        return out

    def count(self, collection_name: str) -> int:
        if collection_name not in self.db.table_names():
            return 0
        return self.db.open_table(collection_name).count_rows()
