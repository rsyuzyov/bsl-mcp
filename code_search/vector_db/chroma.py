import chromadb
from pathlib import Path
from typing import List, Dict, Any

from .base import VectorDB, VectorPoint

class ChromaDBAdapter(VectorDB):
    def __init__(self, path: str):
        self.path = Path(path) / "chroma"
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))

    def create_collection(self, name: str, vector_size: int):
        # vector_size не используется явно, но задаем cosine distance
        self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def delete_collection(self, name: str):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    def upsert(self, collection_name: str, points: List[VectorPoint]):
        if not points:
            return
        
        collection = self.client.get_collection(collection_name)
        
        ids = [str(p.id) for p in points]
        embeddings = [p.vector for p in points]
        documents = [p.text for p in points]
        metadatas = []
        for p in points:
            meta = p.payload.copy()
            metadatas.append(meta)
            
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def delete_by_file_path(self, collection_name: str, file_path: str):
        try:
            collection = self.client.get_collection(collection_name)
            collection.delete(where={"file_path": file_path})
        except Exception:
            pass

    def search(self, collection_name: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[vector],
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['ids']:
                return []

            ids = results['ids'][0]
            dists = results['distances'][0]
            metadatas = results['metadatas'][0]
            docs = results['documents'][0]
            
            out = []
            for i in range(len(ids)):
                dist = dists[i] if dists else 1.0
                score = 1.0 - dist
                
                payload = metadatas[i]
                # Восстанавливаем text в payload, так как код поиска ожидает это
                payload["text"] = docs[i]
                
                out.append({
                    "payload": payload,
                    "score": score
                })
            return out
            
        except Exception:
            return []

    def count(self, collection_name: str) -> int:
        try:
            collection = self.client.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0
