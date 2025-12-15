"""Гибридный поиск."""
import re
import pymorphy3
from qdrant_client import QdrantClient

from ..config import COLLECTION_NAME, model_state


class HybridSearch:
    """Гибридный поиск: семантический + текстовый."""

    def __init__(self, client: QdrantClient):
        self.client = client
        self.morph = pymorphy3.MorphAnalyzer()

    def normalize(self, text: str) -> str:
        """Нормализация текста (лемматизация)."""
        words = re.findall(r"[а-яёa-z0-9]+", text.lower())
        return " ".join(self.morph.parse(w)[0].normal_form for w in words)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Выполнить гибридный поиск."""
        if model_state.loading:
            return [{"file": "", "text": "⏳ Модель загружается, подождите...", "score": 0, "match": "loading"}]
        if model_state.error:
            return [{"file": "", "text": f"❌ Ошибка: {model_state.error}", "score": 0, "match": "error"}]
        
        query_lower = query.lower()
        query_normalized = self.normalize(query)
        
        query_embedding = model_state.model.encode(["query: " + query]).tolist()[0]
        search_results = self.client.query_points(
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
            elif query_normalized in self.normalize(file_path):
                score += 0.8
                match_type = "filename_stem"
            elif query_normalized in self.normalize(text):
                score += 0.3
                match_type = "text_stem"
            else:
                match_type = "semantic"
            
            if file_path not in seen_files:
                seen_files.add(file_path)
                results.append({"file": file_path, "text": text[:500], "score": score, "match": match_type})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
