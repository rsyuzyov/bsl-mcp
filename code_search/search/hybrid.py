"""Гибридный поиск."""
import re
import pymorphy3

from ..model_manager import ModelManager
from ..vector_db import VectorDB


class HybridSearch:
    """Гибридный поиск: семантический + текстовый."""

    def __init__(self, db: VectorDB, collection_name: str, model_name: str, embedding_device: str = "cpu"):
        self.db = db
        self.collection_name = collection_name
        self.model_name = model_name
        self.embedding_device = embedding_device
        self.morph = pymorphy3.MorphAnalyzer()
        self.model_manager = ModelManager()

    def normalize(self, text: str) -> str:
        """Нормализация текста (лемматизация)."""
        words = re.findall(r"[а-яёa-z0-9]+", text.lower())
        return " ".join(self.morph.parse(w)[0].normal_form for w in words)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Выполнить гибридный поиск."""
        model_info = self.model_manager.get_model(self.model_name, self.embedding_device)
        
        if model_info.loading:
            return [{"file": "", "text": "⏳ Модель загружается, подождите...", "score": 0, "match": "loading"}]
        if model_info.error:
            return [{"file": "", "text": f"❌ Ошибка: {model_info.error}", "score": 0, "match": "error"}]
        
        query_lower = query.lower()
        query_normalized = self.normalize(query)
        
        # We need model to be loaded
        query_embedding = model_info.model.encode(["query: " + query]).tolist()[0]
        
        try:
            search_results = self.db.search(
                collection_name=self.collection_name, 
                vector=query_embedding, 
                limit=top_k * 2
            )
        except Exception as e:
            return [{"file": "", "text": f"❌ Ошибка поиска: {e}", "score": 0, "match": "error"}]
        
        results = []
        seen_files = set()
        
        for hit in search_results:
            payload = hit["payload"]
            file_path = payload["file_path"]
            text = payload.get("text", "")
            score = hit["score"]
            
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
