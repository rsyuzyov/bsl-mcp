"""Гибридный поиск."""
import re
import threading
# Синглтон для MorphAnalyzer (инициализация ~20 сек, делаем один раз)
_morph_instance = None
_morph_lock = threading.Lock()

from ..model_manager import ModelManager
from ..vector_db import VectorDB
from ..metadata_utils import is_compilation_directive_match

def get_morph_analyzer():
    """Получить единственный экземпляр MorphAnalyzer."""
    global _morph_instance
    if _morph_instance is None:
        with _morph_lock:
            if _morph_instance is None:
                import time
                import pymorphy3
                print("Инициализация pymorphy3.MorphAnalyzer()...")
                start = time.time()
                _morph_instance = pymorphy3.MorphAnalyzer()
                print(f"pymorphy3 загружен за {time.time() - start:.2f} сек")
    return _morph_instance


class HybridSearch:
    """Гибридный поиск: семантический + текстовый."""
    
    # Минимальный score для чисто семантических результатов (без текстового совпадения)
    SEMANTIC_MIN_SCORE = 0.45

    def __init__(self, db: VectorDB, collection_name: str, model_name: str, embedding_device: str = "cpu"):
        self.db = db
        self.collection_name = collection_name
        self.model_name = model_name
        self.embedding_device = embedding_device
        self.model_manager = ModelManager()

    def normalize(self, text: str) -> str:
        """Нормализация текста (лемматизация)."""
        morph = get_morph_analyzer()
        words = re.findall(r"[а-яёa-z0-9]+", text.lower())
        return " ".join(morph.parse(w)[0].normal_form for w in words)
    
    def extract_snippet(self, text: str, query: str, context_chars: int = 100) -> str:
        """Извлечь сниппет с контекстом вокруг найденного query.
        
        Если query найден — возвращает контекст вокруг него.
        Иначе — первые 200 символов текста.
        """
        if not text:
            return ""
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Ищем прямое вхождение
        pos = text_lower.find(query_lower)
        if pos == -1:
            # Пробуем найти отдельные слова запроса
            query_words = query_lower.split()
            for word in query_words:
                if len(word) >= 3:
                    pos = text_lower.find(word)
                    if pos != -1:
                        break
        
        if pos != -1:
            # Найдено — берём контекст вокруг
            start = max(0, pos - context_chars)
            end = min(len(text), pos + len(query) + context_chars)
            
            # Выравниваем по границам слов
            if start > 0:
                space_pos = text.find(' ', start)
                if space_pos != -1 and space_pos < pos:
                    start = space_pos + 1
            if end < len(text):
                space_pos = text.rfind(' ', pos, end)
                if space_pos != -1:
                    end = space_pos
            
            snippet = text[start:end].strip()
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            return f"{prefix}{snippet}{suffix}"
        else:
            # Не найдено — первые 200 символов
            if len(text) <= 200:
                return text.strip()
            # Обрезаем по границе слова
            end = text.rfind(' ', 0, 200)
            if end == -1:
                end = 200
            return text[:end].strip() + "..."
    
    def highlight_match(self, text: str, query: str) -> str:
        """Выделить совпадение query в тексте через **маркеры**."""
        if not text or not query:
            return text
        
        # Экранируем спецсимволы regex
        pattern = re.escape(query)
        # Case-insensitive замена с сохранением оригинального регистра
        def replacer(match):
            return f"**{match.group(0)}**"
        
        result = re.sub(pattern, replacer, text, flags=re.IGNORECASE, count=1)
        return result

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Выполнить гибридный поиск."""
        model_info = self.model_manager.get_model(self.model_name, self.embedding_device)
        
        if model_info.loading:
            return [{"file": "", "text": "⏳ Модель загружается, подождите...", "score": 0, "match": "loading"}]
        if model_info.error:
            return [{"file": "", "text": f"❌ Ошибка: {model_info.error}", "score": 0, "match": "error"}]
        
        query_lower = query.lower()
        query_normalized = self.normalize(query)
        
        # Encode with DirectML fallback to CPU on crash
        try:
            query_embedding = model_info.model.encode(["query: " + query]).tolist()[0]
        except Exception as e:
            err_str = str(e)
            if "DirectML Native Crash" in err_str or "0xce" in err_str or "0xcf" in err_str or "utf-8" in err_str:
                # Fallback to CPU model
                from ..logger import get_logger
                logger = get_logger("app.search")
                logger.warning(f"DirectML crash during search, falling back to CPU model")
                cpu_model_info = self.model_manager.get_model(self.model_name, "cpu")
                # Wait for CPU model to load if needed
                import time
                for _ in range(60):  # max 30 sec wait
                    if not cpu_model_info.loading:
                        break
                    time.sleep(0.5)
                if cpu_model_info.error:
                    return [{"file": "", "text": f"❌ Ошибка: {cpu_model_info.error}", "score": 0, "match": "error"}]
                if cpu_model_info.loading:
                    return [{"file": "", "text": "⏳ CPU модель загружается, подождите...", "score": 0, "match": "loading"}]
                query_embedding = cpu_model_info.model.encode(["query: " + query]).tolist()[0]
            else:
                return [{"file": "", "text": f"❌ Ошибка эмбеддинга: {e}", "score": 0, "match": "error"}]
        
        try:
            search_results = self.db.search(
                collection_name=self.collection_name, 
                vector=query_embedding, 
                limit=top_k * 3  # Берём больше для фильтрации
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
            
            # Извлекаем метаданные объекта
            object_type = payload.get("object_type", "")
            object_name = payload.get("object_name", "")
            module_type = payload.get("module_type", "")
            
            # Фильтруем ложные совпадения из-за директив компиляции
            if is_compilation_directive_match(text, query):
                score -= 0.5  # Штраф за совпадение с директивой
            
            # Бонусы за совпадение с метаданными объекта
            if object_name and query_lower in object_name.lower():
                score += 1.5
                match_type = "object_name"
            elif object_type and query_lower in object_type.lower():
                score += 1.2
                match_type = "object_type"
            elif query_lower in file_path.lower():
                score += 1.0
                match_type = "filename"
            elif query_lower in text.lower():
                score += 0.5
                match_type = "text"
            elif query_normalized in self.normalize(object_name or ""):
                score += 1.0
                match_type = "object_name_stem"
            elif query_normalized in self.normalize(file_path):
                score += 0.8
                match_type = "filename_stem"
            elif query_normalized in self.normalize(text):
                score += 0.3
                match_type = "text_stem"
            else:
                match_type = "semantic"
            
            # Фильтруем чисто семантические результаты с низким score
            if match_type == "semantic" and hit["score"] < self.SEMANTIC_MIN_SCORE:
                continue
            
            if file_path not in seen_files:
                seen_files.add(file_path)
                
                # Формируем контекст объекта для отображения
                context = ""
                if object_type and object_name:
                    context = f"{object_type}: {object_name}"
                    if module_type:
                        context += f" ({module_type})"
                
                # Умный сниппет с контекстом вокруг query
                snippet = self.extract_snippet(text, query)
                # Выделяем совпадение
                snippet = self.highlight_match(snippet, query)
                
                results.append({
                    "file": file_path, 
                    "text": snippet, 
                    "score": score, 
                    "match": match_type,
                    "payload": payload,  # Возвращаем полный payload
                    "context": context,  # Контекст объекта
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

