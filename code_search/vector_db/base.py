from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass
class VectorPoint:
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    text: str

class VectorDB(ABC):
    """Абстрактный класс для векторной базы данных."""

    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def create_collection(self, name: str, vector_size: int):
        """Создать коллекцию."""
        pass

    @abstractmethod
    def delete_collection(self, name: str):
        """Удалить коллекцию."""
        pass

    @abstractmethod
    def upsert(self, collection_name: str, points: List[VectorPoint]):
        """Добавить или обновить точки."""
        pass

    @abstractmethod
    def delete_by_file_path(self, collection_name: str, file_path: str):
        """Удалить точки по пути файла."""
        pass

    @abstractmethod
    def search(self, collection_name: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Поиск по вектору. Возвращает список словарей с payload и score."""
        pass

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Количество точек в коллекции."""
        pass
