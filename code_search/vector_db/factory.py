from .base import VectorDB


def get_vector_db(name: str, path: str) -> VectorDB:
    name = name.lower().strip()
    if name == "lancedb":
        from .lancedb import LanceDBAdapter
        return LanceDBAdapter(path)
    else:
        # Default to qdrant
        from .qdrant import QdrantAdapter
        return QdrantAdapter(path)
