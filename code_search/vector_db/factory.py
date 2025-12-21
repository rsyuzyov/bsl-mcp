from .base import VectorDB
from .qdrant import QdrantAdapter
from .lancedb import LanceDBAdapter
from .chroma import ChromaDBAdapter

def get_vector_db(name: str, path: str) -> VectorDB:
    name = name.lower().strip()
    if name == "lancedb":
        return LanceDBAdapter(path)
    elif name == "chromadb":
        return ChromaDBAdapter(path)
    else:
        # Default to qdrant
        return QdrantAdapter(path)
