"""Диагностика сирот — сравнение путей в Qdrant и file_hashes.json"""
import json
from pathlib import Path
from qdrant_client import QdrantClient

# Путь к индексу hrm31 (из логов)
index_dir = Path(r"C:\Users\rsyuzyov\repo\bsl-mcp\1c-index\zup3")
qdrant_path = index_dir / "qdrant"
hashes_path = index_dir / "file_hashes.json"

# Загружаем хэши
hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
known_files = set(hashes.keys())

print(f"Файлов в file_hashes.json: {len(known_files)}")
print(f"Первые 3 пути из hashes:\n")
for p in list(known_files)[:3]:
    print(f"  '{p}'")

# Получаем пути из Qdrant
client = QdrantClient(path=str(qdrant_path))
db_files = set()
offset = None
while True:
    results, offset = client.scroll(
        collection_name="code_hrm31",
        limit=1000,
        offset=offset,
        with_payload=["file_path"],
        with_vectors=False
    )
    for point in results:
        fp = point.payload.get("file_path")
        if fp:
            db_files.add(fp)
    if offset is None:
        break

print(f"\nФайлов (уникальных file_path) в Qdrant: {len(db_files)}")
print(f"Первые 3 пути из Qdrant:\n")
for p in list(db_files)[:3]:
    print(f"  '{p}'")

# Сравнение
orphans = db_files - known_files
print(f"\nСироты (в Qdrant, но нет в hashes): {len(orphans)}")
if orphans:
    print("Первые 5 сирот:")
    for p in list(orphans)[:5]:
        print(f"  '{p}'")

missing = known_files - db_files
print(f"\nОтсутствуют в Qdrant (есть в hashes): {len(missing)}")
if missing:
    print("Первые 5 отсутствующих:")
    for p in list(missing)[:5]:
        print(f"  '{p}'")
