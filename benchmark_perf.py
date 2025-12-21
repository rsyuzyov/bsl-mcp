
import time
import shutil
import uuid
from pathlib import Path
from code_search.model_manager import ModelManager
from code_search.vector_db.qdrant import QdrantAdapter, VectorPoint

def benchmark():
    print("Starting benchmark...")
    
    # 1. Benchmark Embedding
    print("Loading model...")
    mm = ModelManager()
    # Force load
    while True:
        info = mm.get_model("cointegrated/rubert-tiny2")
        if not info.loading:
            break
        time.sleep(1)
    
    if info.error:
        print(f"Model load error: {info.error}")
        return

    print("Model loaded. Generating 500 random texts...")
    texts = [f"Function definition number {i} with some code logic inside to make it look like a real chunk." for i in range(500)]
    
    print("Benchmarking Embedding (Batch 500)...")
    start = time.time()
    embeddings = info.model.encode(texts, batch_size=64)
    file_time = time.time() - start
    print(f"Embedding 500 items took {file_time:.4f}s")
    print(f"Speed: {500/file_time:.2f} items/sec")
    print(f"Per item: {file_time/500*1000:.2f} ms")

    # 2. Benchmark DB
    print("\nBenchmarking Qdrant (Local)...")
    db_path = Path("./tmp_bench_qdrant")
    if db_path.exists():
        shutil.rmtree(db_path)
    
    db = QdrantAdapter(str(db_path))
    collection = "bench_col"
    db.create_collection(collection, 312)
    
    points = [
        VectorPoint(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"path": f"file_{i}"},
            text=texts[i]
        )
        for i in range(500)
    ]
    
    start = time.time()
    db.upsert(collection, points)
    upsert_time = time.time() - start
    print(f"Upsert 500 items took {upsert_time:.4f}s")
    print(f"Speed: {500/upsert_time:.2f} items/sec")
    
    # Close DB to release file locks
    if hasattr(db, 'close'):
        db.close()
    
    # Cleanup with retry for Windows
    if db_path.exists():
        for i in range(10):
            try:
                shutil.rmtree(db_path)
                break
            except PermissionError:
                time.sleep(0.5)
        else:
            print(f"Warning: Could not remove {db_path} due to file locks")

if __name__ == "__main__":
    benchmark()
