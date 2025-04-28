from typing import Any

import hydra
from omegaconf import DictConfig
import numpy as np
import faiss
import h5py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import time


def read_embeddings(embeddings_path: str, num_embeddings: int):
    h5f = h5py.File(embeddings_path, 'r')
    return h5f["items"][:num_embeddings]


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


def setup_qdrant(collection_name: str, port: int, emb_dim: int):
    client = QdrantClient("localhost", port=port, timeout=30.0)
    
    try:
        client.get_collection(collection_name)
        client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE, on_disk=True),
    )
    return client


def upload_to_qdrant(client: QdrantClient, collection_name: str, embeddings: np.ndarray, batch_size: int):
    for i in range(0, len(embeddings), batch_size):
        batch_vectors = embeddings[i:i + batch_size]
        
        points = [
            models.PointStruct(
                id=idx + i,
                vector=embedding.tolist()
            )
            for idx, embedding in enumerate(batch_vectors)
        ]
        
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
            )
            print(f"Uploaded batch {i//batch_size + 1}/{(len(embeddings)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Failed to upload batch starting at ID {i}: {str(e)}")


def search_faiss(index: Any, query_embedding: np.ndarray, k: int=5):
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]


def search_qdrant(client: Any, query_embedding: np.ndarray, collection_name: str, k: int=5):
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )
    return [(hit.score, hit.id) for hit in search_result]


@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(conf: DictConfig):
    print("Read sample data...")
    embeddings = read_embeddings(conf["embeddings_path"], conf["num_embeddings"])
    
    print("Building FAISS index...")
    faiss_index = build_faiss_index(embeddings, conf["faiss_index_path"])
    
    print("Setting up Qdrant...")
    qdrant_client = setup_qdrant(
        conf["collection_name"], 
        conf["port"],
        conf["emb_dim"]
    )
    
    print("Uploading to Qdrant...")
    upload_to_qdrant(qdrant_client, conf["collection_name"], embeddings, conf["upload_batch_size"])
    
    query_embedding = embeddings[0]
    
    # FAISS search
    start_time = time.time()
    faiss_distances, faiss_indices = search_faiss(faiss_index, query_embedding, conf["num_neighbours"])
    faiss_time = time.time() - start_time
    
    # Qdrant search
    start_time = time.time()
    qdrant_results = search_qdrant(
        qdrant_client, 
        query_embedding, 
        conf["collection_name"],
        conf["num_neighbours"])
    qdrant_time = time.time() - start_time
    
    print("\nFAISS Results:")
    for dist, idx in zip(faiss_distances, faiss_indices):
        print(f"ID: {idx}, Distance: {dist:.4f}")
    
    print("\nQdrant Results:")
    for score, idx in qdrant_results:
        print(f"ID: {idx}, Score: {score:.4f}")
    
    print(f"\nFAISS search time: {faiss_time:.4f}s")
    print(f"Qdrant search time: {qdrant_time:.4f}s")

if __name__ == "__main__":
    main()