import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add(self, embeddings: list[list[float]], documents: list[dict]):
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding: list[float], top_k: int = 4):
        query = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for i in indices[0]:
            if i < len(self.documents):
                results.append(self.documents[i])

        return results