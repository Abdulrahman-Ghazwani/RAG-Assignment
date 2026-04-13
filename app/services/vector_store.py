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
        for rank, i in enumerate(indices[0]):
            if i < len(self.documents):
                distance = float(distances[0][rank])
                # With normalized embeddings and L2 distance:
                # cosine similarity ~= 1 - (squared_l2_distance / 2)
                score = 1.0 - (distance / 2.0)
                doc = dict(self.documents[i])
                doc["distance"] = distance
                doc["score"] = score
                results.append(doc)

        return results