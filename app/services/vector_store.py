import uuid

import chromadb


class ChromaVectorStore:
    def __init__(self, dim: int, host: str, port: int, collection_name: str):
        self.dim = dim
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection_name = collection_name
        self._collection = None

    def _reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "l2"},
        )

    def _ensure_collection_open(self) -> None:
        if self._collection is not None:
            return
        try:
            self._collection = self._client.get_collection(self._collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "l2"},
            )

    def replace_all(self, embeddings: list[list[float]], documents: list[dict]) -> None:
        if not embeddings:
            raise ValueError("No embeddings to add.")
        if len(embeddings[0]) != self.dim:
            raise ValueError("Embedding dimension does not match store dimension.")
        self._reset_collection()
        assert self._collection is not None
        ids = [f"chunk_{i}" for i in range(len(embeddings))]
        metadatas = [{"page": d["page"], "source": d["source"]} for d in documents]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[d["content"] for d in documents],
            metadatas=metadatas,
        )

    def append(self, embeddings: list[list[float]], documents: list[dict]) -> None:
        if not embeddings:
            raise ValueError("No embeddings to add.")
        if len(embeddings[0]) != self.dim:
            raise ValueError("Embedding dimension does not match store dimension.")
        self._ensure_collection_open()
        assert self._collection is not None
        ids = [f"ch_{uuid.uuid4().hex}" for _ in embeddings]
        metadatas = [{"page": d["page"], "source": d["source"]} for d in documents]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[d["content"] for d in documents],
            metadatas=metadatas,
        )

    def delete_by_source(self, source: str) -> int:
        if self._collection is None:
            self._ensure_collection_open()
        if self._collection is None:
            return 0
        try:
            res = self._collection.get(
                where={"source": {"$eq": source}},
                include=[],
            )
        except Exception:
            try:
                res = self._collection.get(where={"source": source}, include=[])
            except Exception:
                return 0
        ids = res.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def distinct_sources(self) -> list[str]:
        if self._collection is None:
            self._ensure_collection_open()
        if self._collection is None:
            return []
        data = self._collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        names: set[str] = set()
        for m in metas:
            if m and m.get("source"):
                names.add(str(m["source"]))
        return sorted(names, key=str.lower)

    def wipe_entire_collection(self) -> None:
        """Delete the whole Chroma collection for this session (all embeddings)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None

    def search(self, query_embedding: list[float], top_k: int = 4):
        if self._collection is None:
            self._ensure_collection_open()
        if self._collection is None:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )
        dist_batch = results.get("distances") or []
        meta_batch = results.get("metadatas") or []
        doc_batch = results.get("documents") or []
        if not dist_batch or not dist_batch[0]:
            return []
        out = []
        for rank in range(len(dist_batch[0])):
            distance = float(dist_batch[0][rank])
            meta = dict(meta_batch[0][rank] or {})
            content = doc_batch[0][rank] or ""
            out.append({
                "content": content,
                "page": meta.get("page"),
                "source": meta.get("source"),
                "distance": distance,
                "score": 1.0 - (distance / 2.0),
            })
        return out
