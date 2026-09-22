import faiss
import numpy as np
from citeguard.embeddings import embed


class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, text, meta):
        vec = embed([text])
        self.index.add(np.array(vec, dtype="float32"))
        self.metadata.append({"text": text, **meta})

    def is_empty(self):
        return self.index.ntotal == 0

    def search(self, query, k=3):
        if self.is_empty():
            return []
        vec = embed([query])
        k = min(k, self.index.ntotal)
        scores, idx = self.index.search(np.array(vec, dtype="float32"), k)
        results = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            results.append({"score": float(score), **self.metadata[i]})
        return results
