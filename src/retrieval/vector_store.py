import math
from typing import Dict, List, Any, Optional, Tuple


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class VectorStoreRecord:
    def __init__(self, doc_id: str, content: str, vector: List[float], metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        self.vector = vector
        self.metadata = metadata


class VectorStore:
    """Lightweight in-memory vector store supporting metadata filtering and cosine similarity."""

    def __init__(self):
        self.records: List[VectorStoreRecord] = []

    def add_documents(
        self,
        doc_ids: List[str],
        contents: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """Adds vectorized document records to the store."""
        for d_id, text, vec, meta in zip(doc_ids, contents, vectors, metadatas):
            self.records.append(VectorStoreRecord(doc_id=d_id, content=text, vector=vec, metadata=meta))

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs vector similarity search filtered by metadata.
        Returns list of (metadata_dict, similarity_score).
        """
        results: List[Tuple[Dict[str, Any], float]] = []

        for record in self.records:
            # Metadata filter check
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    rec_val = record.metadata.get(k)
                    if isinstance(rec_val, list):
                        if isinstance(v, list):
                            if not any(item in rec_val for item in v) and "*" not in rec_val:
                                match = False
                                break
                        elif v not in rec_val and "*" not in rec_val:
                            match = False
                            break
                    elif rec_val != v and rec_val != "*":
                        match = False
                        break
                if not match:
                    continue

            score = cosine_similarity(query_vector, record.vector)
            results.append((record.metadata, score))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self):
        """Clears all indexed records."""
        self.records.clear()
