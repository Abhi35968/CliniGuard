import json
import os
from typing import List, Dict, Any, Optional
from src.config import SOPS_FILE_PATH, RETRIEVAL_TOP_K
from src.schemas.context import UserContext
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


class SOPRetriever:
    """Semantic SOP Retriever combining Vector Embeddings, Similarity Search, and Metadata Filtering."""

    def __init__(self, sops_path: str = SOPS_FILE_PATH, embedding_service: Optional[EmbeddingService] = None):
        self.sops_path = sops_path
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = VectorStore()
        self._sops_cache: List[Dict[str, Any]] = []
        self._last_mtime: float = 0.0
        self._index_sops()

    def _index_sops(self, force_reload: bool = False):
        """Indexes SOP definitions into vector store."""
        if not os.path.exists(self.sops_path):
            return

        try:
            mtime = os.path.getmtime(self.sops_path)
            if force_reload or not self._sops_cache or mtime > self._last_mtime:
                with open(self.sops_path, "r", encoding="utf-8") as f:
                    self._sops_cache = json.load(f)
                self._last_mtime = mtime

                # Index SOPs into vector store
                self.vector_store.clear()
                doc_ids = []
                contents = []
                metadatas = []

                for sop in self._sops_cache:
                    sop_id = sop.get("id", "UNKNOWN")
                    title = sop.get("title", "")
                    guidance = sop.get("guidance", "")
                    acts = ", ".join(sop.get("target_activities", []))
                    demos = ", ".join(sop.get("target_demographics", []))
                    
                    # Create rich text representation for semantic embedding
                    text_for_embedding = f"SOP {sop_id}: {title}. Category: {sop.get('category')}. Activities: {acts}. Demographics: {demos}. Guidance: {guidance}"
                    
                    doc_ids.append(sop_id)
                    contents.append(text_for_embedding)
                    metadatas.append(sop)

                if contents:
                    vectors = self.embedding_service.embed_documents(contents)
                    self.vector_store.add_documents(doc_ids, contents, vectors, metadatas)
        except Exception as e:
            print(f"[SOPRetriever] Error indexing SOPs: {e}")

    def retrieve(
        self,
        query: str,
        context: UserContext,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant SOPs matching query and context via semantic vector search & metadata evaluation.
        """
        self._index_sops()  # Ensure cache is fresh

        if not self._sops_cache:
            return []

        query_str = query.strip()
        activity = (context.activity or "").lower()
        demographics = [d.lower() for d in context.demographics]

        # Embed query for semantic search
        query_vector = self.embedding_service.embed_query(query_str or activity or "outdoor safety")

        # Perform semantic similarity search
        vector_results = self.vector_store.similarity_search(query_vector=query_vector, top_k=top_k * 2)

        matched_candidates: List[Dict[str, Any]] = []

        for sop_meta, score in vector_results:
            sop_copy = dict(sop_meta)
            sop_copy["similarity_score"] = round(float(score), 4)
            matched_candidates.append(sop_copy)

        # Fallback: if vector store produces empty, return raw SOP cache
        if not matched_candidates:
            matched_candidates = [dict(s) for s in self._sops_cache]

        return matched_candidates[:top_k]
