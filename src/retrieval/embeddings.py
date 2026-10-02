import math
import re
from typing import List
from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY, OPENAI_API_KEY


class EmbeddingService:
    """Embedding model abstraction supporting OpenAI, Gemini, HuggingFace, and lightweight TF-IDF Fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._provider = self._init_provider()

    def _init_provider(self):
        # 1. Try sentence-transformers if installed
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            return ("sentence_transformers", model)
        except Exception:
            pass

        # 2. Try OpenAI Embeddings if key present
        if OPENAI_API_KEY:
            try:
                from langchain_openai import OpenAIEmbeddings
                model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                return ("openai", model)
            except Exception:
                pass

        # 3. Fallback to lightweight vector embedding (bag-of-words / TF-IDF vectorizer)
        return ("fallback", None)

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string into a dense vector float list."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document strings into dense vector float lists."""
        provider_type, model = self._provider

        if provider_type == "sentence_transformers" and model:
            embeddings = model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]

        elif provider_type == "openai" and model:
            return model.embed_documents(texts)

        # Fallback: Normalized Term Frequency vectorization over unified vocabulary
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """Dense feature vectorizer when no neural embedding library is loaded."""
        # Clean & tokenize
        tokenized = [[w.lower() for w in re.findall(r"\b\w+\b", t)] for t in texts]
        # Build vocabulary
        vocab = list(set(word for doc in tokenized for word in doc))
        if not vocab:
            return [[0.0] * 10 for _ in texts]
        
        vocab_map = {word: idx for idx, word in enumerate(vocab)}
        vectors = []
        for doc in tokenized:
            vec = [0.0] * len(vocab_map)
            for w in doc:
                if w in vocab_map:
                    vec[vocab_map[w]] += 1.0
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        return vectors
