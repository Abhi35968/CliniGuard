import math
import re
from typing import List, Tuple, Optional
from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY, OPENAI_API_KEY, LLM_PROVIDER
import os
from src.observability import get_logger

logger = get_logger("embeddings")


# Explicit embedding provider override.
# Values: "sentence_transformers" | "openai" | "google" | "tfidf" | "auto" (default)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").lower()


class EmbeddingService:
    """
    Embedding model abstraction supporting OpenAI, Google Gemini, HuggingFace
    sentence-transformers, and a lightweight TF-IDF fallback.

    Provider resolution order when EMBEDDING_PROVIDER=auto:
        1. sentence-transformers (if installed)
        2. OpenAI  (if OPENAI_API_KEY present)
        3. Google  (if GOOGLE_API_KEY present)
        4. TF-IDF fallback (always available)

    Set EMBEDDING_PROVIDER in .env to force a specific backend, e.g.:
        EMBEDDING_PROVIDER=openai
        EMBEDDING_MODEL=text-embedding-3-small
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._provider_type, self._model = self._init_provider()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_provider(self) -> Tuple[str, Optional[object]]:
        """Resolves and initialises the embedding backend according to EMBEDDING_PROVIDER."""
        requested = EMBEDDING_PROVIDER

        if requested == "openai":
            result = self._try_openai()
            if result:
                return result
            logger.warning("EMBEDDING_PROVIDER=openai requested but init failed – falling back.")
            return self._fallback()

        if requested == "google":
            result = self._try_google()
            if result:
                return result
            logger.warning("EMBEDDING_PROVIDER=google requested but init failed – falling back.")
            return self._fallback()

        if requested == "sentence_transformers":
            result = self._try_sentence_transformers()
            if result:
                return result
            logger.warning("EMBEDDING_PROVIDER=sentence_transformers requested but init failed – falling back.")
            return self._fallback()

        if requested == "tfidf":
            return self._fallback()

        # ── auto: try each in order ────────────────────────────────────
        result = self._try_sentence_transformers()
        if result:
            return result

        result = self._try_openai()
        if result:
            return result

        result = self._try_google()
        if result:
            return result

        return self._fallback()

    def _try_sentence_transformers(self) -> Optional[Tuple[str, object]]:
        try:
            from sentence_transformers import SentenceTransformer
            model_id = self.model_name if self.model_name else "all-MiniLM-L6-v2"
            model = SentenceTransformer(model_id)
            logger.info("Using sentence-transformers: %s", model_id)
            return ("sentence_transformers", model)
        except Exception as e:
            logger.debug("sentence-transformers unavailable: %s", e)
            return None

    def _try_openai(self) -> Optional[Tuple[str, object]]:
        if not OPENAI_API_KEY:
            return None
        try:
            from langchain_openai import OpenAIEmbeddings
            model_id = self.model_name if self.model_name and "embedding" in self.model_name else "text-embedding-3-small"
            model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=model_id)
            logger.info("Using OpenAI embeddings: %s", model_id)
            return ("openai", model)
        except Exception as e:
            logger.debug("OpenAI embeddings unavailable: %s", e)
            return None

    def _try_google(self) -> Optional[Tuple[str, object]]:
        if not GOOGLE_API_KEY:
            return None
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model_id = self.model_name if "embedding" in self.model_name else "models/text-embedding-004"
            model = GoogleGenerativeAIEmbeddings(model=model_id, google_api_key=GOOGLE_API_KEY)
            logger.info("Using Google Gemini embeddings: %s", model_id)
            return ("google", model)
        except Exception as e:
            logger.debug("Google embeddings unavailable: %s", e)
            return None

    def _fallback(self) -> Tuple[str, None]:
        logger.info("Using TF-IDF fallback embeddings.")
        return ("tfidf", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string into a dense vector float list."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document strings into dense vector float lists."""
        if self._provider_type == "sentence_transformers" and self._model:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]

        elif self._provider_type == "openai" and self._model:
            return self._model.embed_documents(texts)

        elif self._provider_type == "google" and self._model:
            return self._model.embed_documents(texts)

        # TF-IDF fallback
        return self._tfidf_embed(texts)

    @property
    def provider(self) -> str:
        """Returns the active provider name for observability."""
        return self._provider_type

    # ------------------------------------------------------------------
    # TF-IDF fallback
    # ------------------------------------------------------------------

    def _tfidf_embed(self, texts: List[str], dim: int = 128) -> List[List[float]]:
        """Fixed-dimension feature hashing fallback when no neural model is available."""
        vectors = []
        for text in texts:
            tokens = [w.lower() for w in re.findall(r"\b\w+\b", text)]
            vec = [0.0] * dim
            for w in tokens:
                idx = abs(hash(w)) % dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors
