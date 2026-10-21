"""Embedding model for hybrid retrieval. Disabled when RAG_EMBEDDING_MODEL is empty or "off".

Two kinds of model:
- OpenAI ("text-embedding-..."): called with the service's OPENAI_API_KEY and asked for 1024 dimensions,
  so the knowledge.chunks vector(1024) column stays as it is. No model runs on our machines.
- sentence-transformers (for example "BAAI/bge-m3"): runs locally; needs the ingest extra.

Documents and queries must use the same model, so the model name and revision are stored on every chunk
and search only compares chunks made by the loaded model.
"""
from dataclasses import dataclass
import os
from typing import Any

DIMENSION = 1024
OPENAI_PREFIX = "text-embedding-"
OPENAI_BATCH = 96
# OpenAI rejects inputs over 8,192 tokens. Korean runs about 1 to 1.5 tokens per character, so 6,000 characters
# is safe. Only the vector uses the cut; keyword search and the excerpt keep the whole chunk.
OPENAI_MAX_CHARS = 6000


@dataclass
class Embedder:
    model_name: str
    revision: str
    model: Any

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.model_name.startswith(OPENAI_PREFIX):
            result = []
            for start in range(0, len(texts), OPENAI_BATCH):
                part = [text[:OPENAI_MAX_CHARS] for text in texts[start:start + OPENAI_BATCH]]
                response = self.model.embeddings.create(model=self.model_name, input=part,
                                                        dimensions=DIMENSION)
                result += [list(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]
        else:
            result = [list(map(float, vector)) for vector in self.model.encode(texts, normalize_embeddings=True)]
        if any(len(vector) != DIMENSION for vector in result):
            raise ValueError("EMBEDDING_DIMENSION_MISMATCH")
        return result


def load_embedder() -> Embedder | None:
    name = os.getenv("RAG_EMBEDDING_MODEL", "").strip()
    if not name or name.lower() in ("off", "none"):
        return None
    if name.startswith(OPENAI_PREFIX):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        from openai import OpenAI
        return Embedder(model_name=name, revision=f"dim{DIMENSION}", model=OpenAI(api_key=key, timeout=30, max_retries=2))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    revision = os.getenv("RAG_EMBEDDING_REVISION", "").strip() or None
    model = SentenceTransformer(name, revision=revision)
    return Embedder(model_name=name, revision=revision or "unpinned", model=model)
