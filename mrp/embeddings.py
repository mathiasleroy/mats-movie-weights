"""Sentence embeddings via fastembed (ONNX-based, low-memory all-MiniLM-L6-v2)."""
import numpy as np
from mrp.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        print(f"  Loading embedding model '{EMBEDDING_MODEL_NAME}' …")
        _model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        print("  ✓ model loaded")
    return _model


def embed(text):
    """Return a 384-dim float32 vector.  Zeros if text is empty or model unavailable."""
    if not text or len(text.strip()) < 10:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        model = _get_model()
        # fastembed returns a generator; next() fetches the first embedding
        vec = next(model.embed([text]))
        return vec.astype(np.float32)
    except Exception as exc:
        print(f"  ⚠ embedding failed ({exc}), using zero vector")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)