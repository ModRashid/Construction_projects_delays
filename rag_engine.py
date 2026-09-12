import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL = None

def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def build_index(records):
    """Embed all document chunks and build an in-memory FAISS index."""
    model = get_embedding_model()
    texts = [record["text"] for record in records]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, records

def retrieve(query, index, records, top_k=5):
    """Retrieve the most semantically relevant chunks."""
    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    k = min(top_k, len(records))
    scores, ids = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(records[idx])
            item["score"] = float(score)
            results.append(item)

    return results
