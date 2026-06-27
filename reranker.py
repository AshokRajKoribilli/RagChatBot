from sentence_transformers import CrossEncoder

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

def rerank(query: str, candidates: list[dict], top_k: int=5) -> list[dict]:
    """
    candidates: list of {"id", "text", "source"} from hybrid_search
    Returns the top_k highest-scoring candidates, ordered by relevance.
    """

    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = _model.predict(pairs)

    scored = sorted(
        zip(candidates, scores),
        key =lambda x: x[1],
        reverse=True
    )

    return [
        {**c, "rerank_score": float(s)}
        for c, s in scored[:top_k]
    ]