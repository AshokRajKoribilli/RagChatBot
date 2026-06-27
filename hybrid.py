from retrievers import bm25_search, vector_search

def reciprocal_rank_fusion(result_lists, k_rrf:int = 60, top_k:int=5 ):
    scores={}
    texts={}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid=item["id"]
            scores[cid] = scores.get(cid, 0)+1/(k_rrf+rank+1)
            texts[cid] = item
    fused = sorted(scores.items(), key= lambda x: x[1], reverse=True)[:top_k]
    return [texts[cid] for cid, _ in fused]

def hybrid_search(query: str, top_k:int =20):
    bm25_hits = bm25_search(query, k=20)
    vector_hits = vector_search(query, k=20)
    return reciprocal_rank_fusion([bm25_hits, vector_hits], top_k=top_k)