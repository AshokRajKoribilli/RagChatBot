# retrievers.py
import re
import sqlite3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def sanitize_fts5_query(query: str) -> str:
    """
    Convert a raw user query into a safe FTS5 MATCH expression.
    Strips punctuation (including ?, !, ., etc.) and wraps each
    remaining token as a literal phrase so FTS5 operators are inert.
    """
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return ""
    # Implicit AND between quoted tokens. Swap to " OR ".join(...) for broader recall.
    return "OR".join(f'"{t}"' for t in tokens)

def bm25_search(query: str, k: int = 10):
    fts_query = sanitize_fts5_query(query)
    if not fts_query:
        return []
    conn = sqlite3.connect("keyword.db")
    conn.row_factory = sqlite3.Row
    # FTS5's rank column is the BM25 score (lower = more relevant)
    rows = conn.execute("""
        SELECT chunk_id, source, content, rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (fts_query, k)).fetchall()
    conn.close()
    return [{"id": r["chunk_id"], "text": r["content"], "source": r["source"]}
            for r in rows]

def vector_search(query: str, k: int = 10):
    results = vectorstore.similarity_search(query, k=k)
    return [{"id": d.metadata.get("chunk_id"),
             "text": d.page_content,
             "source": d.metadata.get("source", "")}
            for d in results]