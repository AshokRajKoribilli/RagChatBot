import hashlib
from pathlib import Path

import sqlite3
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from threading import Lock
from logger import logger

_jobs: dict[str, dict] = {}
_jobs_lock = Lock()


def set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {})
        _jobs[job_id].update(fields)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def run_ingest_job(job_id: str, saved_path: Path, original_name: str):
    """Wrapper around ingest_pdf that records status in the job tracker."""
    set_job(job_id, status="running", original_name=original_name)
    try:
        result = ingest_pdf(saved_path, original_name)
        set_job(job_id, status="done", **result)
    except Exception as e:
        set_job(job_id, status="error", message=str(e))

CHROMA_DIR = "./chrome_db"
SQLITE_PATH = "keyword.db"
UPLOAD_DIR = Path("./documents")
UPLOAD_DIR.mkdir(exist_ok=True)

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _init_sqlite():
    logger.info("Initializing SQLite")
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED, source UNINDEXED, content,
            tokenize='porter'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            file_path TEXT PRIMARY KEY,
            original_name TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _get_vectorstore():
    logger.info("Getting Vectorstore")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)


def _delete_existing_chunks(conn, vectorstore, file_path: str):
    logger.info("Deleting Existing Chunks")
    rows = conn.execute(
        "SELECT chunk_id FROM chunks_fts WHERE source = ?", (file_path,)
    ).fetchall()
    ids = [r[0] for r in rows]
    if ids:
        vectorstore.delete(ids=ids)
        conn.execute("DELETE FROM chunks_fts WHERE source = ?", (file_path,))
    conn.execute("DELETE FROM ingested_files WHERE file_path = ?", (file_path,))
    conn.commit()


def ingest_pdf(saved_path: Path, original_name: str) -> dict:
    """
    Ingest a single PDF that has already been saved to disk.
    Returns a result dict with status + chunk count.
    """
    logger.info("Ingesting PDF")
    conn = _init_sqlite()
    vectorstore = _get_vectorstore()

    file_path = str(saved_path)
    fhash = _file_hash(saved_path)

    # Has this exact file content been ingested before (by any name)?
    existing = conn.execute(
        "SELECT file_path FROM ingested_files WHERE file_hash = ?", (fhash,)
    ).fetchone()
    if existing:
        conn.close()
        return {
            "status": "duplicate",
            "message": f"This file is already ingested as '{existing[0]}'.",
            "chunks_added": 0,
        }

    # If a file at the same path exists with a different hash → it changed
    if conn.execute(
        "SELECT 1 FROM ingested_files WHERE file_path = ?", (file_path,)
    ).fetchone():
        _delete_existing_chunks(conn, vectorstore, file_path)

    # Load + split
    try:
        docs = PyPDFLoader(file_path).load()
    except Exception as e:
        conn.close()
        return {"status": "error", "message": f"Failed to parse PDF: {e}", "chunks_added": 0}

    chunks = _splitter.split_documents(docs)
    if not chunks:
        conn.close()
        return {"status": "error", "message": "PDF contained no extractable text.", "chunks_added": 0}

    chunk_ids = [f"{fhash[:12]}:{i}" for i in range(len(chunks))]
    for cid, c in zip(chunk_ids, chunks):
        c.metadata["chunk_id"] = cid
        c.metadata["source"] = file_path
        c.metadata["original_name"] = original_name

    vectorstore.add_documents(documents=chunks, ids=chunk_ids)
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, source, content) VALUES (?, ?, ?)",
        [(cid, file_path, c.page_content) for cid, c in zip(chunk_ids, chunks)],
    )
    conn.execute(
        "INSERT INTO ingested_files (file_path, original_name, file_hash, chunk_count) "
        "VALUES (?, ?, ?, ?)",
        (file_path, original_name, fhash, len(chunks)),
    )
    conn.commit()
    vectorstore.persist()
    conn.close()

    return {
        "status": "ok",
        "message": f"Ingested '{original_name}'.",
        "chunks_added": len(chunks),
    }


def list_documents() -> list[dict]:
    logger.info("Listing Documents")
    conn = _init_sqlite()
    rows = conn.execute(
        "SELECT file_path, original_name, chunk_count, ingested_at "
        "FROM ingested_files ORDER BY ingested_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"file_path": r[0], "original_name": r[1], "chunk_count": r[2], "ingested_at": r[3]}
        for r in rows
    ]


def delete_document(file_path: str) -> dict:
    logger.info(f"Deleting Document - File Path: {file_path}")
    conn = _init_sqlite()
    vectorstore = _get_vectorstore()
    exists = conn.execute(
        "SELECT 1 FROM ingested_files WHERE file_path = ?", (file_path,)
    ).fetchone()
    if not exists:
        conn.close()
        return {"status": "not_found", "message": "No such document."}
    _delete_existing_chunks(conn, vectorstore, file_path)
    vectorstore.persist()
    conn.close()
    # Optionally remove the file from disk
    Path(file_path).unlink(missing_ok=True)
    return {"status": "ok", "message": "Deleted."}