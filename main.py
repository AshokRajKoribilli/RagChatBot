import hashlib
from ingest import ingest_pdf
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
# from rag import answer_query
from graph import answer_question
from ingest import (
    ingest_pdf, list_documents, delete_document, run_ingest_job, get_job, UPLOAD_DIR
)
from logger import logger


app = FastAPI(title="Rag ChatBot")

MAX_FILE_SIZE_MB = 25
BACKGROUND_THRESHOLD_MB = 5

class Query(BaseModel):
    query: str
    k: int = 5
    thread_id: str | None = None

class Answer(BaseModel):
    answer: str
    sources: list[str]
    thread_id: str
    history_length: int

@app.post("/chat", response_model=Answer)
def chat(q: Query):
    try:
        logger.info(f"Chat API - Query: {q.query}\tThread ID: {q.thread_id}")
        thread_id = q.thread_id or str(uuid4())
        # return answer_query(q.query, final_k=q.k)
        return answer_question(q.query, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_endpoint(background: BackgroundTasks, file: UploadFile = File(...)):
    logger.info(f"Ingest API - File: {file.filename}")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(400, f"Unexpected content type: {file.content_type}")

    content = await file.read()
    size_bytes = len(content)
    size_mb = size_bytes / (1024 * 1024)

    if not content:
        raise HTTPException(400, "Empty file")
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File is too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB).")
    
    fhash = hashlib.sha256(content).hexdigest()
    saved_path = UPLOAD_DIR / f"{fhash[:16]}.pdf"
    if not saved_path.exists():
        saved_path.write_bytes(content)

    if size_mb <= BACKGROUND_THRESHOLD_MB:
        result = ingest_pdf(saved_path, original_name=file.filename)
        if result["status"] == "error":
            raise HTTPException(422, result["message"])
        return {"mode": "sync", **result}

    job_id = str(uuid4())
    background.add_task(
        run_ingest_job, job_id, saved_path, file.filename
    )

    return{
        "mode": "async",
        "status": "queued",
        "job_id": job_id,
        "size_mb": round(size_mb, 2),
        "poll_url": f"/ingest/status/{job_id}",
    }

@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    logger.info(f"Ingest Status API - Job ID: {job_id}")
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown Job ID")
    return {"job_id": job_id, **job}

@app.get("/documents")
def documents():
    logger.info("Documents API - Listing documents")
    return {"Documents": list_documents()}

@app.delete("/documents")
def remove_document(file_path: str):
    logger.info(f"Delete Document API - File Path: {file_path}")
    result = delete_document(file_path)
    if result["status"] == "not_found":
        raise HTTPException(404, result["message"])
    return result

@app.get("/health")
def health():
    logger.info("Health API")
    return {"status": "OK"}
