import os, getpass
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from hybrid import hybrid_search
from reranker import rerank


load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your GEMINI AI API key: ")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorestore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def answer_query(query: str, final_k:int = 5, candidate_k:int=20) -> dict:
    
    candidates = hybrid_search(query, top_k=candidate_k)
    top_hits = rerank(query, candidates, top_k=final_k)

    context = "\n\n".join([f"[{i+1}] {h["text"]}" for i, h in enumerate(top_hits)])
    sources = list({h['source'] for h in top_hits})

    messages = [
        ("system", f"Answer the question using only the context below. If the context doesn't contain the answer, say so. Context:{context}"),
        ("human", f"Question: {query}"),
    ]

    response = model.invoke(messages)

    return {"Answer": response.text, "source": sources}