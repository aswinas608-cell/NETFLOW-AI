import os
import io
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
from rag_engine import RAGEngine

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global RAG engine instance
rag = RAGEngine()

# ─── Request/Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    groq_api_key: str
    model: str = "llama-3.3-70b-versatile"
    retrieval_mode: str = "tfidf"
    embedding_api_key: Optional[str] = None
    top_k: int = 4
    temperature: float = 0.3

class DeleteResponse(BaseModel):
    success: bool
    message: str

class ValidateRequest(BaseModel):
    groq_api_key: str
    model: str = "llama-3.3-70b-versatile"

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/documents")
async def get_documents():
    docs = rag.get_documents()
    result = []
    for doc_id, meta in docs.items():
        result.append({
            "id": doc_id,
            "filename": meta["filename"],
            "chunk_count": meta["chunk_count"],
            "char_count": meta["char_count"],
        })
    return {"documents": result}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    retrieval_mode: str = Form("tfidf"),
    embedding_api_key: str = Form(""),
):
    # Validate file type
    allowed_extensions = {".txt", ".md", ".pdf"}
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: .txt, .md, .pdf"
        )

    content_bytes = await file.read()

    # Parse content
    text_content = ""
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content_bytes))
            pages_text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            text_content = "\n\n".join(pages_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    else:
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content_bytes.decode("latin-1")

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The uploaded file appears to be empty or unreadable.")

    doc_id = str(uuid.uuid4())
    api_key = embedding_api_key.strip() or None

    try:
        meta = rag.add_document(
            filename=filename,
            content=text_content,
            doc_id=doc_id,
            embedding_mode=retrieval_mode,
            api_key=api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": meta["chunk_count"],
        "char_count": meta["char_count"],
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    success = rag.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True, "message": "Document deleted successfully."}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.groq_api_key.strip():
        raise HTTPException(status_code=400, detail="Groq API key is required.")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Check if any documents are indexed
    docs = rag.get_documents()
    if not docs:
        return {
            "answer": "⚠️ No documents have been uploaded yet. Please upload a document first to enable RAG-based responses.",
            "sources": [],
            "model": req.model,
        }

    # Retrieve relevant chunks
    api_key_for_embed = req.embedding_api_key.strip() if req.embedding_api_key else None
    retrieved = rag.retrieve(
        query=req.query,
        top_k=req.top_k,
        embedding_mode=req.retrieval_mode,
        api_key=api_key_for_embed,
    )

    # Build context string
    context_parts = []
    for i, chunk in enumerate(retrieved):
        context_parts.append(
            f"[Source {i+1}: {chunk['filename']} — Chunk {chunk['chunk_idx']+1}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build the augmented prompt
    system_prompt = (
        "You are an expert AI assistant specialized in answering questions based on provided document context. "
        "Your answers should be accurate, well-structured, and grounded in the retrieved context. "
        "If the answer is not found in the context, say so clearly rather than hallucinating. "
        "Always cite which source(s) you used in your answer (e.g., 'According to [Source 1]...')."
    )

    user_message = f"""Here is the relevant context retrieved from the documents:

{context}

---

User Question: {req.query}

Please answer the question based on the above context. Be thorough and cite your sources."""

    # Call Groq API
    try:
        client = Groq(api_key=req.groq_api_key.strip())
        completion = client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=req.temperature,
            max_tokens=2048,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "invalid_api_key" in err_str.lower() or "authentication" in err_str.lower():
            raise HTTPException(status_code=401, detail="Invalid Groq API key. Please check your API key in settings.")
        elif "429" in err_str:
            raise HTTPException(status_code=429, detail="Groq API rate limit exceeded. Please wait a moment and try again.")
        elif "model_not_found" in err_str.lower() or "model" in err_str.lower():
            raise HTTPException(status_code=400, detail=f"Model not found: {req.model}. Please select a valid model.")
        else:
            raise HTTPException(status_code=500, detail=f"Groq API error: {err_str}")

    # Prepare clean source output (without embeddings)
    clean_sources = []
    for chunk in retrieved:
        clean_sources.append({
            "filename": chunk["filename"],
            "chunk_idx": chunk["chunk_idx"],
            "text": chunk["text"],
            "score": chunk.get("score", 0.0),
            "doc_id": chunk["doc_id"],
        })

    return {
        "answer": answer,
        "sources": clean_sources,
        "model": req.model,
    }

@app.post("/validate")
async def validate_api_key(req: ValidateRequest):
    """Validate the Groq API key by making a minimal test call."""
    try:
        client = Groq(api_key=req.groq_api_key.strip())
        response = client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        model_used = response.model
        return {"valid": True, "model": model_used}
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "invalid_api_key" in err_str.lower():
            return {"valid": False, "error": "Invalid API key."}
        elif "model_not_found" in err_str.lower():
            return {"valid": False, "error": f"Model '{req.model}' not found."}
        else:
            return {"valid": False, "error": str(e)}
