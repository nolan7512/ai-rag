from fastapi import FastAPI, UploadFile, File, Form, Request
import json
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from io import BytesIO
import io
import contextlib
from docx import Document
import time
from rag import (
    chunk_by_sentences,
    store_chunks,
    search_context,
    should_use_rag,
    clear_collection,
    export_collection,
    import_collection_from_raw,
    update_file,
    delete_file_by_id,
    list_uploaded_files,
    debug_multilingual_search, get_current_config
)
from model import stream_llm
from qdrant_client.http.exceptions import UnexpectedResponse
import json
import rag
from datetime import datetime, timezone
from uuid import uuid4
import time

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ cho phép "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(request: Request):
    form = await request.form()
    question = form.get("question", "")
    history = form.get("history", "")
    stream_mode = form.get("stream", "true").lower() == "true"

    if not question:
        return {"error": "Missing question."}

    try:
        if should_use_rag(question):
            print("✅ Dùng RAG (có context)")
            context = search_context(question)
        else:
            print("❌ Không dùng RAG (fallback knowledge model)")
            context = ""

        if stream_mode:
            def stream():
                last_sent = time.time()
                try:
                    for chunk in stream_llm(question, context=context):
                        if chunk:
                            yield chunk
                            last_sent = time.time()
                        # Ping keepalive mỗi 5 giây nếu không có chunk mới
                        if time.time() - last_sent > 2:
                            yield " "
                            last_sent = time.time()
                    yield ""  # flush cuối
                except Exception as e:
                    print("❌ Lỗi stream:", e)
                    yield f"\n[Stream error: {str(e)}]"

            return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
        else:
            response_text = "".join(stream_llm(question, context=context))
            return JSONResponse({"answer": response_text})

    except Exception as e:
        print("❌ Lỗi xử lý:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/upload")
async def upload(
    tags: List[str] = Form(...),
    files: List[UploadFile] = File(...)
):
    for file in files:
        contents = await file.read()
        doc = Document(BytesIO(contents))
        text = "\n".join(para.text for para in doc.paragraphs)
        chunks = chunk_by_sentences(text)
        store_chunks(
            chunks,
            metadata={
                "tags": tags,
                "filename": file.filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "source": "file"
            }
        )
    return {"status": "ok"}

@app.post("/upload_text")
async def upload_text(
    text: str = Form(...),
    tags: List[str] = Form(...)
):
    chunks = chunk_by_sentences(text)
    metadata = {
        "file_id": uuid4().hex,
        "filename": "text_input",
        "tags": tags,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "text"
    }
    store_chunks(chunks, metadata)
    return {"message": "Text uploaded successfully"}


@app.get("/list_files")
def list_files():
    return JSONResponse(content=list_uploaded_files())

@app.delete("/delete_file/{file_id}")
def delete_file(file_id: str):
    try:
        delete_file_by_id(file_id)
        return {"message": f"Đã xoá dữ liệu liên quan đến file_id: {file_id}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/update_file")
async def update_file_endpoint(
    file: UploadFile = File(...),
    file_id: str = Form(...),
    tags: List[str] = Form(None)
):
    try:
        result = update_file(file, file_id, tags or [])
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/update_text")
async def update_text(
    file_id: str = Form(...),
    tags: List[str] = Form(...),
    content: str = Form(...)
):
    try:
        delete_file_by_id(file_id)
        chunks = chunk_by_sentences(content)
        store_chunks(chunks, metadata={
            "file_id": file_id,
            "tags": tags,
            "filename": "updated_text",
            "source": "text"
        })
        return {"message": "Text updated successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/get_text_by_file_id")
async def get_text_by_file_id(file_id: str = Form(...)):
    from rag import qdrant, COLLECTION_NAME, Filter, FieldCondition, MatchValue

    filter = Filter(
        must=[
            FieldCondition(
                key="file_id",
                match=MatchValue(value=file_id)
            )
        ]
    )
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter,
        with_payload=True,
        with_vectors=False,
        limit=1000
    )

    texts = [p.payload["text"] for p in points if "text" in p.payload]
    return {"text": "\n\n".join(texts)}

@app.post("/delete_by_id")
async def delete_by_id(file_id: str = Form(...)):
    try:
        delete_file_by_id(file_id)
        return {"message": f"File with ID {file_id} deleted successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/clear_db")
def clear_db():
    try:
        clear_collection()
        return {"message": "DB cleared and collection recreated ✅"}
    except UnexpectedResponse as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/export_db")
def export_db():
    data = export_collection()
    path = "qdrant_export.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return FileResponse(path, media_type="application/json", filename="qdrant_export.json")

@app.post("/import_db")
async def import_db(file: UploadFile):
    content = await file.read()
    raw = json.loads(content)
    import_collection_from_raw(raw)
    return {"message": "Import thành công"}

@app.post("/debug_rag")
async def debug_rag(question: str = Form(...)):
    """
    Debug endpoint to test RAG decision and configuration
    """
    try: 
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            should_use_rag = debug_multilingual_search(question)
        
        debug_output = f.getvalue()
        config = get_current_config()
        
        return JSONResponse({
            "question": question,
            "should_use_rag": should_use_rag,
            "debug_output": debug_output,
            "current_config": config
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/test_business_scenarios")
async def test_business_scenarios():
    """
    Test endpoint for business scenario validation
    """
    try:      
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            test_business_scenarios()
        
        test_output = f.getvalue()
        
        return JSONResponse({
            "test_results": test_output,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/rag_config")
def get_rag_config():
    """
    Get current RAG configuration
    """
    try:
        from rag import get_current_config
        return JSONResponse(get_current_config())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/update_rag_config")
async def update_rag_config(
    enable_multilingual: bool = Form(None),
    enable_strict_semantic: bool = Form(None),
    enable_business_mode: bool = Form(None),
    semantic_threshold: float = Form(None),
    min_embedding_score: float = Form(None),
    fallback_threshold: float = Form(None)
):
    """
    Update RAG configuration at runtime (temporary, not persisted to .env)
    """
    try:
        if enable_multilingual is not None:
            rag.ENABLE_MULTILINGUAL_FEATURES = enable_multilingual
        if enable_strict_semantic is not None:
            rag.ENABLE_STRICT_SEMANTIC_CHECK = enable_strict_semantic
        if enable_business_mode is not None:
            rag.BUSINESS_DOMAIN_MODE = enable_business_mode
        
        # Update environment variables temporarily
        if semantic_threshold is not None:
            os.environ["SEMANTIC_THRESHOLD"] = str(semantic_threshold)
        if min_embedding_score is not None:
            os.environ["MIN_EMBEDDING_SCORE"] = str(min_embedding_score)
        if fallback_threshold is not None:
            os.environ["FALLBACK_THRESHOLD"] = str(fallback_threshold)
        
        return JSONResponse({
            "message": "Configuration updated successfully",
            "new_config": rag.get_current_config()
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)