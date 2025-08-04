from fastapi import FastAPI, UploadFile, File, Form, Request
import json
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from io import BytesIO
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
)
from model import stream_llm
from qdrant_client.http.exceptions import UnexpectedResponse


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
    stream_mode = form.get("stream", "true").lower() == "true"  # mặc định là true

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
                for chunk in stream_llm(question, context=context):
                    try:
                        yield chunk  # không thêm \n hay encode
                    except Exception as e:
                        print("❌ Lỗi stream chunk:", e)

            return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

        else:
            # gom toàn bộ chunk vào 1 string
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
    from datetime import datetime, timezone
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
    from datetime import datetime, timezone
    from uuid import uuid4

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
    return export_collection()

@app.post("/import_db")
async def import_db(file: UploadFile):
    content = await file.read()
    raw = json.loads(content)
    import_collection_from_raw(raw)
    return {"message": "Import thành công"}

