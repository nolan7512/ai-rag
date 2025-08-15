from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from io import BytesIO
from docx import Document
from datetime import datetime, timezone
from uuid import uuid4
import time
import json
import io
import contextlib
import os

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
    debug_multilingual_search, 
    get_current_config
)
from model import stream_llm, stream_translate, get_last_answer_from_history, detect_target_lang

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # chỉnh theo domain FE của bạn nếu muốn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Helpers ---------------------------

TRANSLATE_KEYWORDS = ["dịch", "translate", "翻译"]

def is_translate_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(k in t for k in TRANSLATE_KEYWORDS):
        return True
    # câu kiểu 'dịch sang tiếng Trung', 'dịch qua tiếng Anh'...
    return False

# --------------------------- Chat API ---------------------------

@app.post("/ask")
async def ask(request: Request):
    """
    Chat endpoint:
      - Tự động nhận diện yêu cầu dịch -> dịch câu trả lời gần nhất
      - Nếu không phải dịch: chạy RAG decision + stream trả lời
      - Hỗ trợ stream: true/false (mặc định true)
      - Nhận history: JSON messages hoặc text 'user:/bot:'
    Form fields:
      - question (str)      : câu người dùng hỏi
      - history (str)       : JSON messages hoặc text transcript
      - stream (bool|str)   : "true"/"false" (mặc định "true")
    """
    form = await request.form()
    question = form.get("question", "")
    history = form.get("history", "")
    stream_mode = str(form.get("stream", "true")).lower() == "true"

    if not question:
        return {"error": "Missing question."}

    # 1) Nếu là yêu cầu dịch: dịch câu trả lời gần nhất từ history
    if is_translate_request(question):
        last_answer = get_last_answer_from_history(history)
        if not last_answer:
            # Không có câu trước để dịch -> coi như yêu cầu dịch text người dùng gửi sau từ khóa 'dịch ...: <text>'
            # Thử tách phần sau dấu ":" nếu có
            candidate = question.split(":", 1)
            source_text = candidate[1].strip() if len(candidate) == 2 else question
        else:
            source_text = last_answer

        target_lang = detect_target_lang(question)  # Chinese / Vietnamese / English
        include_original = True  # giữ "(原文|Nguyên văn|Original ...)" như ví dụ

        def _stream_trans():
            last_sent = time.time()
            try:
                for chunk in stream_translate(source_text, target_lang, include_original):
                    if chunk:
                        yield chunk
                        last_sent = time.time()
                    if time.time() - last_sent > 2:
                        yield " "
                        last_sent = time.time()
                yield ""
            except Exception as e:
                yield f"\n[Stream error: {str(e)}]"

        if stream_mode:
            return StreamingResponse(_stream_trans(), media_type="text/plain; charset=utf-8")
        else:
            # gom lại thành 1 string
            text = "".join(list(stream_translate(source_text, target_lang, include_original)))
            return JSONResponse({"answer": text})

    # 2) Không phải yêu cầu dịch: chạy RAG (nếu cần) + gửi kèm history
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
                    for chunk in stream_llm(question, context=context, history=history):
                        if chunk:
                            yield chunk
                            last_sent = time.time()
                        if time.time() - last_sent > 2:
                            yield " "
                            last_sent = time.time()
                    yield ""
                except Exception as e:
                    print("❌ Lỗi stream:", e)
                    yield f"\n[Stream error: {str(e)}]"
            return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
        else:
            response_text = "".join(stream_llm(question, context=context, history=history))
            return JSONResponse({"answer": response_text})

    except Exception as e:
        print("❌ Lỗi xử lý:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# --------------------------- Upload / RAG tools ---------------------------

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
async def get_text_by_file_id_ep(file_id: str = Form(...)):
    from rag import qdrant, COLLECTION_NAME, Filter, FieldCondition, MatchValue

    filter_ = Filter(
        must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
    )
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_,
        with_payload=True,
        with_vectors=False,
        limit=1000
    )

    texts = [p.payload.get("text", "") for p in points]
    return {"text": "\n\n".join([t for t in texts if t])}

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
    except Exception as e:
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
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            should_use = debug_multilingual_search(question)

        debug_output = f.getvalue()
        config = get_current_config()

        return JSONResponse({
            "question": question,
            "should_use_rag": should_use,
            "debug_output": debug_output,
            "current_config": config
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/rag_config")
def get_rag_config_ep():
    try:
        return JSONResponse(get_current_config())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
