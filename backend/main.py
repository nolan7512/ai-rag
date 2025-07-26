from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from io import BytesIO
from docx import Document

from rag import chunk_text, store_chunks, search_context
from model import ask_llm

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ cho phép "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(
    tags: List[str] = Form(...),  # 🏷️ Cho phép nhiều tag
    files: List[UploadFile] = File(...)
):
    for file in files:
        contents = await file.read()
        doc = Document(BytesIO(contents))
        text = "\n".join(para.text for para in doc.paragraphs)
        chunks = chunk_text(text)
        # Gán metadata chứa danh sách tag
        store_chunks(chunks, metadata={"tags": tags})
    return {"status": "ok"}

@app.post("/ask")
async def ask(
    question: str = Form(...),
    filter_tag: str = Form(None)
):
    context = search_context(question, filter_tag=filter_tag)
    answer = ask_llm(question, context)
    return {"answer": answer}
