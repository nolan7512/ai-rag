from fastapi import FastAPI, UploadFile, File
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
async def upload(files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        doc = Document(BytesIO(contents))
        full_text = [para.text for para in doc.paragraphs]
        text = "\n".join(full_text)
        chunks = chunk_text(text)
        store_chunks(chunks)
    return {"status": "ok"}

@app.post("/ask")
async def ask(question: str = File(...)):
    context = search_context(question)
    answer = ask_llm(question, context)
    return {"answer": answer}
