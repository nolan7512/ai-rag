from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PointIdsList
import uuid, os
import requests
from qdrant_client.http.exceptions import UnexpectedResponse
from model import judge_question_relevance
from datetime import datetime, timezone  
from docx import Document
from io import BytesIO
from fastapi import UploadFile
from typing import List

# Chay tren docker thi bo # ra
import nltk
# nltk.download("punkt")
from nltk.tokenize import sent_tokenize
nltk.download("punkt_tab")

# 1. Load model base
model = INSTRUCTOR("hkunlp/instructor-base")  # đã thay đổi từ large → base
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  # lấy từ biến môi trường hoặc mặc định là localhost
# 2. Kết nối Qdrant
# qdrant = QdrantClient(host="qdrant", port=6333, timeout=120.0)
qdrant = QdrantClient(host=QDRANT_HOST, port=6333, timeout=120.0)

# 3. Khai báo collection và embed_dim phù hợp với instructor-base
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kb")  # lấy từ biến môi trường hoặc mặc định là "kb"
EMBED_DIM = 768  # instructor-base đầu ra là 768 dimensions

# Chỉ tạo collection nếu chưa tồn tại
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

def should_use_rag(question: str, low_threshold=0.15, high_threshold=0.4) -> bool:
    """
    Đánh giá nên dùng RAG hay không dựa trên embedding similarity và fallback LLM nếu cần.
    """
    question_vec = model.encode([["Represent the question:", question]])[0]

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vec,
        limit=10
    )

    if not hits:
        return False

    top_score = hits[0].score
    context = "\n".join([hit.payload["text"] for hit in hits])
    print("Top score:", top_score)
    if top_score >= high_threshold:
        return True
    if top_score < low_threshold:
        return False

    # Vùng xám → fallback dùng LLM để đánh giá
    return judge_question_relevance(question, context)
    
def embed_chunks(chunks):
    # return model.encode([["Represent the legal document:", ch] for ch in chunks])
     # chọn prompt phù hợp loại tài liệu
    return model.encode([["Represent the internal document for retrieval:", ch] for ch in chunks])

def chunk_by_sentences(text, max_words=300):
    sentences = sent_tokenize(text)
    chunks, current = [], []

    for sent in sentences:
        current.append(sent)
        if len(" ".join(current).split()) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def store_chunks(chunks, metadata=None):
    embeddings = embed_chunks(chunks)
    file_id = metadata.get("file_id", uuid.uuid4().hex)
    file_name = metadata.get("filename", "unknown.docx")
    tags = metadata.get("tags", [])
    uploaded_at = datetime.now(timezone.utc).isoformat()
    source = metadata.get("source", "file")

    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=vec,
            payload={
                **(metadata or {}),
                "file_id": file_id,
                "filename": file_name,
                "tags": tags,
                "uploaded_at": uploaded_at,
                "source": source,
                "text": chunk,
            }
        )
        for vec, chunk in zip(embeddings, chunks)
    ]
    qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)
    
    

# def chunk_text(text, max_words=512, overlap=50):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words - overlap):
#         chunks.append(" ".join(words[i:i+max_words]))
#     return chunks

# def store_chunks(chunks, metadata=None):
#     embeddings = embed_chunks(chunks)
#     points = [
#         PointStruct(id=uuid.uuid4().hex, vector=vec, payload={**(metadata or {}), "text": chunk})
#         for vec, chunk in zip(embeddings, chunks)
#     ]
#     qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

def search_context(query, filter_tag=None, top_k=10, score_threshold=0.4):
    query_vec = model.encode([["Represent the question:", query]])[0]

    filter_query = {"must": [{"key": "tags", "match": {"value": filter_tag}}]} if filter_tag else None

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        query_filter=filter_query
    )

    # Nếu kết quả theo tag ít quá thì fallback (không tag)
    if filter_tag and len(hits) < 3:
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k
        )

    # Ưu tiên giữ nhiều thông tin hơn
    relevant_chunks = [h.payload["text"] for h in hits if h.score > score_threshold]

    return "\n".join(relevant_chunks)

def clear_collection():
    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        return {"message": "✅ Collection đã được reset thành công."}
    except UnexpectedResponse as e:
        return {"error": str(e)}

def export_collection():
    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            with_payload=True,
            with_vectors=True,
            limit=10000
        )
        return [p.dict() for p in points]
    except Exception as e:
        return {"error": str(e)}

def import_collection_from_raw(raw):
    try:
        points = []
        for p in raw:
            # đảm bảo id là string UUID
            if not isinstance(p.get("id"), str):
                p["id"] = uuid.uuid4().hex
            points.append(PointStruct(**p))

        qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)
        return {"message": f"✅ Đã import thành công {len(points)} điểm vào collection."}
    except Exception as e:
        return {"error": str(e)}

def list_uploaded_files():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        limit=10000,
    )

    seen = {}
    for p in points:
        payload = p.payload
        file_id = payload.get("file_id")
        if file_id and file_id not in seen:
            seen[file_id] = {
                "file_id": file_id,
                "filename": payload.get("filename", ""),
                "tags": payload.get("tags", []),
                "uploaded_at": payload.get("uploaded_at", ""),
                "source": payload.get("source", "file"),
            }

    return list(seen.values())


def delete_file_by_id(file_id: str):
    # Tìm tất cả các điểm (chunk) có file_id này
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
        with_payload=False,
        with_vectors=False,
        limit=10000,
        scroll_filter=filter,
    )

    ids_to_delete = [pt.id for pt in points]

    if ids_to_delete:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids_to_delete)  # ✅ đúng format
        )

def update_file(file: UploadFile, file_id: str, tags: List[str] = []):
    try:
        delete_file_by_id(file_id)
        contents = file.file.read()
        doc = Document(BytesIO(contents))
        text = "\n".join(p.text for p in doc.paragraphs)
        chunks = chunk_by_sentences(text)

        store_chunks(chunks, metadata={
            "file_id": file_id,
            "filename": file.filename,
            "tags": tags,
            "source": "file",
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        })

        return {"message": f"File '{file.filename}' updated successfully"}
    except Exception as e:
        raise RuntimeError(f"Lỗi update file: {str(e)}")
def update_text_by_file_id(file_id: str, new_text: str, tags: List[str] = []):
    try:
        delete_file_by_id(file_id)
        chunks = chunk_by_sentences(new_text)
        store_chunks(chunks, metadata={
            "file_id": file_id,
            "filename": f"text-{file_id}",
            "tags": tags,
            "source": "text",
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        })
        return {"message": f"Text {file_id} updated successfully"}
    except Exception as e:
        raise RuntimeError(f"Lỗi update text: {str(e)}")


def get_text_by_file_id(file_id: str) -> str:
    filter = Filter(
        must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
    )
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter,
        with_payload=True,
        with_vectors=False,
        limit=10000
    )
    sorted_chunks = sorted(points, key=lambda p: p.id)  # Giữ thứ tự
    return "\n".join(p.payload.get("text", "") for p in sorted_chunks)



