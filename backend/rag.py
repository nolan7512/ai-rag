from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid, os

model = INSTRUCTOR("hkunlp/instructor-large")
qdrant = QdrantClient(host="qdrant", port=6333, timeout=120.0)
COLLECTION_NAME = "kb"

qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

def embed_chunks(chunks):
    return model.encode([["Represent the legal document:", ch] for ch in chunks])

def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def store_chunks(chunks):
    embeddings = embed_chunks(chunks)
    points = [
        PointStruct(id=uuid.uuid4().hex, vector=vec, payload={"text": chunk})
        for vec, chunk in zip(embeddings, chunks)
    ]
    qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

def search_context(query, top_k=3):
    query_vec = model.encode([["Represent the question:", query]])[0]
    hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_vec, limit=top_k)
    return "\n".join(h.payload["text"] for h in hits)
