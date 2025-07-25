import requests

def ask_llm(question, context):
    prompt = f"""Bạn là trợ lý ảo. Trả lời câu hỏi dựa trên văn bản sau:

{context}

Câu hỏi: {question}
Trả lời:"""
    r = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "nous-hermes2", "prompt": prompt, "stream": False}
    )
    return r.json()["response"]
