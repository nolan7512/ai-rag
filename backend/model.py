import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Tải các biến môi trường từ file .env

LLM_API_URL = os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "nous-hermes2")

def stream_llm(question, context="Không có ngữ cảnh."):
    if not context.strip():
        prompt = question
    else:
        prompt = f"""
Bạn là trợ lý thông minh. Chỉ sử dụng thông tin trong văn bản sau để trả lời câu hỏi. 
Nếu không tìm thấy thông tin phù hợp, hãy trả lời: "Tôi không tìm thấy thông tin trong database của mình."

Văn bản:
\"\"\"{context}\"\"\"

Câu hỏi: {question}
Trả lời:
"""
    response = requests.post(
        LLM_API_URL,
        json={"model": LLM_MODEL_NAME, "prompt": prompt, "stream": True},
        stream=True,
    )

    for line in response.iter_lines():
        if line:
            try:
                data = line.decode("utf-8")
                if '"response":"' in data:
                    start = data.find('"response":"') + len('"response":"')
                    end = data.find('","', start)
                    chunk = data[start:end] if end != -1 else data[start:]
                    yield chunk
            except Exception:
                continue

def judge_question_relevance(question: str, context: str) -> bool:
    prompt = f"""
Bạn là hệ thống đánh giá truy xuất ngữ cảnh.

Dưới đây là đoạn thông tin nội bộ:
\"\"\"{context}\"\"\" 

Và câu hỏi:
\"\"\"{question}\"\"\" 

Câu hỏi này có liên quan với đoạn văn bản trên không?

Chỉ trả lời duy nhất một từ:
- Nếu có → "yes"
- Nếu không liên quan → "no"
"""

    try:
        response = requests.post(
            LLM_API_URL,
            json={"model": LLM_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=300.0,
        )
        answer = response.json().get("response", "").strip().lower()
        return "yes" in answer
    except Exception as e:
        print("judge_question_relevance failed:", e)
        return False


