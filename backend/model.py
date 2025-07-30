import requests

def stream_llm(question, context="Không có ngữ cảnh."):
    if not context or context == "":
        prompt = question
    else:
        prompt = f"""
    Bạn là trợ lý thông minh. Chỉ sử dụng thông tin trong văn bản sau để trả lời câu hỏi, tránh thêm từ vào văn bản. 
    Nếu không tìm thấy thông tin phù hợp, hãy trả lời: "Tôi không tìm thấy thông tin trong các data nội bộ. Vậy đây là câu hỏi bên ngoài ?"

    Văn bản:
    \"\"\"{context}\"\"\"

    Câu hỏi: {question}
    Trả lời:
    """
    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "nous-hermes2", "prompt": prompt, "stream": True},
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

def judge_question_relevance(question: str, context: str, model_name="nous-hermes2") -> bool:
    """
    Gửi prompt đến LLM để đánh giá mức độ liên quan giữa câu hỏi và context.

    Trả về:
    - True: nếu LLM trả lời 'yes'
    - False: nếu trả lời 'no' hoặc gặp lỗi
    """
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
            "http://host.docker.internal:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=60,
        )
        answer = response.json().get("response", "").strip().lower()
        return "yes" in answer
    except Exception as e:
        print("judge_question_relevance failed:", e)
        return False

