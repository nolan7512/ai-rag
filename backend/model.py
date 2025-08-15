import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "nous-hermes2")

# ====================== Low-level stream to Ollama ======================

def _ollama_stream(prompt: str):
    """
    Gọi Ollama /api/generate ở chế độ stream JSON-lines và yield chuỗi 'response'
    """
    resp = requests.post(
        LLM_API_URL,
        json={"model": LLM_MODEL_NAME, "prompt": prompt, "stream": True},
        stream=True,
        timeout=1200.0,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = line.decode("utf-8")
            # Dòng stream của Ollama thường là JSON nhỏ: {"response":"...","done":false,...}
            if '"response":"' in data:
                start = data.find('"response":"') + len('"response":"')
                end = data.find('","', start)
                yield (data[start:end] if end != -1 else data[start:])
        except Exception:
            # Bỏ qua các mảnh lỗi không phải chuỗi JSON hoàn chỉnh
            continue

# ====================== History formatting (robust) ======================

def _format_history(history: str) -> str:
    """
    Nhận 'history' từ client và chuyển thành transcript gọn cho prompt.
    Ưu tiên JSON messages [{role, content}], fallback sang text thô.
    Nếu parse thất bại, trả về chuỗi rỗng (để tránh lộ JSON thô vào prompt).
    """
    if not history:
        return ""

    s = (history or "").strip()

    def parse_json_payload(txt: str):
        data = json.loads(txt)
        if isinstance(data, dict) and "messages" in data:
            msgs = data["messages"]
        else:
            msgs = data
        lines = []
        for m in msgs:
            role = (m.get("role") or "user").lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role in ("assistant", "bot"):
                lines.append(f"Trợ lý: {content}")
            else:
                lines.append(f"Người dùng: {content}")
        return "\n".join(lines)

    # 1) JSON parse path (kể cả khi bị escape)
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        # thử parse trực tiếp
        try:
            return parse_json_payload(s)
        except Exception:
            pass
        # thử unescape nếu bị chuỗi hoá 2 lần
        try:
            s2 = s.encode("utf-8").decode("unicode_escape")
            return parse_json_payload(s2)
        except Exception:
            pass

    # 2) Fallback: tách text theo pattern role:
    try:
        parts = re.split(r'(?i)(?:^|\n)\s*(Người dùng|user|assistant|bot)\s*:\s*', s)
        # parts = ["", "user", "Xin chào", "assistant", "Chào bạn", ...]
        lines = []
        i = 1
        while i < len(parts):
            role = parts[i].lower()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            if not content:
                i += 2
                continue
            if role in ("assistant", "bot"):
                lines.append(f"Trợ lý: {content}")
            else:
                lines.append(f"Người dùng: {content}")
            i += 2
        if lines:
            return "\n".join(lines)
    except Exception:
        pass

    # 3) Last resort: trả rỗng để khỏi lộ JSON thô
    return ""

# ====================== Prompt builder (no echo history/context) ======================

def _build_chat_prompt(question: str, context: str, history: str) -> str:
    """
    Prompt chat mặc định: có RAG context + history hội thoại nhưng
    YÊU CẦU model KHÔNG in lại history/context ra câu trả lời.
    """
    history_block = _format_history(history) or "(empty)"
    context_block = (context or "").strip() or "(none)"

    rules = (
        "Bạn là trợ lý AI lịch sự, giúp đỡ doanh nghiệp, trả lời ngắn gọn và chính xác.\n"
        "QUAN TRỌNG:\n"
        "- KHÔNG được hiển thị lại 'lịch sử hội thoại' hay 'ngữ cảnh' trong câu trả lời.\n"
        "- Chỉ dùng chúng để hiểu ngữ cảnh hiện tại.\n"
        "- Nếu ngữ cảnh không đủ để trả lời chính xác, nói rõ 'Chưa có đủ thông tin trong dữ liệu nội bộ.'"
        " và KHÔNG bịa.\n"
        "- Không nhắc tới từ 'history', 'context', hay các tiêu đề ẩn bên dưới.\n"
    )

    guide_when_has_context = (
        "- Khi có NGỮ CẢNH, phải ưu tiên thông tin trong NGỮ CẢNH.\n"
        "- Bỏ qua phần dư thừa/không liên quan/có vẻ trùng lặp.\n"
        "- Nếu nhiều đoạn liên quan, tổng hợp súc tích, mạch lạc.\n"
    )

    # Ẩn history/context dưới dạng tham chiếu, không trình bày theo tiêu đề dễ bị model lặp lại
    hidden_refs = f"[HIDDEN_HISTORY]\n{history_block}\n\n[HIDDEN_CONTEXT]\n{context_block}"

    if (context or "").strip():
        prompt = (
            f"{rules}"
            f"{guide_when_has_context}\n"
            f"{hidden_refs}\n\n"
            f"Người dùng: {question}\n"
            f"Trợ lý:"
        )
    else:
        prompt = (
            f"{rules}\n"
            f"{hidden_refs}\n\n"
            f"Người dùng: {question}\n"
            f"Trợ lý:"
        )
    return prompt

# ====================== Public chat/translate APIs ======================

def stream_llm(question: str, context: str = "", history: str = ""):
    """
    Luồng trả lời chuẩn (chat) — dùng khi không phải yêu cầu dịch.
    """
    prompt = _build_chat_prompt(question, context, history)
    yield from _ollama_stream(prompt)

# ---- Translation helpers ----

ZH_ALIASES = ["trung", "tiếng trung", "zh", "chinese", "中文", "汉语", "汉文", "华语"]
VI_ALIASES = ["việt", "tiếng việt", "vn", "vietnamese", "tiếng Việt"]
EN_ALIASES = ["anh", "tiếng anh", "english", "en"]

def detect_target_lang(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ZH_ALIASES):
        return "Chinese"
    if any(k in t for k in VI_ALIASES):
        return "Vietnamese"
    if any(k in t for k in EN_ALIASES):
        return "English"
    # fallback: nếu trong câu có ký tự Hoa, coi như đích là Chinese
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "Chinese"
    return "English"

def build_translation_prompt(source_text: str, target_lang: str, include_original: bool = True) -> str:
    """
    Prompt chuyên dịch, giữ style thân thiện chat.
    """
    note = ""
    if include_original:
        if target_lang == "Chinese":
            note = "（原文：{src}）"
        elif target_lang == "Vietnamese":
            note = "(Nguyên văn: {src})"
        else:
            note = "(Original: {src})"

    guide = (
        f"You are a professional translator. Translate the text to {target_lang} with natural tone for chat apps.\n"
        f"- Keep names/brands as-is.\n"
        f"- Make it concise and friendly.\n"
        f"- Preserve greeting tone if any.\n"
    )

    # Để đơn giản, cho model sinh phần dịch; không ép format cứng dòng (tránh lỗi stream).
    return f"""{guide}

[SOURCE]
\"\"\"{source_text}\"\"\"

[OUTPUT]
"""

def stream_translate(source_text: str, target_lang: str, include_original: bool = True):
    prompt = build_translation_prompt(source_text, target_lang, include_original)
    # Model sinh bản dịch; nếu muốn chèn (原文...) hậu kỳ thì làm ở FE.
    for chunk in _ollama_stream(prompt):
        yield chunk

# ====================== History helpers for FE ======================

def get_last_answer_from_history(history: str) -> str:
    """
    Trả về message mới nhất của trợ lý từ history.
    Hỗ trợ:
      - JSON messages [{role, content}, ...]
      - Plain text: ...\nTrợ lý: xxx / bot: xxx / assistant: xxx
    """
    if not history:
        return ""

    # Thử JSON trước
    try:
        data = json.loads(history)
        msgs = data["messages"] if isinstance(data, dict) and "messages" in data else data
        if isinstance(msgs, list):
            for m in reversed(msgs):
                role = (m.get("role") or "").lower()
                if role in ("assistant", "bot"):
                    return (m.get("content") or "").strip()
    except Exception:
        pass

    # Plain text fallback
    lowered = history.lower()
    markers = ["trợ lý:", "assistant:", "bot:"]
    last_pos = -1
    marker_used = ""
    for mk in markers:
        p = lowered.rfind(mk)
        if p > last_pos:
            last_pos = p
            marker_used = mk
    if last_pos != -1:
        return history[last_pos + len(marker_used):].strip()
    return ""

# ====================== LLM Judge (used by RAG) ======================

def judge_question_relevance(question: str, context: str) -> bool:
    """
    Được RAG gọi trong 'gray zone' để hỏi LLM xem câu hỏi có thật sự liên quan context không.
    Trả về True/False.
    """
    prompt = f"""
Bạn là hệ thống đánh giá truy xuất ngữ cảnh.

Dưới đây là các đoạn văn bản có thể liên quan đến câu hỏi:
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
            timeout=1200.0,
        )
        response.raise_for_status()
        answer = (response.json().get("response") or "").strip().lower()
        return "yes" in answer
    except Exception as e:
        print("judge_question_relevance failed:", e)
        return False
