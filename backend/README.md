
# 🧠 RAG DOCX Assistant (Fullstack)

Trích xuất kiến thức từ file `.docx`, lưu vào Qdrant, và trả lời câu hỏi qua React frontend bằng mô hình LLM (Nous Hermes 2).

---

## ⚙️ Yêu cầu

- Docker Desktop (Windows/macOS)
- Đã cài [Ollama](https://ollama.com/) và pull model:
  ```bash
  ollama run nous-hermes2

  ```

---

## 🚀 Cách chạy dự án (trên Windows)

### 1. Chạy Ollama trên host

```bash
ollama run nous-hermes2

```

Ollama cần chạy sẵn ở `http://localhost:11434`.

---

### 2. Chạy backend + Qdrant (Docker)

```bash
docker-compose up --build
```

- Qdrant chạy tại: `http://localhost:6333`
- Backend FastAPI tại: `http://localhost:8000`

---

### 3. Chạy frontend React

```bash
cd frontend
npm install
npm start
```

- Giao diện truy cập tại: [http://localhost:3000](http://localhost:3000)

---

## 📦 Cấu trúc thư mục

```
rag-docx-full/
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── model.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── index.js
```

---

## ✅ Kết nối backend ↔ Ollama trên Windows

- Backend sử dụng:
  ```python
  requests.post("http://host.docker.internal:11434/api/generate")
  ```
- `host.docker.internal` là **mặc định hoạt động trên Windows**.

---

## ❓ Hỏi đáp

- **Tôi dùng Linux thì sao?**
  → Sửa `model.py` → dùng IP: `http://172.17.0.1:11434`

- **Muốn dùng Ollama bằng docker-compose luôn thì sao?**
  → Bạn cần sửa `docker-compose.yml` thêm service `ollama`.

---

Chúc bạn dùng vui 🎉 Nếu cần mở rộng để hỗ trợ `.pdf`, `.xlsx` hoặc multi-model (OpenAI, Claude...), cứ nhắn!
