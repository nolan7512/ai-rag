# RAG DOCX React App (Nous Hermes 2)

## 🧠 Mô tả
Trích xuất kiến thức từ file `.docx` và hỏi LLM (Nous Hermes 2 - Mistral 7B) qua giao diện React.

## 🚀 Cách chạy

### 1. Cài đặt backend + vectorDB
```bash
docker-compose up --build
```


### xóa, build, run
docker-compose down -v
docker-compose build --no-cache
docker-compose up


### chạy từ lần 2 không build lại 
	docker-compose up -d
    docker-compose logs -f

### 2. Chạy frontend
```bash
cd frontend
npm install
npm start
```

### 3. Mở ứng dụng:
- Giao diện: http://localhost:3000
- Backend API: http://localhost:8000/docs

### 4. Cài riêng Ollama (local)
```bash
ollama run nous-hermes2-mistral
```

## 📌 Chức năng
- Upload nhiều file `.docx`
- Gửi câu hỏi → Trích xuất context → Trả lời

✅ Sử dụng `InstructorEmbedding` cho chất lượng truy vấn tốt hơn
✅ Hỗ trợ multi-doc upload
✅ Frontend React đơn giản, dễ mở rộng
