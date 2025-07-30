FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Cài Python packages + tải dữ liệu nltk punkt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt \
    && python -m nltk.downloader punkt_tab

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
