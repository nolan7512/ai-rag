#!/bin/bash

# Kích hoạt virtual environment
if [ ! -d "venv" ]; then
  echo "Creating virtualenv..."
  python3 -m venv venv
fi

source venv/bin/activate

# Cài requirements nếu chưa có
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Chạy backend (FastAPI)
echo "Starting backend on http://localhost:8000 ..."
uvicorn main:app --host 0.0.0.0 --port 8000
