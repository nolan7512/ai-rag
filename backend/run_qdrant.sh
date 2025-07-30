#!/bin/bash

# Tải Qdrant nếu chưa có
if [ ! -d "qdrant" ]; then
  echo "Downloading Qdrant..."
  wget https://github.com/qdrant/qdrant/releases/download/v1.9.1/qdrant-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
  mv qdrant-x86_64-unknown-linux-gnu qdrant
  rm qdrant-x86_64-unknown-linux-gnu.tar.gz
fi

# Chạy Qdrant
cd qdrant
echo "Starting Qdrant on port 6333..."
./qdrant
