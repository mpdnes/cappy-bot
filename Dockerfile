FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW: Download models without loading into RAM (Prevents OOM) ---
# This saves the files to /root/.cache/huggingface
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2'); \
    snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct')"

COPY handbook.md .
COPY build_index.py .
COPY app.py .

# Build the ChromaDB index
RUN python build_index.py

# Use --timeout 0 to prevent Gunicorn from killing the app during model load
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
