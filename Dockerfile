FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models without loading into RAM (avoids OOM during build)
# This saves files to /root/.cache/huggingface and bakes them into the image
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2'); \
    snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct')"

COPY handbook.md .
COPY build_index.py .
COPY app.py .

# Build the ChromaDB index
RUN python build_index.py

# Expose port (Cloud Run will override with $PORT)
EXPOSE 8080

# Run with gunicorn for production
# timeout 0 prevents killing app during model load (models are baked in, just need RAM load time)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
