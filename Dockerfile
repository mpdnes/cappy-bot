# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (small, ~90MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# Note: Qwen model (~3GB) will download on first container startup to avoid build OOM

# Copy application files
COPY handbook.md .
COPY build_index.py .
COPY app.py .

# Build ChromaDB index at build time (included in container image)
RUN python build_index.py

# Expose port (Cloud Run will override with $PORT)
EXPOSE 8080

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 60 --preload app:app
