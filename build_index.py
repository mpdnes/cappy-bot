#!/usr/bin/env python3
"""
Build ChromaDB index from handbook.md at container build time.
This runs during Docker build to pre-process the handbook.
"""
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with overlap.
    Tries to break at paragraph boundaries when possible.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, save current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from end of current chunk
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
            current_chunk = overlap_text + " " + para
        else:
            current_chunk += ("\n\n" if current_chunk else "") + para

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def build_index():
    """Build and persist ChromaDB index from handbook.md"""
    print("Loading handbook.md...")
    with open('handbook.md', 'r', encoding='utf-8') as f:
        handbook_text = f.read()

    print(f"Handbook length: {len(handbook_text)} characters")

    # Chunk the handbook
    print("Chunking handbook...")
    chunks = chunk_text(handbook_text, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks")

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    persist_directory = "/app/chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)

    # Delete collection if it exists (for rebuilds)
    try:
        client.delete_collection("handbook")
    except:
        pass

    collection = client.create_collection(
        name="handbook",
        metadata={"description": "Employee handbook content"}
    )

    # Add chunks to ChromaDB with embeddings
    print("Generating embeddings and storing in ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_id": i, "source": "handbook.md"}]
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")

    print(f"Index built successfully with {len(chunks)} chunks")
    print(f"Persisted to {persist_directory}")

if __name__ == "__main__":
    build_index()
