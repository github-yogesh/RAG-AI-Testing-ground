##Local RAG Legal Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about legal documents locally using Flan-T5-Base and SentenceTransformers.
No OpenAI API keys are required — fully self-contained and CPU-friendly.

Features

Upload legal documents (.txt) and index them for semantic search.

Use SentenceTransformers embeddings + FAISS for fast chunk retrieval.

Answer queries using Flan-T5-Base, leveraging only the retrieved context.

Handles long documents using chunking with overlap.

FastAPI server exposes simple endpoints for uploading and querying documents.

Tech Stack

Python 3.11+

FastAPI – backend server

FAISS – vector similarity search

SentenceTransformers – embeddings

HuggingFace Transformers – Flan-T5-Base LLM

PyTorch – model inference

Python-multipart – file upload handling

Installation
git clone <your_repo_url>
cd legal-rag-local
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
Usage
1️⃣ Run the FastAPI server
uvicorn app:app --reload

The API will run at http://127.0.0.1:8000

2️⃣ Upload a legal document

Endpoint: POST /upload

Example with curl:

curl -X POST "http://127.0.0.1:8000/upload" -F "file=@sample_contract.txt"

Response:

{
  "message": "Document indexed successfully."
}
3️⃣ Ask a question

Endpoint: GET /ask?query=<your question>

Example:

curl "http://127.0.0.1:8000/ask?query=When does the contract expire?"

Response:

{
  "answer": "The contract is valid from January 1, 2026, for a period of two years. Either party can terminate it with 30 days notice, and any extensions must be agreed in writing."
}
Project Structure
legal-rag-local/
│
├── app.py             # FastAPI server
├── rag.py             # RAG logic: chunking, embeddings, retrieval, generation
├── requirements.txt   # Python dependencies
├── sample_contract.txt # Sample legal document for demo
└── README.md
How It Works

Document Upload & Indexing

File is split into chunks (default 400 tokens, 50-token overlap)

Each chunk is embedded using SentenceTransformer

Chunks are stored in a FAISS index for fast semantic retrieval

Query Handling

Query is embedded

Top-k most relevant chunks are retrieved from FAISS

Answer Generation

Concatenate retrieved chunks as context

Flan-T5-Base generates answer based on context + query

Answer returned via API
