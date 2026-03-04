# rag.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------
# 1️⃣ Embeddings for retrieval
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
documents = []
document_embeddings = None
index = None

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

def create_index(text):
    global documents, document_embeddings, index
    documents = chunk_text(text)
    document_embeddings = embedding_model.encode(documents)
    dim = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(document_embeddings))

def retrieve(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

# -------------------------------
# 2️⃣ LLM Generation (Flan-T5-Base)
# -------------------------------
llm_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

def generate_answer(query, context_chunks):
    # Concatenate top_k chunks
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the question using the context below.

Context:
{context}

Question: {query}
Answer:
"""

    # Tokenize with truncation to fit base model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer