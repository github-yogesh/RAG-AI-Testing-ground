from fastapi import FastAPI, UploadFile, File
from rag import create_index, retrieve, generate_answer

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    create_index(text)
    return {"message": "Document indexed successfully."}

@app.get("/ask")
def ask_question(query: str):
    context_chunks = retrieve(query)
    answer = generate_answer(query, context_chunks)
    return {"answer": answer}