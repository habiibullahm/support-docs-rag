from pathlib import Path

import numpy as np
import os

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer

load_dotenv()
DOCUMENT_PATH = Path("data/support.txt")

def load_document():
    return DOCUMENT_PATH.read_text(encoding="utf-8")

def chunk_document(text: str):
    sections = text.split("\n\n")

    chunks = []
    current_chunk = []

    for section in sections:
        if section.strip():
            current_chunk.append(section.strip())

        if len(current_chunk) == 2:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def generate_answer(question: str, contexts: list[str]):
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    context_text = "\n\n".join(contexts)

    messages = [
        {
            "role": "system",
            "content": """
Kamu adalah customer support assistant.

Aturan:
- Jawab dalam bahasa Indonesia.
- Gunakan hanya informasi dari CONTEXT.
- Jangan mengarang informasi.
- Jika jawabannya tidak ada di CONTEXT, katakan:
  "Informasi tersebut tidak tersedia dalam dokumentasi."
- Jawab dengan singkat dan jelas.
""",
        },
        {
            "role": "user",
            "content": f"""
CONTEXT:
{context_text}

QUESTION:
{question}
""",
        },
    ]

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer


def main():
    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    document = load_document()

    chunks = chunk_document(document)

    print(f"\nLoaded {len(chunks)} chunks.")

    embeddings = model.encode(chunks)

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        question_embedding = model.encode(question)
        results = []

        for index, chunk_embedding in enumerate(embeddings):
            score = cosine_similarity(question_embedding, chunk_embedding)
            results.append({"chunk": chunks[index], "score": float(score)})

        results.sort(key=lambda result: result["score"], reverse=True)

        print("\nTop results:\n")
        for result in results[:3]:
            print(f"Score: {result['score']:.4f}")
            print(result["chunk"])
            print("-" * 60)

        contexts = [result["chunk"] for result in results[:3]]
        answer = generate_answer(question, contexts)

        print("\nAI Answer:\n")
        print(answer)
if __name__ == "__main__":
    main()
