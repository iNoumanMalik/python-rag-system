import os
import faiss
import numpy as np
from google import genai

EMBED_MODEL = "models/gemini-embedding-001" 
GEN_MODEL = "gemini-2.5-flash-lite"  
CHUNK_SIZE = 500
TOP_K = 2


client = genai.Client(
    api_key="AIzaSyDA0wyTy1k_ES1HoAAul2D4VaMDOEmoG8M"
) 

models = client.models.list()
for m in models:
    print(m.name, m.display_name)

def load_documents(folder_path="documents"):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append((file_name, text))
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


def get_embedding(text):
    try:
        result = client.models.embed_content(
            model=EMBED_MODEL, contents=text 
        )
  
        if hasattr(result, "embeddings") and result.embeddings:
            if hasattr(result.embeddings[0], "values"):
                return np.array(result.embeddings[0].values, dtype="float32")
            elif isinstance(result.embeddings[0], list):
                return np.array(result.embeddings[0], dtype="float32")
        return np.array(result.embeddings[0], dtype="float32")
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(768, dtype="float32")  


def build_index(documents):
    all_chunks = []
    metadata = []

    for file_name, text in documents:
        chunks = chunk_text(text)
        print(f"Processing {file_name} with {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            all_chunks.append(embedding)
            metadata.append({"source": file_name, "chunk_id": i, "text": chunk})

    if not all_chunks:
        raise ValueError("No embeddings were generated")

    embeddings_array = np.stack(all_chunks).astype("float32")
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index, metadata


def retrieve(query, index, metadata, top_k=TOP_K):
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []

    for score, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):  
            results.append({"score": float(score), "metadata": metadata[idx]})

    return results


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information found."

    context = "\n\n".join([chunk["metadata"]["text"] for chunk in retrieved_chunks])

    try:
      
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=f"Context:\n{context}\n\nQuestion:\n{query}\n\nPlease answer the question based on the context provided.",
        )
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"


if __name__ == "__main__":
    print("Building index...")


    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' folder. Please add your .txt files there.")
        exit()

    documents = load_documents()
    if not documents:
        print("No .txt files found in the 'documents' folder.")
        exit()

    try:
        index, metadata = build_index(documents)
        print(f"Index ready with {len(metadata)} chunks!")

        while True:
            query = input("\nAsk a question (or type 'exit'): ")
            if query.lower() == "exit":
                break

            retrieved = retrieve(query, index, metadata)

            if not retrieved:
                print("No relevant chunks found.")
                continue

            answer = generate_answer(query, retrieved)

            print("\n--- Retrieved Chunks ---")
            for r in retrieved:
                print(f"\nSource: {r['metadata']['source']} | Score: {r['score']}")
                print(r["metadata"]["text"][:200], "...")

            print("\n--- Answer ---")
            print(answer)
    except Exception as e:
        print(f"Error: {e}")
