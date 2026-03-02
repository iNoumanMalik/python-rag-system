# Simple RAG Prototype (Python + FAISS + OpenAI)

A minimal Retrieval-Augmented Generation (RAG) system built using:

- Python
- OpenAI API (Embeddings + LLM)
- FAISS (Vector Search)
- Plain text files as document source
- Optional FastAPI endpoint

This project demonstrates the complete RAG pipeline:
Document → Chunking → Embedding → Vector Index → Retrieval → LLM Generation

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) improves LLM responses by retrieving relevant context from external documents before generating an answer.

Pipeline:

User Query  
↓  
Embedding  
↓  
Vector Similarity Search (FAISS)  
↓  
Top-K Relevant Chunks  
↓  
LLM Prompt (Context + Query)  
↓  
Final Answer  

---

## 📁 Project Structure

```

rag_project/
│
├── documents/
│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.txt
│
├── rag.py
├── requirements.txt
└── README.md

````

---

## ⚙️ Tech Stack

- Python 3.9+
- OpenAI API
- FAISS (CPU)
- NumPy
- FastAPI (optional)

---

## 🚀 Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd rag_project
````

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set OpenAI API Key

```bash
export OPENAI_API_KEY="your_api_key_here"      # macOS/Linux
set OPENAI_API_KEY="your_api_key_here"         # Windows
```

---

## 📚 Add Documents

Place 3–5 `.txt` files inside the `documents/` folder.

Example:

* doc1.txt
* doc2.txt
* doc3.txt

---

## ▶️ Run the RAG Script

```bash
python rag.py
```

You’ll see:

```
Building index...
Index ready!

Ask a question:
```

Type your question and receive:

* Retrieved chunks
* Similarity scores
* Generated answer

---

## 🌐 Optional: Run as API (FastAPI)

Start server:

```bash
uvicorn rag:app --reload
```

Test endpoint:

POST `/ask`

```json
{
  "query": "What is RAG?"
}
```

Response:

```json
{
  "answer": "...",
  "retrieved_chunks": [...]
}
```

---

## 🏗️ Architecture Overview

### 1️⃣ Data Preparation

* Load text files
* Split into fixed-size chunks (500 chars)

### 2️⃣ Indexing Phase

* Generate embeddings for each chunk
* Store vectors in FAISS
* Save metadata (source, chunk_id, text)

### 3️⃣ Retrieval Phase

* Convert query to embedding
* Retrieve Top-K most similar chunks

### 4️⃣ Generation Phase

* Build prompt using retrieved context
* Generate final answer from LLM

---

## 🔧 Configuration

Inside `rag.py`:

```python
CHUNK_SIZE = 500
TOP_K = 2
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
```

You can modify:

* Chunk size
* Top-K retrieval
* Embedding model
* Generation model

---

## 📌 Features

* Simple character-based chunking
* FAISS L2 similarity search
* Metadata tracking
* Top-K configurable
* Optional REST API
* Clean modular structure

---

## 🚀 Possible Improvements

* Token-based chunking (tiktoken)
* Cosine similarity instead of L2
* Save & load FAISS index (persistence)
* Hybrid search (BM25 + vector)
* Streaming responses
* Clean Architecture refactor
* Dockerization
* Add evaluation metrics

---

## 📖 Learning Objectives

This project demonstrates:

* How embeddings work
* How vector databases operate
* How RAG architecture functions internally
* How LLMs use retrieved context
* End-to-end AI system integration


