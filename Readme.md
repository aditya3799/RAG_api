# 🚀 Boeing 737 Technical Manual RAG API

### Retrieval-Augmented Generation System using Qdrant, Sentence Transformers, BM25, RRF, and Gemini 2.5 Flash

This repository implements a complete **Retrieval-Augmented Generation (RAG)** pipeline designed to answer questions strictly based on the **Boeing 737 Technical Manual**.  
The system uses **PDF → chunk → embed → store → retrieve → generate** stages and returns an answer grounded in the manual along with **relevant manual page numbers**.

---

## 🧠 Overview

This RAG system is built using:

- **Python**
- **FastAPI** (API)
- **SentenceTransformers** (embeddings)
- **BAAI/bge-large-en-v1.5** (dense semantic embeddings)
- **BM25** (sparse keyword search)
- **Reciprocal Rank Fusion (RRF)** (hybrid retrieval)
- **Qdrant Cloud** (vector database)
- **Gemini 2.5 Flash** (LLM for grounded answer generation)
- **LangChain** (prompt templating)

You can ask **any Boeing 737 operational or performance-related question**, and the system will:

1. Retrieve the most relevant page chunks  
2. Fuse results using RRF  
3. Generate a concise 1–2 sentence answer  
4. Provide page citations (1-based index)

---

## 📁 Repository Structure

```
boeing/
├── main.py                     # FastAPI server & response generation
├── qdrant.py                   # PDF → Chunk → Embed → Upload to Qdrant
├── question.py                 # Hybrid retrieval (Dense + BM25 + RRF)
├── Boeing B737 Manual.pdf      
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .env.example                # Example env file
└── README.md                   # This file
```

---

## 🏗️ Full Architecture Explanation

```
PDF Manual
    ↓
PDF Parsing (pdfplumber)
    ↓
Pagewise cleaning
    ↓
Chunking (300 words, 80 overlap)
    ↓
Embedding (bge-large-en-v1.5)
    ↓
Qdrant Upsert
    ↓
User Query
    ↓
Dense Retrieval (Qdrant cosine similarity)
    ↓
Sparse Retrieval (BM25)
    ↓
Reciprocal Rank Fusion
    ↓
Top-k Chunks
    ↓
Gemini 2.5 Flash (strict grounded prompt)
    ↓
Final Answer + Page Numbers
```

---

## 🧩 1. PDF Ingestion & Chunking (`qdrant.py`)

### Process:

1. Load PDF using **pdfplumber**
2. Extract & clean text on a *per-page* basis
3. Chunk pages into:
   - **300-word chunks**
   - **80-word overlap**
4. Embed using `BAAI/bge-large-en-v1.5`
5. Store in Qdrant with metadata:
   - `page`
   - `chunk_index`
   - `text`
   - `source`

### Chunking example:

```python
def chunk_text(text, chunk_size=300, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
```

### Key Features:
- ✅ Preserves page-level context
- ✅ Overlapping chunks prevent information loss
- ✅ Batch embedding & upload to Qdrant
- ✅ Handles malformed PDFs gracefully

---

## 🔍 2. Hybrid Retrieval (`question.py`)

### Dense Retrieval

Using cosine similarity from Qdrant's vector store.


### Sparse Retrieval (BM25)

Using `rank_bm25` to return keyword-based matches.


### Final return:

- Top-k fused chunks
- Page numbers
- Text snippets

---

## 🤖 3. Grounded Answer Generation (`main.py`)

### The LLM:
**Gemini 2.5 Flash**

### Prompt rules:

- 1–2 polished sentences
- No step-by-step reasoning
- No quoting chunks
- No hallucination
- Cite only page numbers (Pages: x, y)
- If context missing → provide a suggestive fallback


## 🔧 Running the API

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set environment variables

Create `.env` file:

```env
QDRANT_URL=https://your-cluster.qdrant.cloud:6333
QDRANT_API_KEY=your_qdrant_api_key
Gemini_api_key=AIzaSyDO3RtS0vxjIV_QOfssYX9XON8pPxpipPk
```

### 3️⃣ Ingest PDF (one-time setup)

```bash
python qdrant.py
```

This will:
- Parse `Boeing B737 Manual.pdf`
- Chunk and embed all pages
- Upload to Qdrant Cloud

### 4️⃣ Start FastAPI server

```bash
python main.py
```

or:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server will start at:

```
http://localhost:8000
```

---

## 🌐 API Endpoint Documentation

### POST `/query`

**Request Body:**

```json
{
  "question": "What is the climb limit weight at 2000 ft and 50°C?"
}
```

**Example Response:**

```json
{
  "response": "Based on the dry runway data at 2,000 ft pressure altitude and 50°C, the climb limit weight is 52,200 kg (Pages: 4, 5).",
  "pages": [4, 5]
}
```

### Example cURL:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended flap setting for takeoff?"}'
```

---

## 🧪 Why Hybrid Retrieval?

Pure dense retrieval sometimes fails with:

- Tables
- Performance charts
- Abbreviations
- Non-NLP text
- Numerical value searches

**BM25 helps retrieve:**

- Keywords
- Numbers
- Abbreviations

**RRF fuses both to get:**

- ✔ High recall
- ✔ Stable ranking
- ✔ Lower false positives
- ✔ Minimal irrelevant pages

---

## 🧠 Challenges & Solutions

### 1. PDF tables not parsed cleanly

→ Solved with overlapping chunking + hybrid retrieval.

### 2. LLM hallucinations

→ Solved using strict grounding prompt:
- no quotes
- no invented values
- page-citations-only

### 3. API model mismatch errors

→ Use correct model: `gemini-2.5-flash`

### 4. Retrieval instability

→ RRF ensures robust ranking.

---

## 🚀 Future Improvements

- [ ] Cohere Re-Ranker integration
- [ ] Better table extraction (Camelot/Tabula)
- [ ] Query rewriting (user → manual style query)
- [ ] Chunk merge for large diagrams
- [ ] Response JSON formatting with structured citations
- [ ] Add conversation history/context
- [ ] Deploy to cloud (AWS/GCP/Azure)

---

## 📄 `.env.example`

```env
QDRANT_URL=https://your-qdrant-cluster.cloud:6333
QDRANT_API_KEY=your_qdrant_api_key_here
Gemini_api_key=your_gemini_api_key_here
```


## 👨‍💻 Author

**Aditya Kudale**


# RAG_api
