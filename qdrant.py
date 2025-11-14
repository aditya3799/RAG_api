# pagewise_chunking_qdrant.py

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import pdfplumber, re, uuid
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv
# -------- CONFIG --------
PDF_PATH = "Boeing B737 Manual.pdf"
COLLECTION = "boeing737_manual_pagewise"
QDRANT_URL = os.getenv("QDRANT_URL")  # or your Qdrant Cloud URL
MODEL_NAME = "BAAI/bge-large-en-v1.5"
api=os.getenv("QDRANT_API_KEY")
CHUNK_WORDS = 300   # each sub-chunk size within a page
OVERLAP = 80        # overlapping words between chunks
BATCH_SIZE = 8
# ------------------------

# -------- INIT --------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(url=QDRANT_URL,api_key=api)

def ensure_collection(dim):
    try:
        qdrant.get_collection(COLLECTION)
    except:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config={"size": dim, "distance": "Cosine"}
        )

def clean_text(t):
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunk_text(text, chunk_size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# -------- PAGE EXTRACTION --------
pages = []
print(f"Extracting text from {PDF_PATH}...")
with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if txt:
            pages.append((i, txt))

print(f"✅ Loaded {len(pages)} pages from manual.")

# -------- CHUNK + EMBED + STORE --------
all_chunks = []
for page_num, text in pages:
    chunks = chunk_text(text)
    for j, c in enumerate(chunks):
        all_chunks.append({
            "id": str(uuid.uuid4()),
            "text": c,
            "meta": {
                "page": page_num,
                "chunk_index": j,
                "source": PDF_PATH
            }
        })

print(f"Prepared {len(all_chunks)} total chunks.")

dim = model.get_sentence_embedding_dimension()
ensure_collection(dim)

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding & uploading"):
    batch = all_chunks[i:i+BATCH_SIZE]
    texts = [b["text"] for b in batch]
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    points = [
        PointStruct(
            id=b["id"],
            vector=e.tolist(),
            payload=b["meta"] | {"text": b["text"]}
        )
        for b, e in zip(batch, emb)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)

print(" All pages processed and uploaded to Qdrant.")
