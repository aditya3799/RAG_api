# fetch_from_qdrant.py

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json, logging
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
load_dotenv()
# ------------- CONFIG -------------
COLLECTION = "boeing737_manual_pagewise"
QDRANT_URL = os.getenv("QDRANT_URL") 
MODEL_NAME = "BAAI/bge-large-en-v1.5"
api=os.getenv("QDRANT_API_KEY")
BM25_DATA_FILE = "boeing_manual_data.json"
TOP_K = 5
# ---------------------------------

# Initialize model & Qdrant client
print("Loading model and connecting to Qdrant...")
model = SentenceTransformer(MODEL_NAME)
client= QdrantClient(url=QDRANT_URL,api_key=api)
# -------- BM25 SETUP --------
def load_bm25_data():
    """
    Load all text from Qdrant and build a BM25 index.
    """
    try:
        logging.info("Building BM25 index from Qdrant...")
        scroll_result = client.scroll(collection_name=COLLECTION, scroll_filter=None, limit=1000)
        documents, pages = [], []

        for point in scroll_result[0]:
            text = point.payload.get("text", "")
            page = point.payload.get("page", "Unknown")
            documents.append(text)
            pages.append(page)

        tokenized_corpus = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        # Cache data locally
        with open(BM25_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"documents": documents, "pages": pages}, f, ensure_ascii=False, indent=2)

        logging.info(f"✅ BM25 index built with {len(documents)} documents.")
        return bm25, documents, pages

    except Exception as e:
        logging.error(f"Error loading BM25 data: {e}")
        return None, [], []


bm25, documents, pages = load_bm25_data()

# -------- Dense Retrieval --------
def get_query_embedding(query):
    return model.encode([query])[0]


def retrieve_dense(query, top_k=TOP_K):
    query_vector = get_query_embedding(query)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k
    )
    return [
        {
            "page": hit.payload.get("page", "Unknown"),
            "text": hit.payload.get("text", ""),
            "score": hit.score
        }
        for hit in results
    ]


# -------- BM25 Retrieval --------
def retrieve_bm25(query, top_k=TOP_K):
    if not bm25:
        logging.warning("BM25 not available.")
        return []

    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "page": pages[i],
            "text": documents[i],
            "score": scores[i]
        }
        for i in top_indices
    ]


# -------- Reciprocal Rank Fusion (RRF) --------
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    combined_scores = {}

    def update_scores(results):
        for rank, doc in enumerate(results, 1):
            page = doc["page"]
            score = 1 / (rank + k)
            combined_scores[page] = combined_scores.get(page, 0) + score

    update_scores(dense_results)
    update_scores(bm25_results)

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)


# -------- Hybrid Retrieval --------
def retrieve_hybrid(query, top_k=TOP_K):
    dense_results = retrieve_dense(query, top_k)
    bm25_results = retrieve_bm25(query, top_k)

    if not dense_results and not bm25_results:
        return []

    fused_results = reciprocal_rank_fusion(dense_results, bm25_results)

    # Merge with text payloads for display
    results_with_text = []
    for page, score in fused_results[:top_k]:
        # Find the first occurrence of this page text
        for doc in dense_results + bm25_results:
            if doc["page"] == page:
                results_with_text.append({
                    "page": page,
                    "score": score,
                    "text": doc["text"]
                })
                break
    return results_with_text


