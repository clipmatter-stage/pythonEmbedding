from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import requests
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Secure configuration
QDRANT_URL = "https://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.iohttps://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzY2NjcwMTM1fQ.c2bNP_BNXhVhM3fApCyKHw7SGV1ITyDMDtT5s1WlGW8"
COLLECTION_NAME = "text_embeddings"

def get_qdrant_headers():
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers

def create_collection():
    """Create the text_embeddings collection if it doesn't exist"""
    try:
        check_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
        response = requests.get(check_url, headers=get_qdrant_headers())
        
        if response.status_code == 404:
            print(f"Creating collection '{COLLECTION_NAME}'...", flush=True)
            create_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
            data = {
                "vectors": {
                    "size": 384,
                    "distance": "Cosine"
                },
                # ✅ Add indexing for faster searches
                "optimizers_config": {
                    "indexing_threshold": 10000
                }
            }
            create_response = requests.put(create_url, json=data, headers=get_qdrant_headers())
            create_response.raise_for_status()
            print(f"Collection '{COLLECTION_NAME}' created successfully", flush=True)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)
        raise

create_collection()

@app.post("/embed")
async def embed(data: dict):
    text = data.get("text", "")
    video_id = data.get("video_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Generate embedding
    vector = model.encode(text).tolist()
    vector_id = f"video_{video_id}_{uuid.uuid4()}" if video_id else str(uuid.uuid4())

    # ✅ Better metadata structure for transcripts
    metadata = {
        "text": text,  # Store full text or use external storage for very long texts
        "text_length": len(text),
        "created_at": datetime.utcnow().isoformat(),
        "source": "embedding_api"
    }
    
    if video_id:
        metadata["video_id"] = video_id
        # ✅ Extract useful transcript info
        metadata["speaker_count"] = len(set(
            line.split("(ID:")[1].split(")")[0] 
            for line in text.split("\n") if "(ID:" in line
        ))

    payload = {
        "points": [{
            "id": vector_id,
            "vector": vector,
            "payload": metadata
        }]
    }
    
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json=payload,
            headers=get_qdrant_headers(),
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
      
    return {
        "id": vector_id,
        "vector_dimension": len(vector),
        "metadata": metadata,
        "status": "success"
    }

@app.post("/search")
async def search(data: dict):
    """Search for similar embeddings"""
    query_text = data.get("query", "")
    top_k = data.get("top_k", 5)
    video_id_filter = data.get("video_id")
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    query_vector = model.encode(query_text).tolist()
    
    search_payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False  # Don't return vectors to save bandwidth
    }
    
    # ✅ Add filtering by video_id if provided
    if video_id_filter:
        search_payload["filter"] = {
            "must": [
                {"key": "video_id", "match": {"value": video_id_filter}}
            ]
        }
    
    try:
        response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json=search_payload,
            headers=get_qdrant_headers(),
            timeout=10
        )
        response.raise_for_status()
        results = response.json()
        
        return {
            "query": query_text,
            "results": [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "text": r["payload"].get("text", "")[:200],  # First 200 chars
                    "video_id": r["payload"].get("video_id"),
                    "created_at": r["payload"].get("created_at")
                }
                for r in results.get("result", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        response = requests.get(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            headers=get_qdrant_headers(),
            timeout=5
        )
        return {
            "status": "healthy",
            "qdrant_connected": response.ok,
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"🚀 Starting FastAPI server on port {port}...", flush=True)
    print(f"📊 Qdrant URL: {QDRANT_URL}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
