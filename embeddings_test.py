from email.mime import text
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import requests
import uuid
import json
import os

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant endpoint - supports both local and cloud
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:3334")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "text_embeddings"

# Helper function to get headers with API key for Qdrant Cloud
def get_qdrant_headers():
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers

# Create collection if it doesn't exist
def create_collection():
    """Create the text_embeddings collection if it doesn't exist"""
    try:
        # Check if collection exists
        check_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
        response = requests.get(check_url, headers=get_qdrant_headers())
        
        if response.status_code == 404:
            # Collection doesn't exist, create it
            print(f"Collection '{COLLECTION_NAME}' not found. Creating...", flush=True)
            create_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
            data = {
                "vectors": {
                    "size": 384,  # all-MiniLM-L6-v2 embeddings
                    "distance": "Cosine"
                }
            }
            create_response = requests.put(create_url, json=data, headers=get_qdrant_headers())
            create_response.raise_for_status()
            print(f"Collection '{COLLECTION_NAME}' created successfully", flush=True)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)

create_collection()

@app.post("/embed")
async def embed(data: dict):
    # log the incoming request
    print(f"Received embed request: {data}")
    text = data.get("text", "")
    video_id = data.get("video_id")  # Optional video ID for metadata

    print("text to embed:", text, flush=True)
    
    if not text:
        return {"error": "Text is required"}, 400
    
    # Generate embedding using sentence-transformers
    print("About to encode...", flush=True)
    vector = model.encode(text).tolist()
    print("Encoding complete",vector, flush=True)

    # Use a unique ID for this embedding
    vector_id = str(uuid.uuid4())

    # Prepare metadata
    metadata = {
        "text": text[:500],  # Store first 500 chars to avoid too large payload
        "text_length": len(text),
        "created_at": str(uuid.uuid1().time)
    }
    
    if video_id:
        metadata["video_id"] = video_id

    # Store embedding in Qdrant using the correct API format
    payload = {
        "points": [
            {
                "id": vector_id,
                "vector": vector,
                "payload": metadata
            }
        ]
    }
    
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json=payload,
            headers=get_qdrant_headers()
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # If collection not found, try to create it and retry
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"Collection not found during embed. Attempting to create...", flush=True)
            create_collection()
            # Retry the request
            try:
                response = requests.put(
                    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                    json=payload,
                    headers=get_qdrant_headers()
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as retry_error:
                return {
                    "error": f"Failed to store in Qdrant after creating collection: {str(retry_error)}",
                    "vector_id": vector_id,
                    "embedding": vector
                }
        else:
            return {
                "error": f"Failed to store in Qdrant: {str(e)}",
                "vector_id": vector_id,
                "embedding": vector
            }
      
    return {
        "id": vector_id,
        "embedding": vector,
        "qdrant_response": response.json() if response.ok else response.text
    }


@app.post("/search")
async def search(data: dict):
    """Search for similar embeddings in Qdrant"""
    query_text = data.get("text", "")
    limit = data.get("limit", 10)
    video_id_filter = data.get("video_id")
    
    if not query_text:
        return {"error": "Query text is required"}, 400
    
    # Generate embedding for query text
    query_vector = model.encode(query_text).tolist()
    
    # Build search request
    search_request = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True
    }
    
    # Add filter if video_id is specified
    if video_id_filter:
        search_request["filter"] = {
            "must": [
                {
                    "key": "video_id",
                    "match": {"value": video_id_filter}
                }
            ]
        }
    
    try:
        response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json=search_request,
            headers=get_qdrant_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # If collection not found, try to create it
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"Collection not found during search. Creating collection...", flush=True)
            create_collection()
            return {
                "error": "Collection was not found and has been created. Please retry your search.",
                "collection_created": True
            }
        return {"error": f"Search failed: {str(e)}"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        qdrant_response = requests.get(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            headers=get_qdrant_headers()
        )
        qdrant_status = "connected" if qdrant_response.ok else "disconnected"
    except:
        qdrant_status = "disconnected"
    
    return {
        "status": "healthy",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimensions": 384,
        "qdrant_status": qdrant_status,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION_NAME
    }


# Terraforming collection example
def create_terraforming_collection():
    """Create terraforming collection with filtering examples"""
    collection_url = f"{QDRANT_URL}/collections/terraforming"
    
    # Create collection
    collection_data = {
        "vectors": {
            "size": 4,
            "distance": "Dot"
        }
    }
    requests.put(collection_url, json=collection_data)
    
    # Add points
    points_url = f"{QDRANT_URL}/collections/terraforming/points"
    points_data = {
        "points": [
            {
                "id": 1,
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"land": "forest", "color": "green", "life": True, "humidity": 40}
            },
            {
                "id": 2,
                "vector": [0.2, 0.3, 0.4, 0.5],
                "payload": {"land": "lake", "color": "blue", "life": True, "humidity": 100}
            },
            {
                "id": 3,
                "vector": [0.3, 0.4, 0.5, 0.6],
                "payload": {"land": "steppe", "color": "green", "life": False, "humidity": 25}
            },
            {
                "id": 4,
                "vector": [0.4, 0.5, 0.6, 0.7],
                "payload": {"land": "desert", "color": "red", "life": False, "humidity": 5}
            },
            {
                "id": 5,
                "vector": [0.5, 0.6, 0.7, 0.8],
                "payload": {"land": "marsh", "color": "black", "life": True, "humidity": 90}
            },
            {
                "id": 6,
                "vector": [0.6, 0.7, 0.8, 0.9],
                "payload": {"land": "cavern", "color": "black", "life": False, "humidity": 15}
            }
        ]
    }
    requests.put(points_url, json=points_data)
    
    # Create indexes
    index_url = f"{QDRANT_URL}/collections/terraforming/index"
    
    # Index life field
    requests.put(index_url, json={
        "field_name": "life",
        "field_schema": "bool"
    })
    
    # Index color field
    requests.put(index_url, json={
        "field_name": "color",
        "field_schema": "keyword"
    })
    
    # Index humidity field
    requests.put(index_url, json={
        "field_name": "humidity",
        "field_schema": {
            "type": "integer",
            "range": True
        }
    })
    return {"status": "terraforming collection created with indexes"}


@app.post("/terraforming/filter")
async def filter_terraforming(filter_data: dict):
    """Apply filters to terraforming collection"""
    scroll_url = f"{QDRANT_URL}/collections/terraforming/points/scroll"
    response = requests.post(scroll_url, json=filter_data)
    return response.json()


@app.get("/terraforming/setup")
async def setup_terraforming():
    """Setup the terraforming collection"""
    return create_terraforming_collection()


# Run the application when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"Starting FastAPI server on port {port}...", flush=True)
    print(f"Qdrant URL: {QDRANT_URL}", flush=True)
    print(f"Qdrant API Key configured: {bool(QDRANT_API_KEY)}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
