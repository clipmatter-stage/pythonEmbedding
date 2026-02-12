from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, MatchText, SearchParams, HnswConfigDiff, 
    OptimizersConfigDiff, QuantizationConfig, ScalarQuantization,
    ScalarType, QuantizationSearchParams
)
import uuid
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict
import re
import requests
from rapidfuzz import fuzz, process
from cachetools import TTLCache
from functools import lru_cache
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import time
from collections import defaultdict

# ============== LOGGING CONFIGURATION ==============
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============== ENVIRONMENT VARIABLES (SECURE) ==============
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
API_KEY = os.getenv("API_KEY")  # Optional API key for authentication

# Validate required environment variables
if not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("Missing required environment variables: QDRANT_URL and QDRANT_API_KEY")
    raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables are required")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Qdrant Search Configuration
QDRANT_SEARCH_PARAMS = SearchParams(
    hnsw_ef=128,  # Higher = more accurate but slower (default: 64)
    exact=False,  # Use HNSW index for speed (set True for 100% accuracy)
    quantization=QuantizationSearchParams(
        ignore=False,  # Use quantization for faster search
        rescore=True,  # Rescore top results with full vectors for accuracy
        oversampling=2.0  # Fetch 2x results before rescoring
    )
)

# Qdrant Collection Configuration
HNSW_CONFIG = HnswConfigDiff(
    m=16,  # Number of edges per node (higher = better recall, more memory)
    ef_construct=100,  # Build-time accuracy (higher = better index quality)
    full_scan_threshold=10000,  # Use full scan for small collections
    max_indexing_threads=0  # Auto-detect CPU cores
)

OPTIMIZERS_CONFIG = OptimizersConfigDiff(
    indexing_threshold=20000,  # Start indexing after 20k points
    memmap_threshold=50000,  # Use memory mapping for large collections
    max_optimization_threads=0  # Auto-detect CPU cores
)

# Quantization Config (reduces memory by ~4x, slight accuracy loss)
QUANTIZATION_CONFIG = ScalarQuantization(
    scalar=ScalarType.INT8,  # 8-bit quantization
    quantile=0.99,  # Use 99th percentile for quantization ranges
    always_ram=True  # Keep quantized vectors in RAM for speed
)

# ============== PYDANTIC MODELS FOR INPUT VALIDATION ==============
class EmbedVideoRequest(BaseModel):
    video_id: int = Field(..., gt=0, description="Video ID must be positive")
    identification_segments: List[dict] = Field(..., min_items=1)
    video_title: str = Field(default="", max_length=500)
    video_filename: str = Field(default="", max_length=500)
    youtube_url: str = Field(default="", max_length=1000)
    language: str = Field(default="", max_length=50)

class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=1000)
    words: List[str] = Field(default=[])
    word: Optional[str] = Field(default=None, max_length=200)
    top_k: int = Field(default=10, ge=1, le=100)
    video_id: Optional[int] = Field(default=None, gt=0)
    speaker: Optional[str] = Field(default=None, max_length=200)
    title: Optional[str] = Field(default=None, max_length=500)
    language: Optional[str] = Field(default=None, max_length=50)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    time_range: Optional[dict] = None
    max_scanned: int = Field(default=10000, ge=100, le=100000)
    exact_search: bool = Field(default=False, description="Use exact search (slower but 100% accurate)")
    hnsw_ef: Optional[int] = Field(default=None, ge=64, le=512, description="HNSW search accuracy (higher=slower+accurate)")
    min_term_coverage: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum % of query terms that must appear (0.5 = 50%)")
    strict_mode: bool = Field(default=False, description="Require ALL query terms to be present")

class SuggestRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    type: str = Field(default="both", pattern="^(speaker|title|both)$")
    limit: int = Field(default=10, ge=1, le=50)

class TitleSearchRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)

# ============== RATE LIMITER ==============
class RateLimiter:
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        # Clean old requests
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if now - req_time < self.window
        ]
        # Check limit
        if len(self.clients[client_ip]) >= self.requests:
            return False
        self.clients[client_ip].append(now)
        return True

rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)

# ============== API KEY AUTHENTICATION ==============
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    """Verify API key if configured, also check rate limits."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # API key verification (only if API_KEY is configured)
    if API_KEY:
        if not api_key or api_key != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )
    return True

# ============== APP INITIALIZATION ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting up Video Transcript Search API...")
    if QDRANT_URL:
        logger.info(f"Qdrant URL: {QDRANT_URL[:50]}...")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Video Transcript Elastic Search API",
    description="Secure API for semantic search over video transcripts",
    version="4.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for query embeddings (max 1000 queries, 1 hour TTL)
embedding_cache = TTLCache(maxsize=1000, ttl=3600)

logger.info("Loading sentence-transformers model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Model loaded successfully")

logger.info("Warming up model...")
_ = model.encode("warmup text", show_progress_bar=False)
logger.info("Model warmed up and ready")

# ============== ELASTIC SEARCH HELPERS ==============

def get_cached_embedding(text: str) -> List[float]:
    """Get embedding with caching for repeated queries."""
    cache_key = text.strip().lower()[:500]  # Normalize and limit key length
    if cache_key not in embedding_cache:
        embedding_cache[cache_key] = model.encode(text, show_progress_bar=False).tolist()
    return embedding_cache[cache_key]

def fuzzy_match_text(query: str, text: str, threshold: int = 75) -> bool:
    """
    Fuzzy match with typo tolerance using rapidfuzz.
    Returns True if query fuzzy-matches text above threshold.
    NOTE: Increased threshold from 65 to 75 to reduce false positives.
    """
    if not query or not text:
        return False
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Exact substring match (fastest)
    if query_lower in text_lower:
        return True
    
    # Fuzzy partial match (handles typos) - stricter threshold
    return fuzz.partial_ratio(query_lower, text_lower) >= threshold

def fuzzy_match_speaker(query: str, speaker: str, threshold: int = 80) -> bool:
    """
    Fuzzy match for speaker names with higher threshold.
    Handles variations like 'John' vs 'Jon', 'Muhammad' vs 'Mohammad'.
    NOTE: Increased threshold from 75 to 80 to reduce false positives.
    """
    if not query or not speaker:
        return False
    
    query_lower = query.lower().strip()
    speaker_lower = speaker.lower().strip()
    
    # Exact match
    if query_lower == speaker_lower or query_lower in speaker_lower:
        return True
    
    # Check each word in speaker name
    speaker_parts = speaker_lower.split()
    for part in speaker_parts:
        if fuzz.ratio(query_lower, part) >= threshold:
            return True
    
    # Full fuzzy match
    return fuzz.ratio(query_lower, speaker_lower) >= threshold

def generate_ngrams(text: str, n: int = 3) -> List[str]:
    """Generate character n-grams for partial matching."""
    text = text.lower().strip()
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def ngram_similarity(query: str, text: str, n: int = 3) -> float:
    """Calculate n-gram based similarity (0-1)."""
    query_ngrams = set(generate_ngrams(query, n))
    text_ngrams = set(generate_ngrams(text, n))
    
    if not query_ngrams or not text_ngrams:
        return 0.0
    
    intersection = query_ngrams & text_ngrams
    return len(intersection) / len(query_ngrams)

def expand_query_variations(query: str, max_variations: int = 5) -> List[str]:
    """
    Generate query variations for better recall.
    Handles common typos and variations.
    NOTE: Limited to prevent over-matching unrelated content.
    """
    variations = [query]
    query_lower = query.lower().strip()
    
    if query_lower not in [v.lower() for v in variations]:
        variations.append(query_lower)
    
    # Only add individual words if they are significant (>3 chars)
    words = [w for w in query.split() if len(w) > 3]
    if len(words) > 1:
        variations.extend(words[:max_variations - len(variations)])
    
    return list(set(variations))[:max_variations]

def calculate_query_term_coverage(query: str, text: str) -> float:
    """
    Calculate what percentage of query terms appear in the text.
    Returns value 0.0-1.0 indicating coverage.
    Prevents results that don't actually contain query terms.
    """
    if not query or not text:
        return 0.0
    
    query_terms = set(query.lower().split())
    text_lower = text.lower()
    
    # Filter out very short terms (stop words like 'a', 'is', 'the')
    significant_terms = {term for term in query_terms if len(term) > 2}
    
    if not significant_terms:
        return 1.0  # No significant terms to check
    
    matched_terms = sum(1 for term in significant_terms if term in text_lower)
    return matched_terms / len(significant_terms)

def calculate_combined_score(
    semantic_score: float = 0.0,
    keyword_match: bool = False, 
    fuzzy_score: float = 0.0,
    exact_match: bool = False,
    term_coverage: float = 0.0
) -> Dict[str, float]:
    """
    Calculate combined relevance score with transparency.
    Weights: semantic=0.45, keyword=0.25, fuzzy=0.12, exact_bonus=0.05, coverage=0.13
    Returns dict with breakdown for transparency.
    NOTE: Added term_coverage to ensure query terms are present.
    """
    score_breakdown = {
        "semantic": semantic_score * 0.45,
        "keyword": 0.25 if keyword_match else 0.0,
        "fuzzy": fuzzy_score * 0.12,
        "exact_bonus": 0.05 if exact_match else 0.0,
        "term_coverage": term_coverage * 0.13  # Penalize missing query terms
    }
    
    total_score = sum(score_breakdown.values())
    score_breakdown["total"] = min(total_score, 1.0)  # Cap at 1.0
    
    return score_breakdown

# ============== END ELASTIC SEARCH HELPERS ==============

SEGMENTS_COLLECTION = "video_transcript_segments"
LEGACY_COLLECTION = "text_embeddings"

logger.info("Connecting to Qdrant...")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
logger.info("Connected to Qdrant successfully")

def create_segments_collection():
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if SEGMENTS_COLLECTION not in collection_names:
            logger.info(f"Creating collection '{SEGMENTS_COLLECTION}' with optimized configuration...")
            qdrant_client.create_collection(
                collection_name=SEGMENTS_COLLECTION,
                vectors_config=VectorParams(
                    size=384, 
                    distance=Distance.COSINE,
                    hnsw_config=HNSW_CONFIG,
                    quantization_config=QUANTIZATION_CONFIG
                ),
                optimizers_config=OPTIMIZERS_CONFIG
            )
            logger.info("  ✓ HNSW index configured (m=16, ef_construct=100)")
            logger.info("  ✓ Quantization enabled (INT8, ~4x memory reduction)")
            logger.info("  ✓ Optimizers configured (threshold=20k, auto-threads)")
            # Create indexes for better performance
            qdrant_client. create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="video_id",
                field_schema=models.PayloadSchemaType. INTEGER
            )
            qdrant_client.create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="speaker",
                field_schema=models.PayloadSchemaType. KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="video_title",
                field_schema=models.PayloadSchemaType.TEXT
            )
            qdrant_client.create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="start_time",
                field_schema=models.PayloadSchemaType.FLOAT
            )
            logger.info(f"Collection '{SEGMENTS_COLLECTION}' created successfully")
        else:
            logger.info(f"Collection '{SEGMENTS_COLLECTION}' already exists")
    except Exception as e:
        logger.info(f"Error managing collection: {str(e)}")
        raise

def create_legacy_collection():
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if LEGACY_COLLECTION not in collection_names:
            logger.info(f"Creating legacy collection '{LEGACY_COLLECTION}'...")
            qdrant_client.create_collection(
                collection_name=LEGACY_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance. COSINE),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000)
            )
            logger.info(f"Collection '{LEGACY_COLLECTION}' created successfully")
    except Exception as e:
        logger.info(f"Error managing legacy collection:  {str(e)}")


def ensure_indexes_http():
    """Create required indexes using direct HTTP requests to avoid client version issues."""
    try: 
        logger.info("Creating indexes via HTTP...")
        
        indexes_to_create = [
            {"field_name": "video_id", "field_schema": "integer"},
            {"field_name": "speaker", "field_schema": "keyword"},
            {"field_name": "video_title", "field_schema": "text"},
            {"field_name": "language", "field_schema": "keyword"},
            {"field_name":  "start_time", "field_schema": "float"},
            {"field_name": "end_time", "field_schema": "float"},
        ]
        
        headers = {
            "api-key":  QDRANT_API_KEY,
            "Content-Type":  "application/json"
        }
        
        for index_config in indexes_to_create:
            field_name = index_config["field_name"]
            field_schema = index_config["field_schema"]
            
            url = f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}/index"
            
            payload = {
                "field_name": field_name,
                "field_schema": field_schema
            }
            
            try:
                response = requests.put(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"  ✓ Created index for '{field_name}'")
                elif response.status_code == 400 and "already exists" in response. text. lower():
                    logger.info(f"  ✓ Index for '{field_name}' already exists")
                else:
                    logger.info(f"  ✗ Failed to create index for '{field_name}':  {response.text}")
                    
            except Exception as e:
                logger.info(f"  ✗ Error creating index for '{field_name}': {str(e)}")
        
        logger.info("Index creation completed")
        
    except Exception as e:
        logger.info(f"Error in ensure_indexes_http: {str(e)}")


# Replace the old ensure_indexes call with this
create_segments_collection()
create_legacy_collection()
ensure_indexes_http()  # Use HTTP version instead


def parse_search_query(query:  str) -> Dict: 
    """
    Parse search query to extract filters and keywords.
    Examples:
      - "speaker: John machine learning" -> speaker filter + semantic query
      - "title:intro python" -> title filter + semantic query
      - "video:123 speaker:Jane AI" -> video_id + speaker filter + semantic query
    """
    parsed = {
        "video_id": None,
        "speaker":  None,
        "title": None,
        "keywords": [],
        "semantic_query": query
    }
    
    remaining_query = query
    
    # Extract video_id filter (video: 123)
    video_pattern = re.compile(r'video:(\d+)', re.IGNORECASE)
    video_match = video_pattern.search(remaining_query)
    if video_match: 
        parsed["video_id"] = int(video_match. group(1))
        remaining_query = video_pattern.sub('', remaining_query)
    
    # Extract speaker filter (speaker:John)
    speaker_pattern = re.compile(r'speaker: (\S+)', re.IGNORECASE)
    speaker_match = speaker_pattern.search(remaining_query)
    if speaker_match:
        parsed["speaker"] = speaker_match.group(1)
        remaining_query = speaker_pattern.sub('', remaining_query)
    
    # Extract title filter (title: something)
    title_pattern = re.compile(r'title: (\S+)', re.IGNORECASE)
    title_match = title_pattern. search(remaining_query)
    if title_match:
        parsed["title"] = title_match.group(1)
        remaining_query = title_pattern.sub('', remaining_query)
    
    # Clean up remaining query - remove extra spaces
    remaining_query = ' '.join(remaining_query.split()).strip()
    parsed["semantic_query"] = remaining_query
    
    # Extract keywords from remaining query
    if remaining_query: 
        parsed["keywords"] = remaining_query.split()
    
    return parsed
    
@app.post("/embed-video")
async def embed_video(data: EmbedVideoRequest, authorized: bool = Depends(verify_api_key)):
    """Embed video transcript segments. Requires API key if configured."""
    try:
        video_id = data.video_id
        identification_segments = data.identification_segments
        if not identification_segments:
            raise HTTPException(status_code=400, detail="identification_segments is required")
        
        video_title = data.video_title
        video_filename = data.video_filename
        youtube_url = data.youtube_url
        language = data.language
        
        logger.info(f"Processing video {video_id} with {len(identification_segments)} segments")
        
        delete_existing_embeddings(video_id)
        
        points = []
        segments_embedded = 0
        segments_without_text = 0
        texts_to_embed = []
        segment_metadata = []
        
        for idx, segment in enumerate(identification_segments):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            diarization_speaker = segment.get("diarizationSpeaker", "")
            match_type = segment.get("match", "")
            confidence = segment.get("confidence", 0)
            
            if not text or len(text. strip()) < 3:
                segments_without_text += 1
                continue
            
            texts_to_embed.append(text)
            segment_metadata.append({
                'idx': idx,
                'speaker':  speaker,
                'diarization_speaker': diarization_speaker,
                'match_type': match_type,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'text': text
            })
        
        if not texts_to_embed: 
            raise HTTPException(
                status_code=400,
                detail=f"No valid segments found to embed.  Total:  {len(identification_segments)}, Without text: {segments_without_text}"
            )
        
        logger.info(f"Generating embeddings for {len(texts_to_embed)} segments in batch.. .")
        batch_start_time = datetime.utcnow()
        vectors = model.encode(texts_to_embed, show_progress_bar=False, batch_size=32).tolist()
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        logger.info(f"Batch embedding completed in {batch_duration:.2f} seconds")
        
        for i, metadata in enumerate(segment_metadata):
            vector = vectors[i]
            id_string = f"video_{video_id}_seg_{metadata['idx']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))
            
            payload = {
                "video_id": video_id,
                "video_title": video_title,
                "video_filename": video_filename,
                "youtube_url": youtube_url,
                "language": language,
                "segment_index": metadata['idx'],
                "speaker": metadata['speaker'],
                "diarization_speaker": metadata['diarization_speaker'],
                "match_type":  metadata['match_type'],
                "start_time": metadata['start_time'],
                "end_time": metadata['end_time'],
                "duration": metadata['end_time'] - metadata['start_time'],
                "text": metadata['text'],
                "text_length": len(metadata['text']),
                "confidence": metadata['confidence'],
                "created_at": datetime.utcnow().isoformat()
            }
            
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            segments_embedded += 1
        
        if not points:
            raise HTTPException(
                status_code=400,
                detail=f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}"
            )
        
        logger.info(f"Inserting {len(points)} points into Qdrant...")
        
        try:
            qdrant_client.upsert(
                collection_name=SEGMENTS_COLLECTION,
                points=points,
                wait=True
            )
            logger.info(f"Successfully inserted {len(points)} points")
        except Exception as e:
            logger.info(f"ERROR:  Qdrant insertion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
        
        return {
            "success": True,
            "video_id": video_id,
            "collection":  SEGMENTS_COLLECTION,
            "segments_embedded": segments_embedded,
            "total_points_inserted": len(points),
            "embedding_time_seconds": round(batch_duration, 2),
            "message": f"Successfully embedded {segments_embedded} segments for video {video_id}"
        }
        
    except Exception as e:
        logger.info(f"Error in embed_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def delete_existing_embeddings(video_id: int):
    try:
        qdrant_client.delete(
            collection_name=SEGMENTS_COLLECTION,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=video_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted existing embeddings for video {video_id}")
    except Exception as e:
        logger.info(f"Note: Could not delete embeddings (may not exist): {str(e)}")

@app.post("/embed")
async def embed(data: dict):
    text = data.get("text", "")
    video_id = data.get("video_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    vector = model.encode(text).tolist()
    vector_id = f"video_{video_id}_{uuid.uuid4()}" if video_id else str(uuid.uuid4())

    metadata = {
        "text": text[: 5000],
        "text_length": len(text),
        "created_at": datetime.utcnow().isoformat(),
        "source": "legacy_embedding_api"
    }
    
    if video_id:
        metadata["video_id"] = video_id

    try:
        qdrant_client.upsert(
            collection_name=LEGACY_COLLECTION,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
      
    return {
        "id": vector_id,
        "embedding": vector,
        "vector_dimension": len(vector),
        "metadata": metadata,
        "status": "success"
    }

@app.post("/search")
async def search(data: SearchRequest, authorized: bool = Depends(verify_api_key)):
    """
    Enhanced search with elastic/fuzzy matching for speaker and title.
    Automatically searches for names in both transcript text AND speaker field.
    
    - query: Text to search (semantic + keyword search in text AND speaker field)
    - words: Additional keywords to find in transcript text
    - speaker: Filter by speaker name (fuzzy/partial match - case insensitive)
    - video_id: Filter by specific video (exact match)
    - title: Filter by video title (fuzzy/partial match - case insensitive)
    - language: Filter by language (exact match)
    
    Example: query="junaid" will search for "junaid" in both transcript text AND speaker names
    """
    query_text = data.query
    words = data.words
    word = data.word
    top_k = data.top_k
    video_id_filter = data.video_id
    speaker_filter = data.speaker
    title_filter = data.title
    language_filter = data.language
    min_score = data.min_score
    time_range = data.time_range
    max_scanned = data.max_scanned
    
    # Check if speaker_filter looks like a search query (contains spaces or is a name to search for)
    speaker_search_in_text = None
    if speaker_filter and " " in speaker_filter: 
        # This looks like a name to search IN the text, not a filter
        speaker_search_in_text = speaker_filter
        speaker_filter = None  # Don't use as exact filter
        # Add to semantic query
        if query_text:
            query_text = f"{query_text} {speaker_search_in_text}"
        else: 
            query_text = speaker_search_in_text
    
    # Handle backward compatibility for single word
    if word: 
        words = [word] if isinstance(word, str) else word
    
    # Parse query for embedded filters
    if query_text and not any([video_id_filter, speaker_filter, title_filter]):
        parsed = parse_search_query(query_text)
        if parsed["video_id"]:
            video_id_filter = parsed["video_id"]
        if parsed["speaker"]:
            speaker_filter = parsed["speaker"]
        if parsed["title"]:
            title_filter = parsed["title"]
        query_text = parsed["semantic_query"]
    
    if not query_text and not words and not title_filter:
        raise HTTPException(
            status_code=400,
            detail="Either 'query', 'words', or 'title' is required"
        )

    # Build Qdrant filter - only use exact filters (video_id, language)
    # Speaker and title will be filtered client-side for fuzzy matching
    filter_conditions = []
    
    if video_id_filter is not None:
        filter_conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=video_id_filter))
        )
    
    # Remove speaker from Qdrant filter - do fuzzy matching client-side instead
    
    if language_filter is not None:
        filter_conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language_filter))
        )
    
    if time_range:
        start_time = time_range.get("start")
        end_time = time_range. get("end")
        if start_time is not None: 
            filter_conditions.append(
                FieldCondition(key="start_time", range=models.Range(gte=start_time))
            )
        if end_time is not None: 
            filter_conditions. append(
                FieldCondition(key="end_time", range=models.Range(lte=end_time))
            )

    search_filter = Filter(must=filter_conditions) if filter_conditions else None

    semantic_results = []
    keyword_results = []

    # Strategy 1: Semantic search with caching
    if query_text:
        try:
            logger.info(f"Elastic semantic search for: '{query_text[:120]}'")
            
            # Use cached embedding for repeated queries
            query_vector = get_cached_embedding(query_text)
            
            # Build custom search params if user overrides
            search_params = QDRANT_SEARCH_PARAMS
            if data.exact_search or data.hnsw_ef:
                search_params = SearchParams(
                    hnsw_ef=data.hnsw_ef if data.hnsw_ef else 128,
                    exact=data.exact_search,
                    quantization=QuantizationSearchParams(
                        ignore=data.exact_search,  # Disable quantization for exact search
                        rescore=True,
                        oversampling=2.0
                    ) if not data.exact_search else None
                )
                logger.info(f"  Custom search params: exact={data.exact_search}, hnsw_ef={data.hnsw_ef or 128}")
            
            # Track search performance
            search_start = time.time()

            # Qdrant search with optimized parameters
            sem_search_results = qdrant_client.search(
                collection_name=SEGMENTS_COLLECTION,
                query_vector=query_vector,
                limit=top_k * 4,  # Get more results for fuzzy filtering
                score_threshold=min_score * 0.8,  # Lower threshold, filter later
                query_filter=search_filter,
                search_params=search_params,  # Use optimized HNSW + quantization
                with_payload=True,
                with_vectors=False
            )
            
            search_time = time.time() - search_start
            logger.info(f"  ✓ Qdrant search: {len(sem_search_results)} results in {search_time:.3f}s (hnsw_ef={search_params.hnsw_ef}, exact={search_params.exact})")

            for r in sem_search_results:
                payload = r.payload or {}
                video_title = payload.get("video_title", "")
                speaker = payload.get("speaker", "")
                text = payload.get("text", "")
                
                # ELASTIC title filter with fuzzy matching
                if title_filter:
                    if not fuzzy_match_text(title_filter, video_title, threshold=60):
                        continue
                
                # ELASTIC speaker filter with fuzzy matching (handles typos)
                if speaker_filter:
                    if not fuzzy_match_speaker(speaker_filter, speaker, threshold=70):
                        continue
                
                # ELASTIC speaker search in text with fuzzy matching
                if speaker_search_in_text:
                    found_speaker = (
                        fuzzy_match_text(speaker_search_in_text, text, threshold=65) or
                        fuzzy_match_speaker(speaker_search_in_text, speaker, threshold=70)
                    )
                    if not found_speaker:
                        continue
                
                # Calculate comprehensive scoring
                semantic_score = float(getattr(r, "score", 0.0))
                fuzzy_boost = 0
                exact_match = False
                
                # Check exact word matches in text
                text_lower = text.lower()
                if query_text and query_text.lower() in text_lower:
                    exact_match = True
                
                # Calculate query term coverage
                term_coverage = calculate_query_term_coverage(query_text, text)
                
                # Apply strict mode: reject if not enough terms present
                if data.strict_mode and term_coverage < 1.0:
                    continue  # Skip this result in strict mode
                
                if term_coverage < data.min_term_coverage:
                    continue  # Skip if below minimum coverage threshold
                
                # Calculate fuzzy scores for filters
                if title_filter and video_title:
                    fuzzy_boost = max(fuzzy_boost, fuzz.partial_ratio(title_filter.lower(), video_title.lower()) / 100)
                if speaker_filter and speaker:
                    fuzzy_boost = max(fuzzy_boost, fuzz.ratio(speaker_filter.lower(), speaker.lower()) / 100)
                
                # Calculate combined score with breakdown
                score_breakdown = calculate_combined_score(
                    semantic_score=semantic_score,
                    keyword_match=False,
                    fuzzy_score=fuzzy_boost,
                    exact_match=exact_match,
                    term_coverage=term_coverage
                )
                
                semantic_results.append({
                    "id": r.id,
                    "score": score_breakdown["total"],
                    "score_breakdown": score_breakdown,
                    "video_id": payload.get("video_id"),
                    "video_title": video_title,
                    "speaker": speaker,
                    "diarization_speaker": payload.get("diarization_speaker", ""),
                    "start_time": payload.get("start_time", 0),
                    "end_time": payload.get("end_time", 0),
                    "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                    "text": text,
                    "text_length": payload.get("text_length", 0),
                    "youtube_url": payload.get("youtube_url", ""),
                    "language": payload.get("language", ""),
                    "created_at": payload.get("created_at"),
                    "match_types": ["semantic"],
                    "fuzzy_score": round(fuzzy_boost, 2)
                })

        except Exception as e:
            logger.info(f"ERROR during semantic search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")

    # Strategy 2: ELASTIC Keyword search with fuzzy matching
    search_words = list(words) if words else []
    
    # Add query words to keyword search for elastic name matching
    if query_text:
        query_words = query_text.strip().split()
        search_words.extend(query_words)
    
    if speaker_search_in_text:
        # Add speaker name words to keyword search
        search_words.extend(speaker_search_in_text.split())
    
    # Expand search words with variations for better recall
    expanded_words = []
    for word in search_words:
        expanded_words.extend(expand_query_variations(word))
    search_words = list(set(expanded_words))
    
    if search_words:
        try:
            logger.info(f"Elastic keyword search for: {search_words[:10]} (max_scanned={max_scanned})")
            words_lower = [w.lower() for w in search_words]
            
            # Optimize page size based on collection size
            page_size = min(1000, max_scanned // 10)  # Adaptive batch size
            scanned = 0
            offset = None

            scroll_filter = search_filter

            while scanned < max_scanned and len(keyword_results) < top_k * 2:
                points, next_offset = qdrant_client.scroll(
                    collection_name=SEGMENTS_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=min(page_size, max_scanned - scanned),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not points:
                    break

                for p in points:
                    scanned += 1
                    payload = p.payload or {}
                    text = (payload.get("text") or "").lower()
                    speaker_field = (payload.get("speaker") or "")
                    video_title = payload.get("video_title", "")
                    combined_text = f"{text} {speaker_field.lower()}"
                    
                    # ELASTIC: Check exact match OR fuzzy match for each word
                    word_matched = False
                    fuzzy_score = 0
                    exact_match = False
                    matched_words = []
                    
                    for w in words_lower:
                        # Skip very short words that cause false matches
                        if len(w) <= 2:
                            continue
                        
                        # Exact substring match
                        if w in combined_text:
                            word_matched = True
                            exact_match = True
                            fuzzy_score = 1.0
                            matched_words.append(w)
                        # Fuzzy match with typo tolerance - stricter threshold
                        elif len(w) >= 4 and fuzzy_match_text(w, combined_text, threshold=75):
                            word_matched = True
                            current_fuzzy = fuzz.partial_ratio(w, combined_text) / 100
                            # Only accept if fuzzy score is reasonably high
                            if current_fuzzy >= 0.75:
                                fuzzy_score = max(fuzzy_score, current_fuzzy)
                                matched_words.append(w)
                    
                    if not word_matched:
                        continue
                    
                    # Calculate term coverage for keyword results
                    full_text = payload.get("text", "")
                    term_coverage = calculate_query_term_coverage(" ".join(search_words), full_text)
                    
                    # Skip if term coverage is too low
                    if term_coverage < data.min_term_coverage:
                        continue
                    
                    # ELASTIC title filter with fuzzy matching
                    if title_filter:
                        if not fuzzy_match_text(title_filter, video_title, threshold=60):
                            continue
                    
                    # ELASTIC speaker filter with fuzzy matching
                    if speaker_filter:
                        if not fuzzy_match_speaker(speaker_filter, speaker_field, threshold=70):
                            continue
                    
                    # Calculate proper combined score for keyword results
                    score_breakdown = calculate_combined_score(
                        semantic_score=0.0,  # No semantic search for keyword-only
                        keyword_match=True,
                        fuzzy_score=fuzzy_score,
                        exact_match=exact_match,
                        term_coverage=term_coverage
                    )
                    
                    keyword_results.append({
                        "id": p.id,
                        "score": score_breakdown["total"],
                        "score_breakdown": score_breakdown,
                        "video_id": payload.get("video_id"),
                        "video_title": video_title,
                        "speaker": speaker_field,
                        "diarization_speaker": payload.get("diarization_speaker", ""),
                        "start_time": payload.get("start_time", 0),
                        "end_time": payload.get("end_time", 0),
                        "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                        "text": payload.get("text", ""),
                        "text_length": payload.get("text_length", 0),
                        "youtube_url": payload.get("youtube_url", ""),
                        "language": payload.get("language", ""),
                        "created_at": payload.get("created_at"),
                        "match_types": ["keyword"],
                        "matched_words": matched_words,
                        "fuzzy_score": round(fuzzy_score, 2)
                    })

                if len(keyword_results) >= top_k * 2:
                    break

                offset = next_offset
                if not next_offset:
                    break

        except Exception as e:
            logger.info(f"ERROR during keyword search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Keyword search error: {str(e)}")

    # Merge results with intelligent score combination
    merged = {}
    
    for r in semantic_results: 
        merged[r["id"]] = r

    for r in keyword_results:
        if r["id"] in merged:
            # Result appears in both semantic and keyword - combine scores intelligently
            existing = merged[r["id"]]
            
            # Add keyword to match types
            if "keyword" not in existing["match_types"]: 
                existing["match_types"].append("keyword")
            
            # Combine score breakdowns for hybrid results
            existing_breakdown = existing.get("score_breakdown", {})
            keyword_breakdown = r.get("score_breakdown", {})
            
            combined_breakdown = {
                "semantic": existing_breakdown.get("semantic", 0),
                "keyword": max(existing_breakdown.get("keyword", 0), keyword_breakdown.get("keyword", 0)),
                "fuzzy": max(existing_breakdown.get("fuzzy", 0), keyword_breakdown.get("fuzzy", 0)),
                "exact_bonus": max(existing_breakdown.get("exact_bonus", 0), keyword_breakdown.get("exact_bonus", 0)),
                "term_coverage": max(existing_breakdown.get("term_coverage", 0), keyword_breakdown.get("term_coverage", 0))
            }
            combined_breakdown["total"] = min(sum(combined_breakdown.values()), 1.0)
            
            # Update with combined score
            existing["score"] = combined_breakdown["total"]
            existing["score_breakdown"] = combined_breakdown
            existing["match_types"] = sorted(existing["match_types"])
            
            # Merge matched words if present
            if "matched_words" in r:
                existing["matched_words"] = r["matched_words"]
        else:
            merged[r["id"]] = r

    merged_list = sorted(
        merged. values(),
        key=lambda x: (x. get("score", 0), -x.get("start_time", 0)),
        reverse=True
    )[:top_k]

    return {
        "query": query_text,
        "words": words,
        "speaker_searched_in_text": speaker_search_in_text,
        "collection": SEGMENTS_COLLECTION,
        "total_semantic_hits": len(semantic_results),
        "total_keyword_hits": len(keyword_results),
        "returned": len(merged_list),
        "search_params": {
            "hnsw_ef": data.hnsw_ef or QDRANT_SEARCH_PARAMS.hnsw_ef,
            "exact_search": data.exact_search,
            "quantization_enabled": not data.exact_search
        },
        "filters_applied": {
            "video_id": video_id_filter,
            "speaker": speaker_filter,
            "title": title_filter,
            "language": language_filter,
            "time_range": time_range,
            "min_score": min_score,
            "min_term_coverage": data.min_term_coverage,
            "strict_mode": data.strict_mode
        },
        "results": [
            {
                "id": r["id"],
                "score": round(r["score"], 4),
                "score_breakdown": {
                    "semantic": round(r.get("score_breakdown", {}).get("semantic", 0), 3),
                    "keyword": round(r.get("score_breakdown", {}).get("keyword", 0), 3),
                    "fuzzy": round(r.get("score_breakdown", {}).get("fuzzy", 0), 3),
                    "exact_bonus": round(r.get("score_breakdown", {}).get("exact_bonus", 0), 3),
                    "term_coverage": round(r.get("score_breakdown", {}).get("term_coverage", 0), 3)
                },
                "match_types": r.get("match_types", []),
                "matched_words": r.get("matched_words", []),
                "video_id": r.get("video_id"),
                "video_title": r.get("video_title", ""),
                "speaker": r.get("speaker", ""),
                "diarization_speaker": r.get("diarization_speaker", ""),
                "start_time": r.get("start_time", 0),
                "end_time": r.get("end_time", 0),
                "duration": r.get("duration", 0),
                "text": r.get("text", ""),
                "text_length": r.get("text_length", 0),
                "youtube_url": r.get("youtube_url", ""),
                "language": r.get("language", ""),
                "created_at": r.get("created_at"),
                "youtube_url_timestamped": f"{r.get('youtube_url', '')}?t={int(r.get('start_time', 0))}" if r.get('youtube_url') else ""
            }
            for r in merged_list
        ]
    }
    
@app.post("/search-by-title")
async def search_by_title(data: TitleSearchRequest, authorized: bool = Depends(verify_api_key)):
    """
    Search for videos by title (returns unique videos, not segments).
    
    Params:
      - title: partial or full video title
      - limit: max number of videos to return
    """
    title = data.title
    limit = data.limit
    
    try:
        logger.info(f"Elastic title search for: '{title}'")
        
        # Scroll through collection to find matching titles with FUZZY matching
        videos = {}
        offset = None
        scanned = 0
        max_scan = 50000
        
        while scanned < max_scan and len(videos) < limit * 2:
            points, next_offset = qdrant_client.scroll(
                collection_name=SEGMENTS_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
            
            for p in points:
                scanned += 1
                payload = p.payload or {}
                video_title = payload.get("video_title", "")
                video_id = payload.get("video_id")
                
                # ELASTIC: Fuzzy title matching with typo tolerance
                if video_id not in videos and fuzzy_match_text(title, video_title, threshold=60):
                    match_score = fuzz.partial_ratio(title.lower(), video_title.lower())
                    videos[video_id] = {
                        "video_id": video_id,
                        "video_title": video_title,
                        "youtube_url": payload.get("youtube_url", ""),
                        "language": payload.get("language", ""),
                        "match_score": match_score
                    }
                
                if len(videos) >= limit:
                    break
            
            offset = next_offset
            if not next_offset:
                break
        
        # Sort by match score
        sorted_videos = sorted(videos.values(), key=lambda x: x.get("match_score", 0), reverse=True)
        
        return {
            "title_query": title,
            "total_videos_found": len(videos),
            "scanned_segments": scanned,
            "videos": sorted_videos[:limit]
        }
        
    except Exception as e:
        logger.info(f"ERROR in search_by_title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-multi-video")
async def search_multi_video(data: dict):
    query_text = data.get("query", "")
    video_ids = data.get("video_ids", [])
    top_k = data.get("top_k", 5)
    min_score = data.get("min_score", 0.5)
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    if not video_ids:
        raise HTTPException(status_code=400, detail="video_ids list is required")
    
    logger.info(f"Searching across {len(video_ids)} videos for: '{query_text[: 100]}'")
    
    query_vector = model.encode(query_text).tolist()
    
    try:
        search_filter = Filter(
            should=[
                FieldCondition(key="video_id", match=MatchValue(value=vid))
                for vid in video_ids
            ]
        )
        
        search_results = qdrant_client.search(
            collection_name=SEGMENTS_COLLECTION,
            query_vector=query_vector,
            limit=top_k * len(video_ids),
            score_threshold=min_score,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False
        )
        
        results_by_video = {}
        for r in search_results:
            vid = r.payload.get("video_id")
            if vid not in results_by_video: 
                results_by_video[vid] = []
            
            if len(results_by_video[vid]) < top_k:
                results_by_video[vid].append({
                    "id": r. id,
                    "score":  round(r.score, 4),
                    "similarity_percentage": round(r.score * 100, 2),
                    "video_title": r.payload.get("video_title", ""),
                    "speaker": r.payload.get("speaker", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time":  r.payload.get("end_time", 0),
                    "text": r.payload.get("text", ""),
                    "youtube_url_timestamped": f"{r. payload.get('youtube_url', '')}?t={int(r.payload.get('start_time', 0))}" if r.payload.get('youtube_url') else ""
                })
        
        return {
            "query": query_text,
            "total_videos_searched": len(video_ids),
            "videos_with_results": len(results_by_video),
            "results_by_video": results_by_video
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}/segments")
async def get_video_segments(video_id: int, limit: int = 100, offset: int = 0, authorized: bool = Depends(verify_api_key)):
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=SEGMENTS_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id))
                ]
            ),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = scroll_result
        
        return {
            "video_id": video_id,
            "total_segments": len(points),
            "offset": offset,
            "limit":  limit,
            "next_offset": next_offset,
            "segments": [
                {
                    "segment_index": (p.payload or {}).get("segment_index"),
                    "speaker": (p.payload or {}).get("speaker"),
                    "start_time": (p.payload or {}).get("start_time"),
                    "end_time": (p.payload or {}).get("end_time"),
                    "duration": (p.payload or {}).get("duration"),
                    "text": (p.payload or {}).get("text"),
                    "text_length": (p.payload or {}).get("text_length"),
                }
                for p in points
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest")
async def suggest(data: SuggestRequest, authorized: bool = Depends(verify_api_key)):
    """
    ELASTIC autocomplete/suggestion endpoint.
    Returns fuzzy-matched suggestions for speakers and titles.
    
    Params:
      - query: partial text to autocomplete
      - type: "speaker" or "title" (default: both)
      - limit: max suggestions (default: 10)
    """
    query = data.query.strip()
    suggest_type = data.type
    limit = data.limit
    
    try:
        logger.info(f"Generating suggestions for: '{query}' (type={suggest_type})")
        
        speakers = {}
        titles = {}
        offset = None
        scanned = 0
        max_scan = 20000
        
        while scanned < max_scan and (len(speakers) < limit or len(titles) < limit):
            points, next_offset = qdrant_client.scroll(
                collection_name=SEGMENTS_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=["speaker", "video_title", "video_id"],
                with_vectors=False
            )
            
            if not points:
                break
            
            for p in points:
                scanned += 1
                payload = p.payload or {}
                
                # Collect speaker suggestions
                if suggest_type in ["speaker", "both"]:
                    speaker = payload.get("speaker", "")
                    if speaker and speaker not in speakers:
                        if fuzzy_match_speaker(query, speaker, threshold=60):
                            score = fuzz.ratio(query.lower(), speaker.lower())
                            speakers[speaker] = score
                
                # Collect title suggestions
                if suggest_type in ["title", "both"]:
                    title = payload.get("video_title", "")
                    video_id = payload.get("video_id")
                    if title and video_id not in titles:
                        if fuzzy_match_text(query, title, threshold=50):
                            score = fuzz.partial_ratio(query.lower(), title.lower())
                            titles[video_id] = {"title": title, "video_id": video_id, "score": score}
            
            offset = next_offset
            if not next_offset:
                break
        
        # Sort by score and limit results
        sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:limit]
        sorted_titles = sorted(titles.values(), key=lambda x: x["score"], reverse=True)[:limit]
        
        return {
            "query": query,
            "type": suggest_type,
            "speakers": [{"name": s[0], "score": s[1]} for s in sorted_speakers] if suggest_type in ["speaker", "both"] else [],
            "titles": [{"video_id": t["video_id"], "title": t["title"], "score": t["score"]} for t in sorted_titles] if suggest_type in ["title", "both"] else [],
            "scanned": scanned
        }
        
    except Exception as e:
        logger.info(f"ERROR in suggest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/video/{video_id}/embeddings")
async def delete_video_embeddings(video_id: int, authorized: bool = Depends(verify_api_key)):
    """Delete all embeddings for a video. Requires API key."""
    try:
        delete_existing_embeddings(video_id)
        logger.info(f"Deleted embeddings for video {video_id}")
        return {
            "success": True,
            "message": f"Deleted all embeddings for video {video_id}"
        }
    except Exception as e:
        logger.error(f"Error deleting embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        
        # Get HNSW and quantization status
        hnsw_config = collection_info.config.hnsw_config if hasattr(collection_info.config, 'hnsw_config') else None
        quantization_enabled = hasattr(collection_info.config, 'quantization_config') and collection_info.config.quantization_config is not None
        
        health_response = {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": {
                "segments": SEGMENTS_COLLECTION,
                "legacy": LEGACY_COLLECTION
            },
            "points_count": collection_info.points_count,
            "vectors_count": getattr(collection_info, 'vectors_count', 0)
        }
        
        # Add HNSW config if available
        if hnsw_config:
            health_response["qdrant_config"] = {
                "hnsw": {
                    "m": getattr(hnsw_config, 'm', 16),
                    "ef_construct": getattr(hnsw_config, 'ef_construct', 100),
                    "full_scan_threshold": getattr(hnsw_config, 'full_scan_threshold', 10000)
                },
                "quantization_enabled": quantization_enabled,
                "optimizer_status": str(getattr(collection_info, 'optimizer_status', 'unknown'))
            }
        
        return health_response
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant_connected": False,
            "error": str(e)
        }

@app.get("/stats")
async def stats():
    try:
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        hnsw_config = getattr(collection_info.config, 'hnsw_config', None)
        quantization_config = getattr(collection_info.config, 'quantization_config', None)
        
        # Safely get vector params
        vectors_param = collection_info.config.params.vectors
        if isinstance(vectors_param, dict):
            # Named vectors
            first_vector = next(iter(vectors_param.values()), None)
            vector_size = getattr(first_vector, 'size', 384) if first_vector else 384
            distance = getattr(getattr(first_vector, 'distance', None), 'name', 'COSINE') if first_vector else 'COSINE'
        else:
            # Single vector
            vector_size = getattr(vectors_param, 'size', 384)
            distance = getattr(getattr(vectors_param, 'distance', None), 'name', 'COSINE')
        
        stats_response = {
            "collection": SEGMENTS_COLLECTION,
            "points_count": collection_info.points_count,
            "vectors_count": getattr(collection_info, 'vectors_count', 0),
            "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', 0),
            "status": str(collection_info.status),
            "optimizer_status": str(getattr(collection_info, 'optimizer_status', 'unknown')),
            "config": {
                "vector_size": vector_size,
                "distance": distance
            },
            "search_params": {
                "default_hnsw_ef": QDRANT_SEARCH_PARAMS.hnsw_ef,
                "default_exact": QDRANT_SEARCH_PARAMS.exact,
                "quantization_rescore": True,
                "cache_enabled": True,
                "cache_size": len(embedding_cache),
                "cache_max_size": embedding_cache.maxsize
            }
        }
        
        # Add HNSW config if available
        if hnsw_config:
            stats_response["config"]["hnsw"] = {
                "m": getattr(hnsw_config, 'm', 16),
                "ef_construct": getattr(hnsw_config, 'ef_construct', 100),
                "full_scan_threshold": getattr(hnsw_config, 'full_scan_threshold', 10000),
                "max_indexing_threads": getattr(hnsw_config, 'max_indexing_threads', 0)
            }
        
        # Add quantization config if available
        stats_response["config"]["quantization"] = {
            "enabled": quantization_config is not None,
            "type": "INT8" if quantization_config else None,
            "memory_reduction": "~75%" if quantization_config else "0%"
        }
        
        return stats_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "Video Transcript Elastic Search API",
        "version": "4.0",
        "features": "Fuzzy/Typo-tolerant search with rapidfuzz",
        "endpoints": {
            "POST /embed-video": "Embed entire video transcript",
            "POST /search": "ELASTIC semantic + keyword + fuzzy search (with Qdrant optimizations)",
            "POST /search-by-title": "Fuzzy search videos by title",
            "POST /search-multi-video": "Search across multiple specific videos",
            "POST /suggest": "Autocomplete suggestions for speakers/titles",
            "GET /video/{video_id}/segments": "Get all segments for a video",
            "DELETE /video/{video_id}/embeddings": "Delete all embeddings for a video",
            "GET /health": "Health check with Qdrant config",
            "GET /stats": "Collection statistics + HNSW/quantization info"
        },
        "qdrant_optimizations": [
            "HNSW indexing (m=16, ef_construct=100) for fast approximate search",
            "INT8 scalar quantization (~75% memory reduction)",
            "Quantization rescoring for accuracy recovery",
            "Configurable search accuracy (hnsw_ef parameter)",
            "Optional exact search mode (exact=true)",
            "Auto-threaded indexing and optimization",
            "Memory-mapped storage for large collections (>50k points)"
        ],
        "elastic_search_features": [
            "Semantic search using embeddings (cached for performance)",
            "Fuzzy/typo-tolerant matching (stricter thresholds to reduce false positives)",
            "Query term coverage validation (ensures query terms are actually present)",
            "Strict mode option (requires ALL query terms to be present)",
            "Minimum term coverage threshold (default 50% of query terms)",
            "N-gram based partial matching",
            "Speaker name autocomplete with fuzzy matching",
            "Title search with typo tolerance",
            "Combined scoring: semantic + keyword + fuzzy + coverage",
            "Query caching for repeated searches",
            "Multi-keyword search with OR logic",
            "Time range filtering",
            "Query parsing (e.g., 'speaker:John title:intro AI')",
            "Short word filtering (prevents 'a', 'is', 'the' false matches)"
        ],
        "example_queries": {
            "semantic": {"query": "machine learning algorithms"},
            "strict_mode": {"query": "deep neural networks", "strict_mode": True},
            "relaxed_mode": {"query": "AI concepts", "min_term_coverage": 0.3},
            "fast_search": {"query": "AI", "hnsw_ef": 64},
            "accurate_search": {"query": "AI", "hnsw_ef": 256},
            "exact_search": {"query": "AI", "exact_search": True},
            "fuzzy_speaker": {"query": "AI", "speaker": "Muhamad"},
            "fuzzy_title": {"query": "python", "title": "introducion"},
            "keyword_fuzzy": {"words": ["nueral", "netwerk"], "query": "AI"},
            "autocomplete": {"endpoint": "/suggest", "body": {"query": "joh", "type": "speaker"}},
            "parsed": {"query": "speaker:John title:intro machine learning"},
            "multi_video": {"query": "AI", "video_ids": [1, 2, 3]}
        },
        "performance_tips": [
            "Use hnsw_ef=64 for fast searches (~2x faster)",
            "Use hnsw_ef=256 for high accuracy (~2x slower)",
            "Use exact_search=true only when 100% accuracy required",
            "Quantization saves ~75% memory with <3% accuracy loss",
            "Query cache handles repeated searches instantly",
            "Collection auto-optimizes after 20k points",
            "Use strict_mode=true to eliminate all irrelevant results",
            "Adjust min_term_coverage (0.3-1.0) to control precision vs recall",
            "Default min_term_coverage=0.5 prevents most false positives"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    logger.info(f"Starting FastAPI Video Elastic Search Service on port {port}...")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Collections: {SEGMENTS_COLLECTION}, {LEGACY_COLLECTION}")
    logger.info("="*60)
    logger.info("Qdrant Optimizations:")
    logger.info(f"  ✓ HNSW Index: m={HNSW_CONFIG.m}, ef_construct={HNSW_CONFIG.ef_construct}")
    logger.info(f"  ✓ Search Params: hnsw_ef={QDRANT_SEARCH_PARAMS.hnsw_ef}, exact={QDRANT_SEARCH_PARAMS.exact}")
    logger.info(f"  ✓ Quantization: INT8 (saves ~75% memory)")
    logger.info(f"  ✓ Query Cache: {embedding_cache.maxsize} queries, {embedding_cache.ttl}s TTL")
    logger.info("="*60)
    logger.info("Features: Fuzzy search, typo tolerance, HNSW+quantization, query caching")
    uvicorn.run(app, host="0.0.0.0", port=port)
