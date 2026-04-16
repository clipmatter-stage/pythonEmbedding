from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from qdrant_client import models, QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchText, SearchParams, ScoredPoint
import uuid
import os
import logging
import string as _string
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import re
import requests
from rapidfuzz import fuzz, process
from cachetools import TTLCache
from functools import lru_cache
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import time
from collections import defaultdict
import openai
from openai import OpenAI
import tiktoken
import asyncio
import json as json_module

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key for advanced embeddings

# Validate required environment variables
if not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("Missing required environment variables: QDRANT_URL and QDRANT_API_KEY")
    raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables are required")

# Configure embedding strategy
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")  # Using large model for best quality
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))  # 3072 dimensions for text-embedding-3-large

# ============== STOP WORDS FOR KEYWORD SEARCH ==============
# Common words that should be filtered out to speed up keyword search
# These words match too many documents and slow down the search significantly
STOP_WORDS = frozenset({
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "no", "every",
    # Pronouns
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    "who", "whom", "whose", "which", "what", "whoever", "whatever",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down", "into",
    "out", "over", "under", "off", "about", "through", "during", "before", "after",
    "above", "below", "between", "among", "against", "toward", "towards",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet", "both", "either", "neither", "not", "only",
    "if", "then", "else", "when", "where", "while", "although", "though", "because", "unless",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "get", "got", "getting", "go", "goes", "going", "went", "gone",
    "say", "says", "said", "saying", "make", "makes", "made", "making",
    "know", "knows", "knew", "known", "knowing", "think", "thinks", "thought", "thinking",
    "see", "sees", "saw", "seen", "seeing", "want", "wants", "wanted", "wanting",
    "come", "comes", "came", "coming", "take", "takes", "took", "taking", "taken",
    "give", "gives", "gave", "giving", "given", "use", "uses", "used", "using",
    "find", "finds", "found", "finding", "tell", "tells", "told", "telling",
    "put", "puts", "putting", "let", "lets", "letting", "keep", "keeps", "kept", "keeping",
    "begin", "begins", "began", "beginning", "begun", "seem", "seems", "seemed", "seeming",
    "help", "helps", "helped", "helping", "show", "shows", "showed", "showing", "shown",
    "try", "tries", "tried", "trying", "ask", "asks", "asked", "asking",
    "work", "works", "worked", "working", "need", "needs", "needed", "needing",
    "feel", "feels", "felt", "feeling", "become", "becomes", "became", "becoming",
    "leave", "leaves", "left", "leaving", "call", "calls", "called", "calling",
    # Adverbs
    "very", "really", "just", "also", "too", "even", "still", "already", "always", "never",
    "often", "sometimes", "now", "then", "here", "there", "where", "when", "why", "how",
    "more", "most", "less", "least", "well", "much", "many", "few", "little", "big", 
    # Other common words
    "one", "two", "first", "last", "new", "old", "good", "great", "high", "low",
    "small", "large", "long", "short", "right", "wrong", "same", "different",
    "other", "another", "such", "own", "back", "away", "around", "again",
    "something", "anything", "everything", "nothing", "someone", "anyone", "everyone", "nobody",
    "thing", "things", "way", "ways", "day", "days", "time", "times", "year", "years",
    "people", "person", "man", "woman", "men", "women", "child", "children",
    "part", "parts", "place", "places", "case", "cases", "week", "weeks", "point", "points",
    "fact", "facts", "hand", "hands", "side", "sides", "world", "life", "being",
    # Urdu common words (transliterated)
    "hai", "hain", "tha", "thi", "the", "ka", "ki", "ke", "ko", "ne", "se", "mein", "par",
    "aur", "ya", "lekin", "jo", "jab", "kya", "kaise", "kahan", "yeh", "woh", "koi", "kuch",
})

# ============== PERSON NAME ALIASES ==============
# Maps known aliases/abbreviations to canonical names with all their variants.
# When any alias is detected in a query or speaker filter, the search automatically
# expands to cover all forms of that person's name.
# Also includes common misspellings, typos, and concatenated forms users may type.
PERSON_ALIASES = {
    "hafiz_naeem": {
        "canonical": "Hafiz Naeem Ur Rehman",
        # All text forms that should trigger recognition of this person
        "aliases": [
            # === Canonical and standard forms ===
            "hafiz naeem ur rehman", "hafiz naeem", "hafiz naeem rehman",
            "naeem ur rehman", "naeem rehman", "h naeem", "naeem",
            "rehman", "hafiz", "hnr", "h.n.r", "h.n.r.",
            "hafiz naeem-ur-rehman", "hafiz naeem ur rahman",
            "hafiz naeemur rehman", "hafiz naeemurrehman",
            
            # === Common concatenated/no-space forms (users type fast) ===
            "hafiznaeem", "naeemrehman", "naeemurrehman", "hafiznaeemurehman",
            "hafiz naeemrehman", "hafiz naeemurrehman",
            "hafiznaeem ur rehman", "hafiznaeem rehman",
            "naeemur rehman", "naeem-rehman", "naeem-ur-rehman",
            "hafiz-naeem", "hafiz-naeem-ur-rehman",
            
            # === Common misspellings & typos ===
            "hafiz naem", "hafiz naem ur rehman", "hafiz naim", "hafiz naim ur rehman",
            "hafiz naaem", "hafiz naeem ur rahman", "hafiz naeem ur rehmaan",
            "hafiz naeem ur rehmann", "hafiz naeeem", "hafiz naeem ur rahmaan",
            "hafiz naeem ur reham", "hafiz naeem rehmann", "hafiz naeem reham",
            "hafiz naeem urehman", "hafiz naeem urehmann",
            "hfiz naeem", "hafz naeem", "hafix naeem", "hafis naeem",
            "hafiz naeem ur reman", "hafiz naeem ur reman",
            "hafiz naeemeurahman", "hafiz naeemeurehman",
            "hafiz naeemurahman", "hafiz naeemurrahman",
            "hafi naeem", "hafi naeemeurahman", "hafi naeemurahman",
            "hafi naeemurrehman", "hafi naeem ur rehman",
            "hafiz naeem urrehman", "hafiz naeem urrahman",
            "hafiz naem rehman", "hafiz naim rehman",
            "hafiz neem", "hafiz neem ur rehman", "hafiz neem rehman",
            "hafz naeem ur rehman", "hafiz naam", "hafiz naam ur rehman",
            "hafis naeem ur rehman", "haafiz naeem", "haafiz naeem ur rehman",
            "hafiz naeem u rehman", "hafiz naeem u rahman",
            "hafiz naeem oorehman", "hafiz naeem oorahman",
            "hafiz naeem-rehman", "naeem-ur-rahman", "naeem ur rahman",
            "naem ur rehman", "naem rehman", "naim ur rehman", "naim rehman",
            "naaem ur rehman", "naaem rehman", "naeem rehmaan", "naeem rehmann",
            "naeemur rahman", "naeemurrahman", "naeemeurrehman", "naeemeurahman",
            "naeem urehman", "naeem ure hman",
            
            # === Short/abbreviated forms ===
            "h naeem ur rehman", "h. naeem", "h.naeem",
            "h naeem rehman", "hn rehman", "hnr", "h.n.r.",
            "hafiz n", "hafiz nr", "h n r",
            
            # === Case variations (handled by lowering, but listed for completeness) ===
            "HAFIZ NAEEM", "Hafiz Naeem", "NAEEM UR REHMAN", "NAEEM REHMAN",
            
            # === Special characters / punctuation variants ===
            "hafiz_naeem", "hafiz.naeem", "hafiz,naeem",
            "naeem_ur_rehman", "naeem.ur.rehman",
            "hafiz naeem'ur rehman", "hafiz naeem`ur rehman",
            
            # === Urdu script forms (various spellings) ===
            "حافظ نعیم", "حافظ نعیم الرحمن", "نعیم الرحمن", "حافظ نعیم ورحمان",
            "حافظ نعیم الرحمان", "نعیم الرحمان", "حافظ نعیم ور رحمان",
            "نعیم", "حافظ نعیم رحمان", "نعیم رحمان",
            "حافظ نائیم", "نائیم الرحمن", "حافظ نائیم الرحمن",
            "حافظ نعیم ار رحمن", "حافظ نعیم اور رحمن",
            "حافظ صاحب", "نعیم صاحب", "نعیم بھائی",
            
            # === Urdu transliterations (Roman Urdu) ===
            "hafiz sahab", "hafiz sahib", "naeem sahab", "naeem sahib",
            "naeem bhai", "hafiz bhai",
            
            # === Common informal references ===
            "ameer jamaat", "ameer e jamaat", "jamaati ameer",
            "ameer jamat", "ameer-e-jamat", "ameer jamat e islami",
            "ameer jamaat e islami", "ameer ji", "ameer sahab",
            "ameer jama'at", "ameer jamaat islami",
        ],
        # Speaker field values stored in Qdrant (various forms used at ingestion time)
        "speaker_variants": [
            "Hafiz Naeem Ur Rehman", "Hafiz Naeem", "hafiz naeem ur rehman",
            "Hafiz Naeem ur Rehman", "HAFIZ NAEEM", "Naeem Ur Rehman",
            "Hafiz Naeem Ur Rahman", "Naeem Ur Rahman", "Naeem Rehman",
            "Hafiz Naeem Rehman", "hafiz naeem", "naeem ur rehman",
        ],
    },
}

# Build a quick reverse-lookup: lowercased alias → person key
_ALIAS_LOOKUP: Dict[str, str] = {}
for _person_key, _person_data in PERSON_ALIASES.items():
    for _alias in _person_data["aliases"]:
        _ALIAS_LOOKUP[_alias.lower().strip()] = _person_key


def detect_person_alias(text: str) -> Optional[str]:
    """
    Returns the PERSON_ALIASES key if the text matches any known alias,
    otherwise returns None.
    Checks exact matches, substring matches, and fuzzy matches for typo tolerance.
    """
    if not text:
        return None
    text_lower = text.lower().strip()
    
    # Remove common punctuation/special chars that users might accidentally type
    text_cleaned = re.sub(r'[_.,;:!?\'"`~@#$%^&*()+=\[\]{}|\\/<>]', ' ', text_lower)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()

    # 1. Exact match against aliases
    if text_lower in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[text_lower]
    if text_cleaned != text_lower and text_cleaned in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[text_cleaned]

    # 2. Check if text contains any known alias as a whole-word substring
    for alias, person_key in _ALIAS_LOOKUP.items():
        # Skip very short single-word aliases (e.g. "naeem", "hafiz") for substring search
        # to avoid false positives — only use them on exact match (handled above)
        if len(alias.split()) == 1 and len(alias) <= 7:
            continue
        if alias in text_lower:
            return person_key

    # 3. Fuzzy matching for typos/misspellings (e.g. "hafi naeemeurahman")
    #    Only check multi-word aliases or aliases >= 6 chars to avoid false positives
    if len(text_cleaned) >= 5:
        best_score = 0
        best_key = None
        for alias, person_key in _ALIAS_LOOKUP.items():
            # Skip short single-word aliases for fuzzy matching
            if len(alias) < 6 and len(alias.split()) == 1:
                continue
            
            # Use both ratio and partial_ratio for different matching scenarios
            score = fuzz.ratio(text_cleaned, alias)
            partial = fuzz.partial_ratio(text_cleaned, alias)
            
            # For concatenated forms (naeemurrehman vs naeem ur rehman),
            # also compare without spaces
            text_no_space = text_cleaned.replace(' ', '')
            alias_no_space = alias.replace(' ', '')
            no_space_score = fuzz.ratio(text_no_space, alias_no_space)
            
            effective_score = max(score, partial * 0.9, no_space_score)
            
            # Threshold depends on length — longer strings need lower threshold
            threshold = 75 if len(alias) >= 10 else 80 if len(alias) >= 6 else 90
            
            if effective_score >= threshold and effective_score > best_score:
                best_score = effective_score
                best_key = person_key
        
        if best_key:
            return best_key
    
    # 4. Word-component matching: check if key name parts appear in text
    #    e.g. "hafiz speech about economy" → detect "hafiz"
    #    This catches queries where a known name is embedded in a longer phrase
    text_words = text_cleaned.split()
    if len(text_words) >= 1:
        for person_key, person_data in PERSON_ALIASES.items():
            canonical_words = [w.lower() for w in person_data["canonical"].split()]
            # Count how many canonical name words appear in the text (exact or fuzzy)
            matched_canonical = 0
            for cw in canonical_words:
                if len(cw) < 3:  # skip "ur", "e" etc.
                    continue
                for tw in text_words:
                    if tw == cw or (len(tw) >= 4 and len(cw) >= 4 and fuzz.ratio(tw, cw) >= 82):
                        matched_canonical += 1
                        break
            # If 2+ significant canonical words match, it's likely this person
            significant_canonical = [w for w in canonical_words if len(w) >= 3]
            if len(significant_canonical) >= 2 and matched_canonical >= 2:
                return person_key

    return None


def expand_speaker_from_aliases(speaker_query: str) -> List[str]:
    """
    Given a speaker query that may be an alias, returns a list of all
    speaker name variants to search against.
    If no alias is found, returns the original query wrapped in a list.
    """
    person_key = detect_person_alias(speaker_query)
    if person_key:
        data = PERSON_ALIASES[person_key]
        # Combine canonical + all speaker_variants (deduplicated, preserving order)
        seen = set()
        result = []
        for name in [data["canonical"]] + data["speaker_variants"]:
            if name.lower() not in seen:
                seen.add(name.lower())
                result.append(name)
        return result
    return [speaker_query]


def expand_query_with_aliases(query: str) -> Tuple[str, List[str]]:
    """
    If query contains a known person alias, expands it to include the
    canonical name. Returns (expanded_query, extra_keyword_terms).
    """
    person_key = detect_person_alias(query)
    if not person_key:
        return query, []
    data = PERSON_ALIASES[person_key]
    canonical = data["canonical"]
    # Replace the alias in the query with the canonical name
    # and collect extra keyword terms for keyword search
    extra_terms = [canonical] + data["speaker_variants"][:3]
    # Keep the original query but also inject the canonical name
    expanded = f"{query} {canonical}" if canonical.lower() not in query.lower() else query
    return expanded, extra_terms


# ============== Initialize OpenAI client if enabled
openai_client = None
if USE_OPENAI_EMBEDDINGS:
    if not OPENAI_API_KEY:
        logger.warning("USE_OPENAI_EMBEDDINGS is true but OPENAI_API_KEY not set. Falling back to sentence-transformers.")
        USE_OPENAI_EMBEDDINGS = False
    else:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL} (dim={EMBEDDING_DIMENSION})")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# ============== PYDANTIC MODELS FOR INPUT VALIDATION ==============
class EmbedVideoRequest(BaseModel):
    video_id: int = Field(..., gt=0, description="Video ID must be positive")
    identification_segments: List[dict] = Field(..., min_length=1)
    video_title: str = Field(default="", max_length=500)
    video_filename: str = Field(default="", max_length=500)
    youtube_url: Optional[str] = Field(default="", max_length=1000)
    language: Optional[str] = Field(default="", max_length=50)
    batch_info: Optional[dict] = Field(default=None, description="Batch processing info: batch_number, total_batches, segments_in_batch")
    speakers_transcript: Optional[List[dict]] = Field(default=None, description="Full speakers transcript (ignored)")
    diarization_segments: Optional[List[dict]] = Field(default=None, description="Diarization segments (ignored)")
    # Enriched metadata for dual-mode search (stored in Qdrant payload)
    video_created_at: Optional[str] = Field(default=None, description="ISO datetime of when video was created")
    processing_status: str = Field(default="completed", max_length=50)
    approval_status: str = Field(default="approved", max_length=50)
    is_archived: bool = Field(default=False)
    user_id: Optional[int] = Field(default=None)
    speakers_count: int = Field(default=0)
    audio_duration_seconds: float = Field(default=0)
    video_description: str = Field(default="", max_length=5000)
    video_summary: str = Field(default="", max_length=10000)
    video_summary_english: str = Field(default="", max_length=10000)
    video_summary_urdu: str = Field(default="", max_length=10000)

class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=1000)
    words: List[str] = Field(default=[])
    word: Optional[str] = Field(default=None, max_length=200)
    top_k: int = Field(default=20, ge=1, le=500)
    video_id: Optional[int] = Field(default=None, gt=0)
    speaker: Optional[str] = Field(default=None, max_length=200)
    title: Optional[str] = Field(default=None, max_length=500)
    language: Optional[str] = Field(default=None, max_length=50)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    time_range: Optional[dict] = None
    max_scanned: int = Field(default=10000, ge=100, le=100000)
    # Dual-mode search: 'semantic' (vector + LLM) or 'simple' (structured filters only)
    search_mode: str = Field(default="semantic", pattern="^(semantic|simple)$")
    # Simple mode: which single filter is active
    filter_type: Optional[str] = Field(default=None, pattern="^(video|speaker|date|language|title|summary|text)$")
    # Date filter fields for simple mode date filtering within Qdrant
    filter_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    filter_month: Optional[int] = Field(default=None, ge=1, le=12)
    filter_date: Optional[str] = Field(default=None, max_length=20)

class SuggestRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    type: str = Field(default="both", pattern="^(speaker|title|both)$")
    limit: int = Field(default=10, ge=1, le=50)

class TitleSearchRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)

class IncrementalSearchRequest(BaseModel):
    """Request model for incremental cursor-based search"""
    query: str = Field(default="", max_length=1000)
    words: List[str] = Field(default=[])
    word: Optional[str] = Field(default=None, max_length=200)
    video_id: Optional[int] = Field(default=None, gt=0)
    speaker: Optional[str] = Field(default=None, max_length=200)
    title: Optional[str] = Field(default=None, max_length=500)
    language: Optional[str] = Field(default=None, max_length=50)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    time_range: Optional[dict] = None
    max_scanned: int = Field(default=10000, ge=100, le=100000)
    search_mode: str = Field(default="semantic", pattern="^(semantic|simple)$")
    filter_type: Optional[str] = Field(default=None, pattern="^(video|speaker|date|language|title|summary|text)$")
    filter_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    filter_month: Optional[int] = Field(default=None, ge=1, le=12)
    filter_date: Optional[str] = Field(default=None, max_length=20)
    # Cursor-based pagination fields
    cursor: Optional[str] = Field(default=None, description="Base64-encoded cursor for pagination")
    batch_size: int = Field(default=10, ge=1, le=50, description="Number of results per batch")
    search_session_id: Optional[str] = Field(default=None, max_length=100, description="UUID to identify search session")

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

# Cache for search results (max 1000 search sessions, 30 minute TTL)
# Key: search_session_id (UUID), Value: {results: List[Dict], query_params: Dict, timestamp: float}
search_result_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes

# ============== FASTEMBED MODEL (lightweight CPU fallback) ==============
# FastEmbed: ~200MB RAM vs ~2GB for sentence-transformers+PyTorch
# BAAI/bge-small-en-v1.5 is fast and English-focused (384-dim)
# Used ONLY as fallback when OpenAI API is unavailable
FASTEMBED_MODEL_NAME = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_DIMENSION = 384  # Dimension for bge-small-en-v1.5

# Lazy-loaded to avoid blocking startup; initialized on first use
_fastembed_model = None

def get_fastembed_model() -> TextEmbedding:
    """Return the FastEmbed model, initializing it on first call."""
    global _fastembed_model
    if _fastembed_model is None:
        logger.info(f"Loading FastEmbed model: {FASTEMBED_MODEL_NAME}...")
        _fastembed_model = TextEmbedding(model_name=FASTEMBED_MODEL_NAME)
        logger.info("FastEmbed model loaded successfully")
        logger.info("Warming up FastEmbed model...")
        _ = list(_fastembed_model.embed(["warmup text"]))
        logger.info("FastEmbed model warmed up and ready")
    return _fastembed_model

# ============== CURSOR-BASED PAGINATION FUNCTIONS ==============

import base64

def encode_cursor(segment_id: str, score: float, index: int) -> str:
    """
    Encode cursor as base64 JSON string.
    Cursor format: {id: segment_id, score: score, index: position_in_results}
    """
    cursor_data = {
        "id": segment_id,
        "score": float(score),
        "index": int(index)
    }
    cursor_json = json_module.dumps(cursor_data)
    cursor_b64 = base64.b64encode(cursor_json.encode('utf-8')).decode('utf-8')
    return cursor_b64

def decode_cursor(cursor_str: Optional[str]) -> Optional[Dict]:
    """
    Decode base64 cursor string to dict.
    Returns None if cursor is invalid or None.
    """
    if not cursor_str:
        return None
    
    try:
        cursor_json = base64.b64decode(cursor_str.encode('utf-8')).decode('utf-8')
        cursor_data = json_module.loads(cursor_json)
        return cursor_data
    except Exception as e:
        logger.warning(f"Failed to decode cursor: {e}")
        return None

def get_cached_results(search_session_id: str) -> Optional[Dict]:
    """
    Retrieve cached search results by session ID.
    Returns cached data or None if not found/expired.
    """
    if not search_session_id:
        return None
    
    cached = search_result_cache.get(search_session_id)
    if cached:
        logger.info(f"Cache HIT for session {search_session_id[:8]}... ({len(cached.get('results', []))} results)")
    else:
        logger.info(f"Cache MISS for session {search_session_id[:8] if search_session_id else 'None'}...")
    return cached

def cache_search_results(search_session_id: str, results: List[Dict], query_params: Dict, top_k_used: int = 20):
    """
    Cache search results with session ID.
    Stores top_k used to enable progressive query expansion.
    """
    cache_data = {
        "results": results,
        "query_params": query_params,
        "timestamp": time.time(),
        "top_k_used": top_k_used  # Track how many results were queried
    }
    search_result_cache[search_session_id] = cache_data
    logger.info(f"Cached {len(results)} results (top_k={top_k_used}) for session {search_session_id[:8]}...")

def extract_batch_from_results(results: List[Dict], cursor: Optional[Dict], batch_size: int) -> Tuple[List[Dict], Optional[str], bool]:
    """
    Extract a batch of results starting from cursor position.
    Returns segments until we have batch_size unique VIDEO groups (not raw segments).
    This ensures the PHP controller gets enough segments to display batch_size videos.
    
    Args:
        results: Full list of search results (flat segments)
        cursor: Decoded cursor dict with {id, score, index}
        batch_size: Number of unique videos to return segments for
    
    Returns:
        Tuple of (batch_results, next_cursor_str, has_more)
    """
    if not results:
        return [], None, False
    
    # Determine starting index
    start_index = 0
    if cursor and 'index' in cursor:
        start_index = cursor['index'] + 1  # Start after the cursor position
    
    # Collect segments until we have batch_size unique video_ids
    batch = []
    unique_video_ids = set()
    end_index = start_index
    
    for i in range(start_index, len(results)):
        result = results[i]
        video_id = result.get('video_id')
        
        # If this is a new video and we already have enough videos, stop
        if video_id and video_id not in unique_video_ids and len(unique_video_ids) >= batch_size:
            break
        
        if video_id:
            unique_video_ids.add(video_id)
        batch.append(result)
        end_index = i
    
    if not batch:
        return [], None, False
    
    # Determine if there are more results
    has_more = (end_index + 1) < len(results)
    
    # Create next cursor if there are more results
    next_cursor = None
    if has_more and batch:
        last_result = batch[-1]
        next_cursor = encode_cursor(
            segment_id=last_result.get('id', ''),
            score=last_result.get('score', 0.0),
            index=end_index
        )
    
    logger.info(f"Batch extracted: start={start_index}, end={end_index + 1}, segments={len(batch)}, unique_videos={len(unique_video_ids)}, has_more={has_more}")
    return batch, next_cursor, has_more

def can_expand_incremental_cache(results_count: int, top_k_used: int, max_top_k: int = 200) -> bool:
    """
    Decide whether we should advertise another cursor page so the next request can
    trigger progressive top_k expansion.

    We only do this when:
      1) We have not yet reached max_top_k
      2) Current result count is near the queried top_k (likely capped)
    """
    if top_k_used >= max_top_k:
        return False
    return results_count >= max(1, int(top_k_used * 0.9))

def maybe_mark_expandable_boundary(
    batch: List[Dict],
    next_cursor: Optional[str],
    has_more: bool,
    all_results: List[Dict],
    top_k_used: int,
    max_top_k: int = 200
) -> Tuple[Optional[str], bool, bool]:
    """
    If we reached the end of currently cached results but cache is expandable,
    return a boundary cursor and has_more=True so frontend can request the next
    page and trigger progressive expansion.

    Returns:
      (next_cursor, has_more, expansion_pending)
    """
    # If normal pagination already has more, do nothing.
    if has_more:
        return next_cursor, has_more, False

    # No batch means nothing was returned; do not force continuation.
    if not batch or not all_results:
        return next_cursor, has_more, False

    # If cache does not look capped/expandable, do nothing.
    if not can_expand_incremental_cache(len(all_results), top_k_used, max_top_k=max_top_k):
        return next_cursor, has_more, False

    boundary_index = len(all_results) - 1
    boundary_result = all_results[boundary_index]
    boundary_cursor = encode_cursor(
        segment_id=boundary_result.get('id', ''),
        score=boundary_result.get('score', 0.0),
        index=boundary_index
    )

    logger.info(
        f"[INCREMENTAL SEARCH] Reached expandable boundary at index={boundary_index}; "
        f"exposing cursor to allow next request expansion (top_k={top_k_used}, results={len(all_results)})."
    )

    return boundary_cursor, True, True

# ============== ADVANCED EMBEDDING FUNCTIONS ==============

def get_openai_embedding(text: str, model_name: str = None) -> List[float]:
    """
    Get embedding from OpenAI API with advanced text-embedding-3 models.
    These models provide significantly better semantic understanding.
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized")
    
    model_name = model_name or OPENAI_EMBEDDING_MODEL
    
    try:
        # Replace newlines to avoid API issues
        text = text.replace("\n", " ").strip()
        
        response = openai_client.embeddings.create(
            input=text,
            model=model_name,
            dimensions=EMBEDDING_DIMENSION  # Can reduce dimensions for faster search
        )
        
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"OpenAI embedding error: {str(e)}")
        # No fallback - FastEmbed produces 384-dim vectors which fail on 3072-dim collection
        raise

def get_openai_embeddings_batch(texts: List[str], model_name: str = None) -> List[List[float]]:
    """
    Get embeddings for multiple texts using OpenAI API.
    Handles sub-batching for large inputs (API limit is 2048 inputs per call).
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized")
    
    model_name = model_name or OPENAI_EMBEDDING_MODEL
    
    try:
        # Clean texts
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        
        # Sub-batch for OpenAI API token limit (max 300,000 tokens per request)
        # Some videos have very long segments (10K+ tokens), so use small batches
        # 20 segments * ~10K tokens max = 200K tokens, safely under 300K limit
        SUB_BATCH_SIZE = 20
        all_embeddings = []
        
        for i in range(0, len(cleaned_texts), SUB_BATCH_SIZE):
            sub_batch = cleaned_texts[i:i + SUB_BATCH_SIZE]
            logger.info(f"OpenAI sub-batch {i // SUB_BATCH_SIZE + 1}: {len(sub_batch)} texts")
            
            response = openai_client.embeddings.create(
                input=sub_batch,
                model=model_name,
                dimensions=EMBEDDING_DIMENSION
            )
            
            # Sort by index to maintain order within sub-batch
            sub_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(sub_embeddings)
        
        return all_embeddings
    except Exception as e:
        logger.error(f"OpenAI batch embedding error: {str(e)}")
        # No fallback - FastEmbed produces 384-dim vectors which fail on 3072-dim collection
        raise

def expand_query_with_gpt(query: str) -> List[str]:
    """
    DEPRECATED: Use understand_query() instead.
    Kept for backward compatibility.
    """
    if not openai_client or not query.strip():
        return [query]
    result = understand_query(query)
    variations = [result.get("semantic_query", query)]
    if result.get("semantic_query_translated"):
        variations.append(result["semantic_query_translated"])
    variations.extend(result.get("expanded_terms", [])[:3])
    return variations[:5]


def understand_query(query: str) -> Dict:
    """
    LLM-powered query understanding using GPT-4o-mini.
    Replaces simple query expansion with structured intent parsing.
    Handles Urdu, English, and mixed queries.
    """
    # Check cache first
    cache_key = f"intent_{query.strip().lower()[:500]}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    default_result = {
        "semantic_query": query,
        "semantic_query_translated": "",
        "detected_language": "unknown",
        "extracted_speaker": None,
        "extracted_keywords": query.split(),
        "expanded_terms": [],
        "query_type": "general"
    }
    
    if not openai_client or not query.strip():
        return default_result
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a multilingual search query analyzer for a video transcript search engine.
Content is primarily in Urdu with some English. Given a user query (may be Urdu, English, Roman Urdu, or mixed), extract structured search intent.

Return ONLY valid JSON with these fields:
- semantic_query: the core search meaning (keep original language)
- semantic_query_translated: translate to the OTHER language (Urdu→English, English→Urdu transliteration, if same language leave empty)
- detected_language: "urdu" | "english" | "roman_urdu" | "mixed"
- extracted_speaker: person name ONLY if the user is clearly searching for content SPOKEN BY that person, else null
- extracted_keywords: array of key terms for keyword matching
- expanded_terms: 3-5 synonyms/related terms in BOTH languages (mix of Urdu and English)
- query_type: "speaker_search" | "topic_search" | "quote_search" | "title_search" | "general"

IMPORTANT RULES for extracted_speaker:
1. ONLY set extracted_speaker when user wants content BY/FROM that person (e.g. "Hafiz Naeem speech" = user wants Hafiz Naeem's speech)
2. Do NOT extract speaker if the query looks like a VIDEO TITLE (e.g. "The Future Mark Zuckerberg Is Trying To Build" is a title ABOUT Zuckerberg, not content BY him)
3. Do NOT extract speaker if the person is the SUBJECT/TOPIC of the query rather than the SPEAKER
4. When in doubt, set extracted_speaker to null - it's better to not filter than to wrongly filter

Examples:
Query: "Hafiz Naeem election speech"
{"semantic_query": "Hafiz Naeem election speech", "semantic_query_translated": "حافظ نعیم الیکشن تقریر", "detected_language": "english", "extracted_speaker": "Hafiz Naeem", "extracted_keywords": ["election", "speech"], "expanded_terms": ["انتخابات", "vote", "ووٹ", "political rally", "سیاسی جلسہ"], "query_type": "speaker_search"}

Query: "ملک کے حالات"
{"semantic_query": "ملک کے حالات", "semantic_query_translated": "situation of the country", "detected_language": "urdu", "extracted_speaker": null, "extracted_keywords": ["ملک", "حالات"], "expanded_terms": ["country situation", "Pakistan crisis", "پاکستان", "معیشت", "economy"], "query_type": "topic_search"}

Query: "The Future Mark Zuckerberg Is Trying To Build"
{"semantic_query": "The Future Mark Zuckerberg Is Trying To Build", "semantic_query_translated": "", "detected_language": "english", "extracted_speaker": null, "extracted_keywords": ["future", "Zuckerberg", "build", "Meta"], "expanded_terms": ["metaverse", "VR", "virtual reality", "Facebook", "technology"], "query_type": "title_search"}

Query: "what did Imran Khan say about economy"
{"semantic_query": "what did Imran Khan say about economy", "semantic_query_translated": "عمران خان نے معیشت کے بارے میں کیا کہا", "detected_language": "english", "extracted_speaker": "Imran Khan", "extracted_keywords": ["economy", "Imran Khan"], "expanded_terms": ["معیشت", "economic policy", "PTI", "finance"], "query_type": "speaker_search"}"""},
                {"role": "user", "content": f"Analyze this search query: {query}"}
            ],
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json_module.loads(response.choices[0].message.content.strip())
        
        # Validate required fields exist
        for key in ["semantic_query", "detected_language"]:
            if key not in result:
                result[key] = default_result[key]
        
        # Ensure arrays
        if not isinstance(result.get("expanded_terms"), list):
            result["expanded_terms"] = []
        if not isinstance(result.get("extracted_keywords"), list):
            result["extracted_keywords"] = query.split()
        
        # Cache result
        embedding_cache[cache_key] = result
        logger.info(f"Query understood: lang={result.get('detected_language')}, speaker={result.get('extracted_speaker')}, type={result.get('query_type')}")
        return result
        
    except Exception as e:
        logger.warning(f"Query understanding error: {str(e)}")
        return default_result


def validate_query_relevance(query: str, top_results: List[Dict]) -> Dict:
    """
    Validate if ANY of the top search results are actually relevant to the query.
    This prevents showing completely unrelated results (e.g., "bill gates" returning Pakistan politics).
    
    Returns dict with:
    - is_relevant: bool - True if at least one result is relevant
    - max_relevance: float - Highest relevance score (0-1)
    - relevant_count: int - Number of results with score >= 0.5
    - explanation: str - Why results were deemed irrelevant
    """
    if not openai_client or not top_results or not query.strip():
        # If no LLM available, assume results are valid
        return {"is_relevant": True, "max_relevance": 1.0, "relevant_count": len(top_results), "explanation": ""}
    
    try:
        # Check top 5 results only (fast validation)
        check_results = top_results[:5]
        
        # Build compact summary for validation
        docs_summary = []
        for i, r in enumerate(check_results):
            text = r.get("text", "")[:150]
            speaker = r.get("speaker", "")
            title = r.get("video_title", "")[:80]
            docs_summary.append(f"[{i+1}] {title} | Speaker: {speaker} | Text: {text}")
        
        docs_text = "\n".join(docs_summary)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a relevance validator for a search engine. Your job is to determine if search results actually match the user's query.

Return ONLY valid JSON: {"relevant": boolean, "scores": [score1, score2, ...], "explanation": "reason"}

Score each result 0-10:
- 8-10: Directly discusses the query topic/person
- 5-7: Related to the query domain
- 2-4: Loosely connected or tangential
- 0-1: Completely unrelated

Set "relevant": true ONLY if at least one result scores >= 5.
Set "relevant": false if ALL results are off-topic.

Be STRICT: If searching for "Bill Gates" but results are about Pakistan politics, that's irrelevant.
If searching for a person/topic not in the database, all results will be off-topic."""},
                {"role": "user", "content": f"Query: {query}\n\nTop Results:\n{docs_text}\n\nAre these results relevant to the query?"}
            ],
            max_tokens=300,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json_module.loads(response.choices[0].message.content.strip())
        is_relevant = result.get("relevant", True)
        scores = result.get("scores", [])
        explanation = result.get("explanation", "")
        
        # Normalize scores to 0-1
        normalized_scores = [s / 10.0 for s in scores if isinstance(s, (int, float))]
        max_relevance = max(normalized_scores) if normalized_scores else 0.0
        relevant_count = sum(1 for s in normalized_scores if s >= 0.5)
        
        logger.info(f"Query relevance validation: is_relevant={is_relevant}, max_score={max_relevance:.2f}, relevant_count={relevant_count}/{len(check_results)}")
        if not is_relevant:
            logger.info(f"Irrelevant query detected: {explanation}")
        
        return {
            "is_relevant": is_relevant,
            "max_relevance": max_relevance,
            "relevant_count": relevant_count,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.warning(f"Query relevance validation error: {str(e)}")
        # On error, assume results are valid to avoid blocking searches
        return {"is_relevant": True, "max_relevance": 1.0, "relevant_count": len(top_results), "explanation": ""}


def rerank_with_llm(query: str, results: List[Dict], top_k: int = 20) -> List[Dict]:
    """
    LLM-based reranking using GPT-4o-mini.
    Replaces Cohere (English-only) with multilingual GPT-4o-mini reranker.
    Handles Urdu, English, and mixed content properly.
    
    STRICT SCORING: Only truly relevant results get high scores.
    Irrelevant results (wrong topic/person) get scores < 3/10.
    """
    if not openai_client or not results or not query.strip():
        return results[:top_k]
    
    try:
        # Take top 30 candidates for reranking (cost efficient)
        candidates = results[:min(len(results), 30)]
        
        # Build compact document list for the LLM
        docs = []
        for i, r in enumerate(candidates):
            text = r.get("text", "")[:200]  # Truncate for token efficiency
            speaker = r.get("speaker", "")
            title = r.get("video_title", "")
            docs.append(f"[{i}] Speaker: {speaker} | Title: {title} | Text: {text}")
        
        docs_text = "\n".join(docs)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a STRICT search result relevance judge for a video transcript search engine.
Content may be in Urdu, English, or mixed. Rate each transcript segment's relevance to the user's search query.

Return ONLY valid JSON: {"scores": [{"i": 0, "s": 8}, {"i": 1, "s": 3}, ...]}
Where "i" is the document index and "s" is relevance score 0-10.

Scoring guide (BE STRICT):
- 10: Perfect match - directly answers the query, exact topic/person
- 7-9: Highly relevant - clearly discusses the queried topic/person
- 4-6: Partially relevant - mentions topic but not the main focus
- 1-3: Marginally relevant - only loosely connected, different topic
- 0: Completely unrelated - wrong topic/person/domain entirely

CRITICAL RULES:
- If query is "Bill Gates" but result is about Pakistan politics → score 0-1
- If query is about technology but result is about religion → score 0-2
- If query person/topic is NOT mentioned in the result → score 0-3
- Only give scores >= 7 if the result is ACTUALLY ABOUT what the user searched for

Be VERY strict. Most results should get low scores if they don't match the query topic."""}, 
                {"role": "user", "content": f"Search Query: {query}\n\nDocuments:\n{docs_text}"}
            ],
            max_tokens=500,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        llm_result = json_module.loads(response.choices[0].message.content.strip())
        scores = llm_result.get("scores", [])
        
        # Build score map
        score_map = {}
        for item in scores:
            idx = item.get("i", item.get("index", -1))
            score = item.get("s", item.get("score", 5))
            if 0 <= idx < len(candidates):
                score_map[idx] = score / 10.0  # Normalize to 0-1
        
        # Apply LLM scores
        reranked = []
        for i, r in enumerate(candidates):
            result_copy = r.copy()
            llm_score = score_map.get(i, 0.3)  # Default to LOW score (0.3) if not scored
            
            # Combined score: LLM (60%) + original semantic (25%) + keyword/fuzzy (15%)
            # INCREASED LLM weight to 60% so strict LLM scores have more impact
            original_score = result_copy.get("score", 0)
            result_copy["llm_relevance_score"] = round(llm_score, 4)
            result_copy["original_score"] = original_score
            result_copy["score"] = round(
                llm_score * 0.60 + original_score * 0.25 + result_copy.get("fuzzy_score", 0) * 0.15,
                4
            )
            
            if "match_types" not in result_copy:
                result_copy["match_types"] = []
            result_copy["match_types"].append("llm_reranked")
            reranked.append(result_copy)
        
        # STRICTER filtering: LLM score < 0.3 (3/10) is considered irrelevant
        # But ALWAYS keep title matches and exact phrase matches (these are verified matches)
        reranked = [
            r for r in reranked 
            if r.get("llm_relevance_score", 0) >= 0.3 
            or "title_match" in r.get("match_types", []) 
            or "exact_phrase_match" in r.get("match_types", [])
        ]
        
        # Sort by new combined score
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Add any remaining results that weren't reranked (beyond top 30)
        reranked_ids = {r["id"] for r in reranked}
        for r in results[30:]:
            if r["id"] not in reranked_ids:
                reranked.append(r)
        
        # Log statistics
        high_relevance = sum(1 for r in reranked if r.get("llm_relevance_score", 0) >= 0.7)
        med_relevance = sum(1 for r in reranked if 0.5 <= r.get("llm_relevance_score", 0) < 0.7)
        low_relevance = sum(1 for r in reranked if 0.3 <= r.get("llm_relevance_score", 0) < 0.5)
        filtered_count = len(candidates) - len([r for r in reranked if r.get("llm_relevance_score")])
        
        logger.info(f"LLM reranking: {len(candidates)} candidates -> {len(reranked)} results (high={high_relevance}, med={med_relevance}, low={low_relevance}, filtered={filtered_count})")
        return reranked[:top_k]
        
    except Exception as e:
        logger.warning(f"LLM reranking error: {str(e)}")
        return results[:top_k]


def rerank_simple_results(query: str, results: List[Dict], filter_type: str, top_k: int = 200) -> List[Dict]:
    """
    LLM-based reranking for simple search mode results.
    Uses GPT-4o-mini with filter-type-aware prompts to validate and score
    each result's actual relevance, then combines with original match score.
    
    Filter-type-aware scoring:
      - speaker: Does the speaker name match the query? Is it the right person?
      - text: Does the transcript text actually discuss the query topic?
      - title: Does the video title match what the user is looking for?
      - summary: Does the summary relate to the query topic?
    
    Returns reranked results with llm_relevance_score populated.
    Falls back to original results if OpenAI is unavailable.
    """
    if not openai_client or not results or not query.strip():
        return results[:top_k]
    
    # Skip reranking for very small result sets (not worth the latency)
    if len(results) < 3:
        return results[:top_k]
    
    # Filter-type-specific system prompts
    FILTER_PROMPTS = {
        "speaker": """You are a speaker name matching judge for a video transcript search engine.
The user searched for a SPEAKER NAME. For each result, judge how well the speaker field matches the query.

Scoring guide:
- 10: Exact name match (e.g., query "Sam Altman", speaker "Sam Altman")
- 8-9: Clear match with minor variation (e.g., query "sam", speaker "Sam Altman" — first name matches)
- 5-7: Partial match — one name component matches (e.g., query "Ali", speaker "Ali Hassan")
- 2-4: Weak match — fuzzy similarity but different person likely
- 0-1: No match — completely different name

Focus ONLY on whether the speaker name matches the query. Ignore transcript content.""",

        "text": """You are a transcript relevance judge for a video transcript search engine.
Content may be in Urdu, English, or mixed. The user searched for specific text/topic in transcripts.
For each result, judge how relevant the transcript text is to the search query.

Scoring guide:
- 10: Text directly discusses the exact query topic
- 7-9: Highly relevant — clearly related to the query
- 4-6: Partially relevant — mentions related concepts
- 1-3: Marginally relevant — loosely connected
- 0: Completely unrelated

Be strict: if the query topic is not present in the text, score 0-2.""",

        "title": """You are a video title matching judge for a video transcript search engine.
The user searched for a video by title. For each result, judge how well the video title matches the query.

Scoring guide:
- 10: Exact title match
- 8-9: Very close match — all key words present
- 5-7: Partial match — some key words present
- 2-4: Weak match — only loosely related
- 0-1: No match — completely different title

Focus on whether the title matches the search query.""",

        "summary": """You are a video summary relevance judge for a video transcript search engine.
Content may be in Urdu, English, or mixed. The user searched for a topic in video summaries.
For each result, judge how relevant the video summary is to the search query.

Scoring guide:
- 10: Summary directly covers the query topic
- 7-9: Highly relevant — clearly discusses related material
- 4-6: Partially relevant — touches on the topic
- 1-3: Marginally relevant — loosely connected
- 0: Completely unrelated

Be strict: the summary should actually discuss the query topic to score above 5."""
    }
    
    system_prompt = FILTER_PROMPTS.get(filter_type)
    if not system_prompt:
        return results[:top_k]
    
    try:
        # Take top 30 candidates for reranking (cost efficient)
        candidates = results[:min(len(results), 30)]
        
        # Build compact document list based on filter type
        docs = []
        for i, r in enumerate(candidates):
            if filter_type == "speaker":
                speaker = r.get("speaker", "") or r.get("diarization_speaker", "")
                text_preview = r.get("text", "")[:100]
                docs.append(f"[{i}] Speaker: {speaker} | Text: {text_preview}")
            elif filter_type == "text":
                text = r.get("text", "")[:200]
                speaker = r.get("speaker", "")
                docs.append(f"[{i}] Speaker: {speaker} | Text: {text}")
            elif filter_type == "title":
                title = r.get("video_title", "")
                docs.append(f"[{i}] Title: {title}")
            elif filter_type == "summary":
                title = r.get("video_title", "")
                docs.append(f"[{i}] Title: {title}")
        
        docs_text = "\n".join(docs)
        
        full_prompt = system_prompt + """\n\nReturn ONLY valid JSON: {"scores": [{"i": 0, "s": 8}, {"i": 1, "s": 3}, ...]}
Where "i" is the document index and "s" is relevance score 0-10."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"Search Query: {query}\n\nResults:\n{docs_text}"}
            ],
            max_tokens=500,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        llm_result = json_module.loads(response.choices[0].message.content.strip())
        scores = llm_result.get("scores", [])
        
        # Build score map
        score_map = {}
        for item in scores:
            idx = item.get("i", item.get("index", -1))
            score = item.get("s", item.get("score", 5))
            if 0 <= idx < len(candidates):
                score_map[idx] = score / 10.0  # Normalize to 0-1
        
        # Apply LLM scores: combined = 60% LLM + 40% original match score
        reranked = []
        for i, r in enumerate(candidates):
            result_copy = r.copy()
            llm_score = score_map.get(i, 0.3)  # Default to LOW if not scored
            original_score = result_copy.get("score", 0)
            
            result_copy["llm_relevance_score"] = round(llm_score, 4)
            result_copy["original_score"] = original_score
            result_copy["score"] = round(
                llm_score * 0.60 + original_score * 0.40,
                4
            )
            
            if "match_types" not in result_copy:
                result_copy["match_types"] = []
            result_copy["match_types"].append("llm_reranked")
            reranked.append(result_copy)
        
        # Filter out results with LLM score < 0.3 (3/10 = irrelevant)
        reranked = [r for r in reranked if r.get("llm_relevance_score", 0) >= 0.3]
        
        # Sort by new combined score
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Append any remaining results beyond top 30 (unranked)
        reranked_ids = {r["id"] for r in reranked}
        for r in results[30:]:
            if r["id"] not in reranked_ids:
                reranked.append(r)
        
        # Log statistics
        high_rel = sum(1 for r in reranked if r.get("llm_relevance_score", 0) >= 0.7)
        med_rel = sum(1 for r in reranked if 0.5 <= r.get("llm_relevance_score", 0) < 0.7)
        low_rel = sum(1 for r in reranked if 0.3 <= r.get("llm_relevance_score", 0) < 0.5)
        filtered = len(candidates) - len([r for r in reranked if r.get("llm_relevance_score")])
        
        logger.info(f"[SIMPLE RERANK] filter={filter_type}, query='{query[:50]}': {len(candidates)} candidates -> {len(reranked)} results (high={high_rel}, med={med_rel}, low={low_rel}, filtered={filtered})")
        return reranked[:top_k]
        
    except Exception as e:
        logger.warning(f"[SIMPLE RERANK ERROR] {str(e)}")
        return results[:top_k]


# ============== ELASTIC SEARCH HELPERS ==============

def get_cached_embedding(text: str) -> List[float]:
    """Get embedding with caching for repeated queries. Uses OpenAI if enabled."""
    cache_key = text.strip().lower()[:500]  # Normalize and limit key length
    if cache_key not in embedding_cache:
        if USE_OPENAI_EMBEDDINGS and openai_client:
            logger.info(f"Using OpenAI embeddings for query")
            embedding_cache[cache_key] = get_openai_embedding(text)
        else:
            logger.info("Using FastEmbed for query embedding")
            result = list(get_fastembed_model().embed([text]))
            embedding_cache[cache_key] = result[0].tolist() if hasattr(result[0], 'tolist') else list(result[0])
    return embedding_cache[cache_key]

def fuzzy_match_text(query: str, text: str, threshold: int = 65) -> bool:
    """
    Fuzzy match with typo tolerance using rapidfuzz.
    Returns True if query fuzzy-matches text above threshold.
    Uses whole-word matching to prevent 'tan' matching 'pakistan' etc.
    """
    if not query or not text:
        return False
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Whole-word match (prevents substring false positives)
    if whole_word_match(query_lower, text_lower):
        return True
    
    # Fuzzy match against individual words (not substrings)
    if fuzzy_word_match(query_lower, text_lower, threshold) > 0:
        return True
    
    return False

def fuzzy_match_speaker(query: str, speaker: str, threshold: int = 75) -> bool:
    """
    Fuzzy match for speaker names with higher threshold.
    Handles variations like 'John' vs 'Jon', 'Muhammad' vs 'Mohammad'.
    Also checks PERSON_ALIASES so that 'HNR', 'naeem', etc. match
    'Hafiz Naeem Ur Rehman' and vice-versa.
    Also handles concatenated forms like 'naeemurrehman' vs 'naeem ur rehman'.
    """
    if not query or not speaker:
        return False

    query_lower = query.lower().strip()
    speaker_lower = speaker.lower().strip()
    
    # Clean special characters from query (users may type underscores, dots, etc.)
    query_cleaned = re.sub(r'[_.,;:!?\'"]+', ' ', query_lower)
    query_cleaned = re.sub(r'\s+', ' ', query_cleaned).strip()

    # Exact match
    if query_lower == speaker_lower or query_lower in speaker_lower:
        return True
    if query_cleaned != query_lower and (query_cleaned == speaker_lower or query_cleaned in speaker_lower):
        return True
    
    # No-space comparison (handles concatenated forms like "naeemurrehman")
    query_no_space = query_cleaned.replace(' ', '')
    speaker_no_space = speaker_lower.replace(' ', '').replace('_', '')
    if query_no_space == speaker_no_space or query_no_space in speaker_no_space or speaker_no_space in query_no_space:
        return True
    if len(query_no_space) >= 5 and len(speaker_no_space) >= 5:
        no_space_ratio = fuzz.ratio(query_no_space, speaker_no_space)
        if no_space_ratio >= threshold:
            return True

    # ── Alias expansion: if query is a known alias, try all speaker_variants ──
    person_key = detect_person_alias(query)
    if person_key:
        data = PERSON_ALIASES[person_key]
        for variant in data["speaker_variants"]:
            vl = variant.lower().strip()
            if vl == speaker_lower or vl in speaker_lower or speaker_lower in vl:
                return True
            if fuzz.ratio(vl, speaker_lower) >= threshold:
                return True
            # Also check no-space variant comparison
            vl_no_space = vl.replace(' ', '')
            if vl_no_space == speaker_no_space or vl_no_space in speaker_no_space:
                return True

    # ── Alias expansion: if speaker is a known alias, try all speaker_variants ──
    person_key_s = detect_person_alias(speaker)
    if person_key_s:
        data_s = PERSON_ALIASES[person_key_s]
        for variant in data_s["speaker_variants"]:
            vl = variant.lower().strip()
            if vl == query_lower or vl in query_lower or query_lower in vl:
                return True
            if fuzz.ratio(vl, query_lower) >= threshold:
                return True

    # Check each word in speaker name — require stricter matching for short words
    speaker_parts = speaker_lower.split()
    for part in speaker_parts:
        # Skip non-name parts like "SPEAKER_01", "SPEAKER_02" etc.
        if part.startswith("speaker_") or part.startswith("speaker "):
            continue
        # For short words (<=4 chars), require near-exact match to prevent 'cleo'→'ceo'
        effective_threshold = threshold
        if len(query_lower) <= 4 or len(part) <= 4:
            effective_threshold = max(threshold, 95)
            # Also require length difference <= 1 for short words
            if abs(len(query_lower) - len(part)) > 1:
                continue
        if fuzz.ratio(query_lower, part) >= effective_threshold:
            return True

    # Full fuzzy match — also stricter for short queries
    full_threshold = threshold
    if len(query_lower) <= 4:
        full_threshold = max(threshold, 90)
    return fuzz.ratio(query_lower, speaker_lower) >= full_threshold

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

def expand_query_variations(query: str) -> List[str]:
    """
    Generate query variations for better recall.
    Handles common typos and variations.
    """
    variations = [query]
    query_lower = query.lower().strip()
    
    if query_lower not in [v.lower() for v in variations]:
        variations.append(query_lower)
    
    # Add individual words for multi-word queries
    words = query.split()
    if len(words) > 1:
        variations.extend(words)
    
    return list(set(variations))

# Punctuation normalization table: curly quotes → straight, strip misc punctuation
_PUNCT_NORMALIZE_TABLE = str.maketrans({
    '\u2018': "'", '\u2019': "'",  # curly single quotes → straight
    '\u201C': '"', '\u201D': '"',  # curly double quotes → straight
    '\u2013': '-', '\u2014': '-',  # en/em dash → hyphen
    '\u2026': ' ',                  # ellipsis → space
})
_STRIP_CHARS = _string.punctuation + '\u2018\u2019\u201C\u201D\u2013\u2014\u2026'

def normalize_for_matching(text: str) -> str:
    """Normalize text for keyword matching: lowercase, normalize quotes/dashes, strip edge punctuation."""
    return text.translate(_PUNCT_NORMALIZE_TABLE).lower()

def normalize_word(word: str) -> str:
    """Normalize a single search word: lowercase, normalize quotes, strip surrounding punctuation."""
    return word.translate(_PUNCT_NORMALIZE_TABLE).strip(_STRIP_CHARS).lower()


def whole_word_match(word: str, text: str) -> bool:
    """
    Check if 'word' appears as a whole word in 'text', NOT as a substring
    of a larger word. Uses regex word boundaries (\b).
    Prevents 'tan' matching 'pakistan', 'dance' matching 'abundance', etc.
    Both inputs should be pre-normalized (lowercase).
    """
    if not word or not text:
        return False
    try:
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text, re.UNICODE))
    except re.error:
        return word in text  # Safe fallback


def whole_phrase_match(phrase: str, text: str) -> bool:
    """
    Check if 'phrase' appears as whole words in 'text'.
    Prevents 'dance hall' matching 'abundance hallway'.
    Both inputs should be pre-normalized (lowercase).
    """
    if not phrase or not text:
        return False
    try:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        return bool(re.search(pattern, text, re.UNICODE))
    except re.error:
        return phrase in text  # Safe fallback


def word_variant_match(word: str, text: str) -> bool:
    """
    Match common word variants for short/medium query terms.
    Example: 'fail' should match 'failed', 'failing', 'failure'.
    """
    if not word or not text:
        return False

    w = normalize_word(word)
    t = normalize_for_matching(text)
    if not w or not t:
        return False

    # Keep this conservative to avoid very broad matches on short terms.
    if len(w) < 4:
        return False

    try:
        # Word starts with query stem and has a small suffix.
        # Allows: fail, failed, failing, failure.
        pattern = r"\b" + re.escape(w) + r"[a-z]{0,6}\b"
        if re.search(pattern, t, re.UNICODE):
            return True
    except re.error:
        pass

    return False


def fuzzy_word_match(word: str, text: str, threshold: int = 85) -> float:
    """
    Fuzzy match a word against individual words in text (not substrings).
    Returns the best match score (0-1) or 0 if no match above threshold.
    Prevents 'dance' fuzzy-matching 'abundance' by comparing word-to-word.
    For short words (<=4 chars), enforces stricter matching to prevent
    false positives like 'cleo' matching 'ceo'.
    """
    if not word or not text:
        return 0.0
    # Short words need stricter thresholds to prevent false positives
    # e.g., 'cleo' vs 'ceo' = 86% which passes 80% but shouldn't match
    effective_threshold = threshold
    if len(word) <= 4:
        effective_threshold = max(threshold, 95)  # Very strict for short words
    elif len(word) <= 6:
        effective_threshold = max(threshold, 90)  # Stricter for medium words
    best_score = 0.0
    for text_word in text.split():
        # Only compare against words of comparable length
        if len(text_word) < 2:
            continue
        # For short words (<=5 chars), require length difference <= 1
        # This prevents 'cleo'(4) matching 'ceo'(3), but allows 'sam'(3) matching 'sam'(3)
        if len(word) <= 5 and abs(len(word) - len(text_word)) > 1:
            continue
        ratio = fuzz.ratio(word, text_word)
        if ratio >= effective_threshold and ratio > best_score:
            best_score = ratio
    return best_score / 100.0 if best_score >= effective_threshold else 0.0


def query_closeness_score(query_words: List[str], text: str, threshold: int = 80) -> float:
    """
    Estimate how closely query words match result text (0-1).
    - Exact whole-word matches contribute 1.0
    - Fuzzy close matches contribute in [threshold..1.0]
    Returns average closeness across query words.
    """
    if not query_words or not text:
        return 0.0

    normalized_text = normalize_for_matching(text)
    scores = []
    for qw in query_words:
        qn = normalize_word(qw)
        if not qn:
            continue
        if whole_word_match(qn, normalized_text):
            scores.append(1.0)
            continue
        fw = fuzzy_word_match(qn, normalized_text, threshold=threshold)
        if fw > 0:
            scores.append(fw)
        else:
            scores.append(0.0)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def calculate_combined_score(semantic_score: float, keyword_match: bool, fuzzy_score: float = 0) -> float:
    """
    Calculate combined relevance score.
    Semantic is the primary signal, keyword and fuzzy are bonuses.
    Score ranges:
      - Pure semantic: up to 0.90
      - Semantic + keyword: up to 0.95
      - Semantic + keyword + fuzzy: up to 1.0
    """
    base_score = semantic_score * 0.90  # Semantic is primary — preserve most of cosine similarity
    if keyword_match:
        base_score += 0.07  # Keyword bonus
    if fuzzy_score > 0:
        base_score += fuzzy_score * 0.05  # Small fuzzy bonus
    return min(base_score, 1.0)  # Cap at 1.0

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
            logger.info(f"Creating collection '{SEGMENTS_COLLECTION}' with dimension={EMBEDDING_DIMENSION}...")
            qdrant_client.create_collection(
                collection_name=SEGMENTS_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000)
            )
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
            # Full-text index on transcript text for keyword search
            {"field_name": "text", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
            # Full-text index on summary_en for cross-language keyword search
            {"field_name": "summary_en", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
            # Full-text index on speaker for speaker name text search
            {"field_name": "speaker", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
            # Full-text index on diarization_speaker for speaker name text search
            {"field_name": "diarization_speaker", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
            # ── Enriched metadata indexes for dual-mode search ──
            {"field_name": "video_created_at", "field_schema": "keyword"},
            {"field_name": "processing_status", "field_schema": "keyword"},
            {"field_name": "approval_status", "field_schema": "keyword"},
            {"field_name": "is_archived", "field_schema": "bool"},
            {"field_name": "user_id", "field_schema": "integer"},
            {"field_name": "speakers_count", "field_schema": "integer"},
            {"field_name": "audio_duration_seconds", "field_schema": "float"},
            # Full-text indexes on video summaries for simple-mode summary search
            {"field_name": "video_summary", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
            {"field_name": "video_summary_english", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
            {"field_name": "video_summary_urdu", "field_schema": {"type": "text", "tokenizer": "word", "min_token_len": 2, "max_token_len": 30, "lowercase": True}},
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
        
        # Enriched metadata for dual-mode search (stored in Qdrant payload)
        video_created_at = data.video_created_at or ""
        processing_status = data.processing_status or "completed"
        approval_status = data.approval_status or "approved"
        is_archived = data.is_archived
        user_id = data.user_id
        speakers_count = data.speakers_count
        audio_duration_seconds = data.audio_duration_seconds
        video_description = data.video_description or ""
        video_summary = data.video_summary or ""
        video_summary_english = data.video_summary_english or ""
        video_summary_urdu = data.video_summary_urdu or ""
        
        logger.info(f"Processing video {video_id} with {len(identification_segments)} segments")
        
        # Only delete existing embeddings on the FIRST batch (or if no batch info = single request)
        batch_number = 1
        total_batches = 1
        if data.batch_info:
            batch_number = data.batch_info.get("batch_number", 1)
            total_batches = data.batch_info.get("total_batches", 1)
            logger.info(f"Batch {batch_number}/{total_batches} for video {video_id}")
        
        if batch_number == 1:
            delete_existing_embeddings(video_id)
        else:
            logger.info(f"Skipping delete for batch {batch_number} (only delete on batch 1)")
        
        points = []
        segments_embedded = 0
        segments_without_text = 0
        texts_to_embed = []
        segment_metadata = []
        
        for idx, segment in enumerate(identification_segments):
            # Use segment_index from payload if provided (preserves global index across batches)
            # Fall back to enumerate index for backward compatibility
            segment_index = segment.get("segment_index", idx)
            
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
            
            # BUILD ENRICHED EMBEDDING TEXT — includes speaker, title, language context
            # This makes vector search find segments by speaker name and video context
            enriched_parts = []
            if speaker and speaker != "UNKNOWN":
                enriched_parts.append(f"[Speaker: {speaker}]")
            if video_title:
                enriched_parts.append(f"[Title: {video_title}]")
            if language:
                enriched_parts.append(f"[Language: {language}]")
            enriched_parts.append(text)
            
            enriched_text = " ".join(enriched_parts)
            
            # Truncate to max 6000 chars (~1500-2000 tokens) to stay under OpenAI limits
            if len(enriched_text) > 6000:
                enriched_text = enriched_text[:6000] + "..."
                logger.debug(f"Truncated segment {segment_index} from {len(text)} to 6000 chars")
            
            texts_to_embed.append(enriched_text)
            segment_metadata.append({
                'idx': segment_index,  # Use global segment_index, not batch-local idx
                'speaker':  speaker,
                'diarization_speaker': diarization_speaker,
                'match_type': match_type,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'text': text,
                'enriched_text': enriched_text
            })
        
        if not texts_to_embed: 
            raise HTTPException(
                status_code=400,
                detail=f"No valid segments found to embed.  Total:  {len(identification_segments)}, Without text: {segments_without_text}"
            )
        
        logger.info(f"Generating embeddings for {len(texts_to_embed)} segments in batch...")
        batch_start_time = datetime.utcnow()
        
        # Use OpenAI embeddings if enabled, otherwise use FastEmbed
        if USE_OPENAI_EMBEDDINGS and openai_client:
            logger.info(f"Using OpenAI {OPENAI_EMBEDDING_MODEL} for batch embedding")
            vectors = get_openai_embeddings_batch(texts_to_embed)
        else:
            logger.info("Using FastEmbed for batch embedding")
            results_list = list(get_fastembed_model().embed(texts_to_embed))
            vectors = [r.tolist() if hasattr(r, 'tolist') else list(r) for r in results_list]
        
        # GENERATE LLM ENGLISH SUMMARIES for non-English segments (bilingual bridge)
        summaries_en = {}
        if openai_client and language and language.lower() not in ["english", "en"]:
            try:
                logger.info(f"Generating English summaries for {len(segment_metadata)} segments...")
                # Process in batches of 20 for efficiency
                for batch_start in range(0, len(segment_metadata), 20):
                    batch_end = min(batch_start + 20, len(segment_metadata))
                    batch_texts = []
                    for m in segment_metadata[batch_start:batch_end]:
                        batch_texts.append(f"[{m['idx']}] {m['text'][:300]}")
                    
                    prompt_text = "\n".join(batch_texts)
                    summary_response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Translate/summarize each numbered transcript segment into a brief English summary (1 line each). Return one summary per line, prefixed with the segment number in brackets. Keep it concise."},
                            {"role": "user", "content": prompt_text}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    
                    # Parse summaries
                    for line in summary_response.choices[0].message.content.strip().split("\n"):
                        line = line.strip()
                        if line and "[" in line:
                            try:
                                idx_str = line.split("]")[0].replace("[", "").strip()
                                summary_text = "]".join(line.split("]")[1:]).strip().lstrip("- :")
                                summaries_en[int(idx_str)] = summary_text
                            except (ValueError, IndexError):
                                pass
                
                logger.info(f"Generated {len(summaries_en)} English summaries")
            except Exception as e:
                logger.warning(f"Summary generation failed (non-critical): {str(e)}")
        
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        logger.info(f"Batch embedding completed in {batch_duration:.2f} seconds")
        
        for i, metadata in enumerate(segment_metadata):
            vector = vectors[i]
            id_string = f"video_{video_id}_seg_{metadata['idx']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))
            
            # Get English summary if available
            summary_en = summaries_en.get(metadata['idx'], "")
            
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
                "summary_en": summary_en,  # English summary for cross-language search
                "enriched_text": metadata.get('enriched_text', metadata['text']),  # context-rich text
                "created_at": datetime.utcnow().isoformat(),
                # Enriched metadata for dual-mode search eligibility & filtering
                "video_created_at": video_created_at,
                "processing_status": processing_status,
                "approval_status": approval_status,
                "is_archived": is_archived,
                "user_id": user_id,
                "speakers_count": speakers_count,
                "audio_duration_seconds": audio_duration_seconds,
                "video_description": video_description,
                "video_summary": video_summary,
                "video_summary_english": video_summary_english,
                "video_summary_urdu": video_summary_urdu,
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
            # Sub-batch upsert to stay under Qdrant's 32MB payload limit
            # Each point is ~70KB (3072-dim vector + payload), so 150 points ≈ 10MB (safe)
            QDRANT_BATCH_SIZE = 150
            for i in range(0, len(points), QDRANT_BATCH_SIZE):
                sub_batch = points[i:i + QDRANT_BATCH_SIZE]
                logger.info(f"Upserting Qdrant sub-batch {i // QDRANT_BATCH_SIZE + 1}: {len(sub_batch)} points")
                qdrant_client.upsert(
                    collection_name=SEGMENTS_COLLECTION,
                    points=sub_batch,
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
    
    # Use FastEmbed for legacy endpoint
    result = list(get_fastembed_model().embed([text]))
    vector = result[0].tolist() if hasattr(result[0], 'tolist') else list(result[0])
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
    ADVANCED SEARCH with:
    - OpenAI embeddings for superior semantic understanding
    - GPT-powered query expansion for better recall
    - Cohere reranking for improved relevance
    - Elastic/fuzzy matching for speaker and title
    - Automatically searches for names in both transcript text AND speaker field
    
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
    search_mode = data.search_mode  # 'semantic' or 'simple'
    filter_type = data.filter_type  # 'video', 'speaker', 'date', 'language', 'title', 'summary'
    filter_year = data.filter_year
    filter_month = data.filter_month
    filter_date = data.filter_date

    # OpenAI-only mode: avoid scan-heavy fallbacks and prefer vector+LLM flow.
    openai_only_search = os.getenv("OPENAI_ONLY_SEARCH", "true").lower() == "true"
    if openai_only_search and not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_ONLY_SEARCH is enabled but OpenAI client is not configured"
        )

    # Keep simple mode on its own code path. OPENAI_ONLY_SEARCH should not reroute
    # simple filters into semantic mode, otherwise simple behavior becomes inconsistent.

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMPLE SEARCH MODE — structured filters only, no vector/LLM operations.
    # Only one filter is active at a time. Returns results from Qdrant scroll.
    # Eligibility: processing_status=completed, approval_status=approved, is_archived=false
    # ═══════════════════════════════════════════════════════════════════════════
    if search_mode == "simple":
        logger.info(f"[SIMPLE SEARCH] filter_type={filter_type}, query='{query_text}', speaker={speaker_filter}, video_id={video_id_filter}, language={language_filter}, title={title_filter}")
        
        # Build eligibility filter — only searchable videos
        eligibility_conditions = [
            FieldCondition(key="processing_status", match=MatchValue(value="completed")),
            FieldCondition(key="approval_status", match=MatchValue(value="approved")),
            FieldCondition(key="is_archived", match=MatchValue(value=False)),
        ]
        
        simple_results = []
        scanned = 0
        max_scan_simple = min(max_scanned, 50000)
        
        try:
            if filter_type == "video":
                # Filter by specific video ID
                if not video_id_filter:
                    raise HTTPException(status_code=400, detail="video_id is required for video filter")
                scroll_conditions = eligibility_conditions + [
                    FieldCondition(key="video_id", match=MatchValue(value=video_id_filter))
                ]
                scroll_filter = Filter(must=scroll_conditions)
                offset = None
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        simple_results.append({
                            "id": p.id,
                            "score": 1.0,
                            "video_id": payload.get("video_id"),
                            "video_title": payload.get("video_title", ""),
                            "speaker": payload.get("speaker", ""),
                            "diarization_speaker": payload.get("diarization_speaker", ""),
                            "start_time": payload.get("start_time", 0),
                            "end_time": payload.get("end_time", 0),
                            "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                            "text": payload.get("text", ""),
                            "text_length": payload.get("text_length", 0),
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_video_filter"],
                            "fuzzy_score": 1.0,
                            "matched_field": "video_id",
                        })
                    if len(simple_results) >= top_k:
                        break
                    offset = next_offset
                    if not next_offset:
                        break

            elif filter_type == "speaker":
                # Speaker search — fuzzy match against speaker fields
                # Supports multi-word names like "Hafiz Naeem" matching individual word components
                if not speaker_filter:
                    raise HTTPException(status_code=400, detail="speaker is required for speaker filter")
                scroll_filter = Filter(must=eligibility_conditions) if eligibility_conditions else None
                offset = None
                min_speaker_score = 0.45  # Minimum score threshold for quality
                # Split speaker query into words for multi-word matching
                speaker_query_words = [w.lower().strip() for w in speaker_filter.split() if len(w.strip()) >= 2]
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        spk = payload.get("speaker", "")
                        diar_spk = payload.get("diarization_speaker", "")
                        speaker_combined = f"{spk} {diar_spk}".strip()
                        if not speaker_combined:
                            continue
                        
                        # Try the existing fuzzy_match_speaker (handles aliases, exact match, etc.)
                        alias_match = fuzzy_match_speaker(speaker_filter, speaker_combined, threshold=75)
                        
                        if not alias_match:
                            # Multi-word fallback: check if each word in the query
                            # appears in the speaker fields (handles "rehan Tariq" etc.)
                            if len(speaker_query_words) > 1:
                                speaker_lower = speaker_combined.lower()
                                words_found = sum(1 for sw in speaker_query_words if whole_word_match(sw, speaker_lower))
                                if words_found < len(speaker_query_words):
                                    continue  # Not all words found
                            else:
                                continue  # Single word didn't match fuzzy_match_speaker
                        
                        # Calculate match score — use individual speaker name (spk), NOT combined
                        # Combined string includes diarization_speaker (e.g. "SPEAKER_01") which
                        # dilutes fuzz.ratio and causes partial/exact name matches to score below threshold
                        spk_lower = spk.lower().strip() if spk else ""
                        if len(speaker_query_words) > 1:
                            speaker_lower = spk_lower or speaker_combined.lower()
                            words_found = sum(1 for sw in speaker_query_words if whole_word_match(sw, speaker_lower))
                            word_coverage = words_found / max(len(speaker_query_words), 1)
                            partial_score = fuzz.partial_ratio(speaker_filter.lower(), speaker_lower) / 100
                            match_score = max(word_coverage * 0.85, partial_score * 0.80)
                        else:
                            # For single-word queries ("sam", "cleo"), use the best of:
                            # 1) ratio against speaker name only (not combined)
                            # 2) partial_ratio against speaker name (handles substring matches)
                            query_low = speaker_filter.lower().strip()
                            ratio_spk = fuzz.ratio(query_low, spk_lower) / 100 if spk_lower else 0
                            partial_spk = fuzz.partial_ratio(query_low, spk_lower) / 100 if spk_lower else 0
                            match_score = max(ratio_spk, partial_spk * 0.90)
                        
                        # Enforce minimum quality threshold
                        if match_score < min_speaker_score:
                            continue
                        simple_results.append({
                            "id": p.id,
                            "score": round(match_score, 4),  # REMOVED score inflation (was max(match_score, 0.5))
                            "video_id": payload.get("video_id"),
                            "video_title": payload.get("video_title", ""),
                            "speaker": spk,
                            "diarization_speaker": diar_spk,
                            "start_time": payload.get("start_time", 0),
                            "end_time": payload.get("end_time", 0),
                            "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                            "text": payload.get("text", ""),
                            "text_length": payload.get("text_length", 0),
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_speaker_filter"],
                            "fuzzy_score": round(match_score, 2),
                            "matched_field": "speaker",
                        })
                    if len(simple_results) >= top_k:
                        break
                    offset = next_offset
                    if not next_offset:
                        break

            elif filter_type == "date":
                # Date filter — match video_created_at using prefix or exact match
                if not filter_year and not filter_date:
                    raise HTTPException(status_code=400, detail="filter_year or filter_date is required for date filter")
                scroll_conditions = list(eligibility_conditions)
                if filter_date:
                    # Exact date match via keyword prefix (ISO: "2024-03-15")
                    scroll_conditions.append(
                        FieldCondition(key="video_created_at", match=MatchText(text=filter_date))
                    )
                elif filter_year and filter_month:
                    prefix = f"{filter_year}-{filter_month:02d}"
                    scroll_conditions.append(
                        FieldCondition(key="video_created_at", match=MatchText(text=prefix))
                    )
                elif filter_year:
                    scroll_conditions.append(
                        FieldCondition(key="video_created_at", match=MatchText(text=str(filter_year)))
                    )
                scroll_filter = Filter(must=scroll_conditions)
                offset = None
                seen_video_ids = set()
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        vid = payload.get("video_id")
                        # For date filter, return one entry per video
                        if vid in seen_video_ids:
                            continue
                        seen_video_ids.add(vid)
                        simple_results.append({
                            "id": p.id,
                            "score": 1.0,
                            "video_id": vid,
                            "video_title": payload.get("video_title", ""),
                            "speaker": "",
                            "diarization_speaker": "",
                            "start_time": 0,
                            "end_time": 0,
                            "duration": 0,
                            "text": "",
                            "text_length": 0,
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_date_filter"],
                            "fuzzy_score": 1.0,
                            "matched_field": "video_created_at",
                            "is_video_only": True,
                        })
                    if len(simple_results) >= top_k:
                        break
                    offset = next_offset
                    if not next_offset:
                        break

            elif filter_type == "language":
                # Language filter — exact match on language field
                if not language_filter:
                    raise HTTPException(status_code=400, detail="language is required for language filter")
                scroll_conditions = eligibility_conditions + [
                    FieldCondition(key="language", match=MatchValue(value=language_filter))
                ]
                scroll_filter = Filter(must=scroll_conditions)
                offset = None
                seen_video_ids = set()
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        vid = payload.get("video_id")
                        if vid in seen_video_ids:
                            continue
                        seen_video_ids.add(vid)
                        simple_results.append({
                            "id": p.id,
                            "score": 1.0,
                            "video_id": vid,
                            "video_title": payload.get("video_title", ""),
                            "speaker": "",
                            "diarization_speaker": "",
                            "start_time": 0,
                            "end_time": 0,
                            "duration": 0,
                            "text": "",
                            "text_length": 0,
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_language_filter"],
                            "fuzzy_score": 1.0,
                            "matched_field": "language",
                            "is_video_only": True,
                        })
                    if len(simple_results) >= top_k:
                        break
                    offset = next_offset
                    if not next_offset:
                        break

            elif filter_type == "title":
                # Title search — word-level matching against video_title
                # Exact-title matches should score 1.0.
                # Otherwise, require exact-word coverage (no fuzzy title fallback).
                if not title_filter:
                    raise HTTPException(status_code=400, detail="title is required for title filter")
                scroll_filter = Filter(must=eligibility_conditions) if eligibility_conditions else None
                offset = None
                seen_video_ids = set()
                exact_title_results = []
                partial_title_results = []
                min_title_score = 0.45  # Minimum score threshold
                # Split query into individual words for word-level matching
                title_query_words = [normalize_word(w) for w in title_filter.split() if len(normalize_word(w)) >= 2]
                if not title_query_words:
                    title_query_words = [normalize_word(title_filter)]
                logger.info(f"Simple title search: query='{title_filter}', words={title_query_words}")
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        vid = payload.get("video_id")
                        if vid in seen_video_ids:
                            continue
                        video_title = payload.get("video_title", "")
                        if not video_title:
                            continue
                        # Token-normalize both sides so punctuation/spacing/case variants
                        # still count as exact title matches.
                        video_title_tokens = [normalize_word(w) for w in video_title.split() if normalize_word(w)]
                        query_tokens = [normalize_word(w) for w in title_filter.split() if normalize_word(w)]
                        video_title_norm = " ".join(video_title_tokens)
                        query_normalized = " ".join(query_tokens)
                        
                        # STRATEGY 1: strict normalized full-title equality
                        is_full_title_match = query_normalized == video_title_norm

                        # STRATEGY 2: exact-word coverage in title
                        words_matched = 0.0
                        for qw in title_query_words:
                            # Check whole-word match (prevents 'tan' matching 'pakistan')
                            if whole_word_match(qw, video_title_norm):
                                words_matched += 1
                            elif len(qw) >= 6 and fuzzy_word_match(qw, video_title_norm, threshold=90) > 0:
                                # Very conservative typo tolerance for long words only.
                                words_matched += 0.6
                        
                        word_coverage = words_matched / max(len(title_query_words), 1)

                        # Calculate final match score
                        if is_full_title_match:
                            # Full title match → 1.0 (perfect score)
                            match_score = 1.0
                        elif word_coverage >= 1.0:
                            # All query words found (exact words) → high confidence
                            match_score = 0.90
                        else:
                            # Not enough overlap — reject
                            continue
                        
                        # Enforce minimum quality threshold
                        if match_score < min_title_score:
                            continue
                        
                        seen_video_ids.add(vid)
                        result_item = {
                            "id": p.id,
                            "score": round(match_score, 4),
                            "video_id": vid,
                            "video_title": video_title,
                            "speaker": "",
                            "diarization_speaker": "",
                            "start_time": 0,
                            "end_time": 0,
                            "duration": 0,
                            "text": "",
                            "text_length": 0,
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_title_filter", "simple_exact_title_match"] if is_full_title_match else ["simple_title_filter"],
                            "fuzzy_score": round(match_score, 2),
                            "matched_field": "video_title",
                            "is_video_only": True,
                        }

                        if is_full_title_match:
                            exact_title_results.append(result_item)
                        else:
                            partial_title_results.append(result_item)

                    if len(exact_title_results) >= top_k or (not exact_title_results and len(partial_title_results) >= top_k):
                        break
                    offset = next_offset
                    if not next_offset:
                        break
                # If exact title exists, return ONLY exact title matches.
                if exact_title_results:
                    exact_title_results.sort(key=lambda x: x["score"], reverse=True)
                    simple_results = exact_title_results[:top_k]
                else:
                    partial_title_results.sort(key=lambda x: x["score"], reverse=True)
                    simple_results = partial_title_results[:top_k]

            elif filter_type == "summary":
                # Summary keyword search — STRICT whole-word matching in video summaries
                # Prevents false matches like 'tan' matching 'pakistan', 'tech' matching 'technology'
                search_query = query_text or " ".join(words or [])
                if not search_query:
                    raise HTTPException(status_code=400, detail="query is required for summary filter")
                scroll_filter = Filter(must=eligibility_conditions) if eligibility_conditions else None
                # Normalize and filter search words (>= 2 chars to allow 'ai', 'gpt', etc.)
                search_words_lower = [normalize_word(w) for w in search_query.split() if len(normalize_word(w)) >= 2]
                if not search_words_lower:
                    raise HTTPException(status_code=400, detail="query must contain at least one word with 2+ characters for summary filter")
                offset = None
                seen_video_ids = set()
                min_summary_score = 0.40  # Minimum score threshold
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        vid = payload.get("video_id")
                        if vid in seen_video_ids:
                            continue
                        # Combine all summary fields and normalize
                        summary_all = normalize_for_matching(" ".join([
                            payload.get("video_summary", ""),
                            payload.get("video_summary_english", ""),
                            payload.get("video_summary_urdu", ""),
                        ]))
                        if not summary_all.strip():
                            continue
                        # WHOLE-WORD matching: prevents 'tan' matching 'pakistan'
                        matched_count = sum(1 for w in search_words_lower if whole_word_match(w, summary_all))
                        if matched_count == 0:
                            continue
                        # Calculate accurate score: require high word coverage for good scores
                        match_score = matched_count / max(len(search_words_lower), 1)
                        # Enforce minimum quality threshold
                        if match_score < min_summary_score:
                            continue
                        seen_video_ids.add(vid)
                        simple_results.append({
                            "id": p.id,
                            "score": round(min(match_score, 1.0), 4),
                            "video_id": vid,
                            "video_title": payload.get("video_title", ""),
                            "speaker": "",
                            "diarization_speaker": "",
                            "start_time": 0,
                            "end_time": 0,
                            "duration": 0,
                            "text": "",
                            "text_length": 0,
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_summary_filter"],
                            "fuzzy_score": round(match_score, 2),
                            "matched_field": "video_summary",
                            "is_video_only": True,
                        })
                    if len(simple_results) >= top_k:
                        break
                    offset = next_offset
                    if not next_offset:
                        break
                simple_results.sort(key=lambda x: x["score"], reverse=True)

            elif filter_type == "text":
                # Text search — STRICT whole-word matching against segment text
                # Prevents false matches like 'dance' matching 'abundance'
                # Supports numeric searches (e.g., "15", "2023", etc.)
                search_query = query_text or " ".join(words or [])
                if not search_query:
                    raise HTTPException(status_code=400, detail="Please enter search text")
                
                # Normalize and filter search words (>= 1 char to allow single digits like '1', '2', '5')
                # BUT skip stop words to avoid overly broad matches
                search_words_raw = [w.strip() for w in search_query.split() if w.strip()]
                search_words_normalized = []
                skipped_words = []
                
                for w in search_words_raw:
                    normalized = normalize_word(w)
                    # Skip empty results
                    if not normalized:
                        continue
                    # Skip stop words UNLESS it's a pure number (numbers are always valid search terms)
                    if normalized.lower() in STOP_WORDS and not normalized.isdigit():
                        skipped_words.append(w)
                        continue
                    search_words_normalized.append(normalized)
                
                if not search_words_normalized:
                    if skipped_words:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Your search only contains common words that are too broad to search: {', '.join(skipped_words[:3])}. Please add more specific terms."
                        )
                    else:
                        raise HTTPException(
                            status_code=400, 
                            detail="Please enter valid search text (letters, numbers, or meaningful words)"
                        )
                
                logger.info(f"Text search: query='{search_query}' → normalized_terms={search_words_normalized} (skipped: {skipped_words})")
                # Build Qdrant scroll with text match conditions (Qdrant pre-filtering)
                scroll_conditions = list(eligibility_conditions)
                # Add MatchText for each search word on the 'text' field
                # For numeric-only searches, use fewer constraints to improve recall
                is_numeric_search = all(w.replace('.', '').replace('-', '').replace(',', '').isdigit() for w in search_words_normalized)
                max_filter_words = 3 if is_numeric_search else 5
                
                for sw in search_words_normalized[:max_filter_words]:  # Limit words to avoid overly strict filters
                    scroll_conditions.append(
                        FieldCondition(key="text", match=MatchText(text=sw))
                    )
                
                if is_numeric_search:
                    logger.info(f"Numeric-only search detected: using relaxed filtering (max {max_filter_words} terms)")
                scroll_filter = Filter(must=scroll_conditions)
                offset = None
                min_text_score = 0.40  # Minimum score threshold
                while scanned < max_scan_simple and len(simple_results) < top_k:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=min(1000, max_scan_simple - scanned),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not points:
                        break
                    for p in points:
                        scanned += 1
                        payload = p.payload or {}
                        seg_text = payload.get("text", "")
                        if not seg_text:
                            continue
                        # Normalize text for accurate matching
                        seg_text_normalized = normalize_for_matching(seg_text)
                        # WHOLE-WORD matching: prevents 'tan' matching 'pakistan', 'dance' matching 'abundance'
                        # For numeric searches, also check substring matches to catch numbers embedded in text
                        matched_count = 0
                        for w in search_words_normalized:
                            if whole_word_match(w, seg_text_normalized):
                                matched_count += 1
                            elif is_numeric_search and w in seg_text_normalized:
                                # Allow substring match for numbers (e.g., "15" in "15th" or "2015")
                                matched_count += 0.7  # Slightly lower weight for substring match
                        
                        if matched_count == 0:
                            continue
                        
                        # Calculate accurate score: require high word coverage
                        match_score = matched_count / max(len(search_words_normalized), 1)
                        # Enforce minimum quality threshold (lower for numeric searches)
                        effective_min_score = min_text_score * 0.7 if is_numeric_search else min_text_score
                        if match_score < effective_min_score:
                            continue
                        simple_results.append({
                            "id": p.id,
                            "score": round(min(match_score, 1.0), 4),
                            "video_id": payload.get("video_id"),
                            "video_title": payload.get("video_title", ""),
                            "speaker": payload.get("speaker", ""),
                            "diarization_speaker": payload.get("diarization_speaker", ""),
                            "start_time": payload.get("start_time", 0),
                            "end_time": payload.get("end_time", 0),
                            "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                            "text": seg_text,
                            "text_length": payload.get("text_length", 0),
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "video_created_at": payload.get("video_created_at", ""),
                            "match_types": ["simple_text_filter"],
                            "fuzzy_score": round(match_score, 2),
                            "matched_field": "text",
                        })
                    if len(simple_results) >= top_k:
                        break
                    offset = next_offset
                    if not next_offset:
                        break
                
                logger.info(f"Text search complete: found {len(simple_results)} matches from {scanned} scanned segments (query: '{search_query}', terms: {search_words_normalized})")
                simple_results.sort(key=lambda x: x["score"], reverse=True)

            else:
                raise HTTPException(status_code=400, detail=f"Invalid filter_type: {filter_type}. Must be one of: video, speaker, date, language, title, summary, text")

            unique_videos = len(set(r["video_id"] for r in simple_results if r.get("video_id")))
            logger.info(f"[SIMPLE SEARCH COMPLETE] {len(simple_results)} results from {unique_videos} videos (scanned {scanned} segments)")

            # ── LLM RERANKING for content-based simple filters ──
            # Rerank speaker/text/title/summary results with GPT-4o-mini
            # Skip structural filters (date/language/video) — they are exact matches
            use_simple_reranking = os.getenv("USE_SIMPLE_RERANKING", "true").lower() == "true"
            if use_simple_reranking and filter_type in ("speaker", "text", "title", "summary") and simple_results:
                if filter_type == "title":
                    logger.info("[SIMPLE SEARCH] Skipping LLM rerank for title filter to preserve exact-title scoring")
                    rerank_query = ""
                    simple_single_word = False
                else:
                    rerank_query = speaker_filter or title_filter or query_text or " ".join(words or [])
                    simple_single_word = bool(rerank_query and len(rerank_query.split()) == 1 and len(normalize_word(rerank_query)) >= 2)
                if simple_single_word:
                    logger.info("[SIMPLE SEARCH] Skipping LLM rerank for single-word query to keep scores stable")
                else:
                    if rerank_query.strip():
                        simple_results = rerank_simple_results(
                            query=rerank_query,
                            results=simple_results,
                            filter_type=filter_type,
                            top_k=top_k,
                        )
                        unique_videos = len(set(r["video_id"] for r in simple_results if r.get("video_id")))
                        logger.info(f"[SIMPLE SEARCH POST-RERANK] {len(simple_results)} results from {unique_videos} videos")

            return {
                "query": query_text,
                "words": words,
                "speaker_filter": speaker_filter,
                "collection": SEGMENTS_COLLECTION,
                "search_mode": "simple",
                "filter_type": filter_type,
                "total_speaker_hits": len(simple_results) if filter_type == "speaker" else 0,
                "total_semantic_hits": 0,
                "total_keyword_hits": 0,
                "total_title_hits": len(simple_results) if filter_type == "title" else 0,
                "total_text_hits": len(simple_results) if filter_type == "text" else 0,
                "total_exact_phrase_hits": 0,
                "returned": len(simple_results),
                "unique_videos": unique_videos,
                "filters_applied": {
                    "filter_type": filter_type,
                    "video_id": video_id_filter,
                    "speaker": speaker_filter,
                    "title": title_filter,
                    "language": language_filter,
                    "filter_year": filter_year,
                    "filter_month": filter_month,
                    "filter_date": filter_date,
                },
                "results": [
                    {
                        "id": r["id"],
                        "segment_ids": [r["id"]],
                        "match_count": 1,
                        "is_exact_phrase_match": False,
                        "is_multi_match": False,
                        "matched_terms": [],
                        "score": round(r["score"], 4),
                        "match_types": r.get("match_types", []),
                        "matched_field": r.get("matched_field", ""),
                        "fuzzy_score": round(r.get("fuzzy_score", 0), 4),
                        "matched_words_count": 0,
                        "video_id": r.get("video_id"),
                        "video_title": r.get("video_title", ""),
                        "speaker": r.get("speaker", ""),
                        "diarization_speaker": r.get("diarization_speaker", ""),
                        "start_time": r.get("start_time", 0),
                        "end_time": r.get("end_time", 0),
                        "duration": r.get("duration", 0),
                        "text": r.get("text", ""),
                        "text_length": r.get("text_length", 0),
                        "summary_en": r.get("summary_en", ""),
                        "youtube_url": r.get("youtube_url", ""),
                        "language": r.get("language", ""),
                        "created_at": r.get("created_at"),
                        "video_created_at": r.get("video_created_at", ""),
                        "llm_relevance_score": r.get("llm_relevance_score"),
                        "youtube_url_timestamped": f"{r.get('youtube_url', '')}?t={int(r.get('start_time', 0))}" if r.get('youtube_url') and r.get('start_time', 0) > 0 else r.get('youtube_url', ''),
                    }
                    for r in simple_results
                ]
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[SIMPLE SEARCH ERROR] {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC SEARCH MODE — full vector + LLM pipeline (existing behavior)
    # ═══════════════════════════════════════════════════════════════════════════

    # ── PERSON ALIAS EXPANSION ────────────────────────────────────────────────
    # Detect known person aliases in the query and speaker filter, then expand
    # so that "HNR", "naeem", "rehman", "hafiz" all find Hafiz Naeem Ur Rehman.
    alias_speaker_variants: List[str] = []       # Extra speaker names to search in parallel
    alias_extra_keywords: List[str] = []          # Extra keyword terms injected into keyword search
    alias_person_key: Optional[str] = None        # Which person was detected (if any)

    # 1. Check the explicit speaker filter first
    if speaker_filter:
        person_key = detect_person_alias(speaker_filter)
        if person_key:
            alias_person_key = person_key
            data_p = PERSON_ALIASES[person_key]
            alias_speaker_variants = data_p["speaker_variants"]
            alias_extra_keywords = [data_p["canonical"]] + data_p["speaker_variants"][:3]
            logger.info(f"Alias detected in speaker filter '{speaker_filter}' → expanding to {len(alias_speaker_variants)} variants for '{data_p['canonical']}'")
            # Replace speaker_filter with canonical name for cleaner matching
            speaker_filter = data_p["canonical"]

    # 2. Check the query text (but only if speaker filter didn't already trigger)
    if not alias_person_key and query_text:
        person_key = detect_person_alias(query_text)
        if person_key:
            alias_person_key = person_key
            data_p = PERSON_ALIASES[person_key]
            alias_speaker_variants = data_p["speaker_variants"]
            alias_extra_keywords = [data_p["canonical"]] + data_p["speaker_variants"][:3]
            logger.info(f"Alias detected in query '{query_text[:40]}' → expanding to person '{data_p['canonical']}'")
            # Inject canonical name into query and words list
            canonical = data_p["canonical"]
            if canonical.lower() not in query_text.lower():
                query_text = f"{query_text} {canonical}"
            for term in alias_extra_keywords:
                tw = term.strip()
                if tw and tw.lower() not in [w.lower() for w in words]:
                    words.append(tw)

    # 3. If no full-phrase match, do single-word check on query words
    # e.g. query "hafiz speech" triggers alias even if full phrase doesn't match
    if not alias_person_key and query_text:
        for qw in query_text.lower().split():
            person_key = detect_person_alias(qw)
            if person_key:
                alias_person_key = person_key
                data_p = PERSON_ALIASES[person_key]
                alias_speaker_variants = data_p["speaker_variants"]
                alias_extra_keywords = [data_p["canonical"]] + data_p["speaker_variants"][:3]
                logger.info(f"Alias detected via word '{qw}' in query → person '{data_p['canonical']}'")
                canonical = data_p["canonical"]
                if canonical.lower() not in query_text.lower():
                    query_text = f"{query_text} {canonical}"
                # Auto-set speaker_filter if none set (single-word alias implies speaker search)
                if not speaker_filter:
                    speaker_filter = data_p["canonical"]
                    logger.info(f"Auto-set speaker_filter='{speaker_filter}' from single-word alias '{qw}'")
                break
    # ─────────────────────────────────────────────────────────────────────────

    # Enable/disable advanced features via environment or request
    use_query_expansion = os.getenv("USE_QUERY_EXPANSION", "true").lower() == "true"
    use_reranking = os.getenv("USE_RERANKING", "true").lower() == "true"
    use_llm_understanding = os.getenv("USE_LLM_UNDERSTANDING", "true").lower() == "true"
    normalized_query = normalize_word(query_text) if query_text else ""
    single_word_query = bool(normalized_query and len(query_text.split()) == 1 and len(normalized_query) >= 2)
    # Always enable scan strategies for known person aliases — we need speaker/keyword/title search
    is_person_alias_query = alias_person_key is not None
    use_scan_strategies = (not openai_only_search) or single_word_query or is_person_alias_query

    if openai_only_search:
        # Keep semantic search responsive by constraining expensive fallbacks.
        use_query_expansion = False
        if is_person_alias_query:
            # Person searches need more scanning to find all speaker/title/text matches
            max_scanned = min(max_scanned, 50000)
            logger.info(f"[OPENAI-ONLY] Person alias detected ('{alias_person_key}'): enabling all scan strategies, max_scanned={max_scanned}")
        else:
            max_scanned = min(max_scanned, 12000 if single_word_query else 5000)
        if single_word_query:
            logger.info("[OPENAI-ONLY] Single-word query detected: enabling lightweight keyword fallback")
    
    # LLM QUERY UNDERSTANDING — parse intent, detect language, extract speaker
    query_intent = None
    if query_text and use_llm_understanding and openai_client:
        try:
            query_intent = understand_query(query_text)
            logger.info(f"LLM intent: {query_intent.get('query_type')}, lang={query_intent.get('detected_language')}, speaker={query_intent.get('extracted_speaker')}")
            
            # Auto-extract speaker from query ONLY if it's a speaker_search query type
            # Do NOT apply speaker filter for title_search, topic_search, etc.
            # NOTE: We store the speaker for PARALLEL speaker search, but DON'T use it as
            # an exclusive filter on semantic search - we want BOTH transcript mentions AND speaker segments
            if not speaker_filter and query_intent.get("extracted_speaker"):
                query_type = query_intent.get("query_type", "general")
                if query_type == "speaker_search":
                    # Store speaker for parallel search, but DON'T apply as exclusive filter
                    # This allows semantic search to find transcript mentions of the speaker
                    speaker_filter = query_intent["extracted_speaker"]
                    logger.info(f"Speaker search detected: '{speaker_filter}' - will search BOTH transcript content AND speaker fields")
                else:
                    logger.info(f"Skipping speaker filter (query_type={query_type}, not speaker_search)")
            
            # AUTO LANGUAGE FILTER: If query is English, only search English transcripts
            # This prevents showing Urdu videos when user searches in English
            # NOTE: Database stores ISO codes: "en", "ur", etc. (not "English", "Urdu")
            # EXCEPTION: Skip auto-filter for short title-like queries (may be video titles)
            def looks_like_title(q):
                """Check if query looks like a video title rather than a topic search"""
                if not q:
                    return False
                q = q.strip()
                words = q.split()
                # Very short queries (1-5 words) are likely titles
                if len(words) <= 5:
                    return True
                # Queries with unusual capitalization (proper nouns/titles)
                if any(w[0].isupper() for w in words if len(w) > 0):
                    # Has capital letters beyond just first word
                    caps = sum(1 for w in words if len(w) > 0 and w[0].isupper())
                    if caps >= 2:  # Multiple capitalized words = likely a title
                        return True
                return False
            
            if not language_filter and query_intent.get("detected_language") == "english":
                if is_person_alias_query:
                    logger.info(f"Skipping auto-language filter: person alias query ('{alias_person_key}') — need to search ALL languages")
                elif looks_like_title(query_text):
                    logger.info(f"Skipping auto-language filter: query looks like a title ('{query_text[:50]}')")
                elif query_intent.get("query_type") == "speaker_search":
                    logger.info(f"Skipping auto-language filter: speaker search — need to search ALL languages")
                else:
                    language_filter = "en"  # ISO code matching database value
                    logger.info(f"Auto-applied language filter: en (based on English query)")
        except Exception as e:
            logger.warning(f"LLM understanding failed, continuing without: {e}")
    
    # Speaker filter is ALWAYS used as a filter on the speaker/diarization_speaker fields.
    speaker_search_in_text = None  # Not used anymore
    
    # Handle backward compatibility for single word
    if word: 
        words = [word] if isinstance(word, str) else word
    
    # Parse query for embedded filters (e.g. "speaker:John title:intro AI")
    if query_text and not any([video_id_filter, speaker_filter, title_filter]):
        parsed = parse_search_query(query_text)
        if parsed["video_id"]:
            video_id_filter = parsed["video_id"]
        if parsed["speaker"]:
            speaker_filter = parsed["speaker"]
        if parsed["title"]:
            title_filter = parsed["title"]
        query_text = parsed["semantic_query"]
    
    # Safety net: if query is empty but words are provided, reconstruct query text
    # This ensures semantic search runs even when the frontend only sends words
    if not query_text and words:
        query_text = " ".join(words)
        logger.info(f"Reconstructed query from words: '{query_text[:120]}'")
    
    # Allow search with: query, words, title, OR speaker (speaker-only search is valid)
    if not query_text and not words and not title_filter and not speaker_filter:
        raise HTTPException(
            status_code=400,
            detail="Either 'query', 'words', 'title', or 'speaker' is required"
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
        end_time = time_range.get("end")
        if start_time is not None: 
            filter_conditions.append(
                FieldCondition(key="start_time", range=models.Range(gte=start_time))
            )
        if end_time is not None: 
            filter_conditions.append(
                FieldCondition(key="end_time", range=models.Range(lte=end_time))
            )

    search_filter = Filter(must=filter_conditions) if filter_conditions else None
    
    # Title search filter: NEVER apply language filter to title searches
    # This ensures users searching by title (e.g., "Hnr Book Ceremony") find
    # that specific video regardless of what language the transcript is in.
    # A video might have an English title but Urdu/Hindi transcript.
    
    title_filter_conditions = []
    if video_id_filter is not None:
        title_filter_conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=video_id_filter))
        )
    # NOTE: Deliberately NOT adding language_filter for title search
    # Video titles are often in English even when transcripts are in other languages
    # (e.g., "Hnr Book Ceremony" with Urdu transcript)
    logger.info(f"Title search: skipping language filter to search ALL languages")
    if time_range:
        start_t = time_range.get("start")
        end_t = time_range.get("end")
        if start_t is not None:
            title_filter_conditions.append(
                FieldCondition(key="start_time", range=models.Range(gte=start_t))
            )
        if end_t is not None:
            title_filter_conditions.append(
                FieldCondition(key="end_time", range=models.Range(lte=end_t))
            )
    title_search_filter = Filter(must=title_filter_conditions) if title_filter_conditions else None

    semantic_results = []
    keyword_results = []
    speaker_results = []  # New: results from speaker-only scroll search
    title_results = []    # Results from title fuzzy matching
    
    # Strategy 0: Speaker field search - find segments BY a specific speaker
    # This runs in TWO cases:
    # 1. Speaker-only search: no query/words, just finding segments by speaker
    # 2. Speaker search WITH query: run alongside semantic search to find BOTH:
    #    - Segments BY that speaker (speaker field match)
    #    - Segments MENTIONING that speaker (transcript content match via semantic search)
    speaker_only_search = speaker_filter and not query_text and not words
    speaker_parallel_search = speaker_filter and query_text  # Run speaker search in parallel with semantic
    
    if use_scan_strategies and (speaker_only_search or speaker_parallel_search):
        try:
            logger.info(f"Speaker field search for: '{speaker_filter}'")
            offset = None
            scanned = 0
            max_scan_speaker = min(max_scanned, 50000)
            
            # Build a more targeted scroll filter for speaker search
            # Use MatchText on speaker/diarization_speaker fields to pre-filter at Qdrant level
            # This avoids scanning ALL segments and only returns candidates with matching text
            speaker_scroll_conditions = list(filter_conditions)  # Start with existing filters (video_id, language, etc.)
            
            # Extract meaningful name parts for Qdrant text matching
            speaker_name_parts = [w.strip().lower() for w in speaker_filter.split() if len(w.strip()) >= 3]
            
            use_text_filter = False
            speaker_scroll_filter = search_filter  # Default fallback
            
            if speaker_name_parts:
                try:
                    # Use "should" (OR) filter: match any name part in either speaker or diarization_speaker field
                    speaker_text_conditions = []
                    for part in speaker_name_parts:
                        speaker_text_conditions.append(
                            FieldCondition(key="speaker", match=MatchText(text=part))
                        )
                        speaker_text_conditions.append(
                            FieldCondition(key="diarization_speaker", match=MatchText(text=part))
                        )
                    
                    # Combine: must match existing filters AND should match at least one speaker name part
                    candidate_filter = Filter(
                        must=speaker_scroll_conditions if speaker_scroll_conditions else None,
                        should=speaker_text_conditions
                    )
                    
                    # Test the filter with a small scroll to see if text indexes work
                    test_points, _ = qdrant_client.scroll(
                        collection_name=SEGMENTS_COLLECTION,
                        scroll_filter=candidate_filter,
                        limit=1,
                        with_payload=False,
                        with_vectors=False
                    )
                    speaker_scroll_filter = candidate_filter
                    use_text_filter = True
                    logger.info(f"Using MatchText filter for speaker search (text indexes available)")
                except Exception as filter_err:
                    logger.warning(f"MatchText filter failed (text indexes may not exist), falling back to full scan: {str(filter_err)}")
                    speaker_scroll_filter = search_filter
            
            while scanned < max_scan_speaker and len(speaker_results) < top_k:
                points, next_offset = qdrant_client.scroll(
                    collection_name=SEGMENTS_COLLECTION,
                    scroll_filter=speaker_scroll_filter,
                    limit=min(1000, max_scan_speaker - scanned),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                
                for p in points:
                    scanned += 1
                    payload = p.payload or {}
                    spk = payload.get("speaker", "")
                    diar_spk = payload.get("diarization_speaker", "")
                    speaker_combined = f"{spk} {diar_spk}".strip()
                    video_title = payload.get("video_title", "")
                    
                    # Fuzzy match speaker name against speaker fields
                    if not fuzzy_match_speaker(speaker_filter, speaker_combined, threshold=70):
                        continue
                    
                    # Apply title filter if present
                    if title_filter and not fuzzy_match_text(title_filter, video_title, threshold=60):
                        continue
                    
                    # Calculate score based on speaker match quality
                    match_score = fuzz.ratio(speaker_filter.lower(), speaker_combined.lower()) / 100
                    
                    speaker_results.append({
                        "id": p.id,
                        "score": round(max(match_score, 0.5), 4),  # Speaker matches get decent score
                        "video_id": payload.get("video_id"),
                        "video_title": video_title,
                        "speaker": spk,
                        "diarization_speaker": diar_spk,
                        "start_time": payload.get("start_time", 0),
                        "end_time": payload.get("end_time", 0),
                        "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                        "text": payload.get("text", ""),
                        "text_length": payload.get("text_length", 0),
                        "youtube_url": payload.get("youtube_url", ""),
                        "language": payload.get("language", ""),
                        "created_at": payload.get("created_at"),
                        "match_types": ["speaker_filter"],
                        "fuzzy_score": round(match_score, 2),
                        "matched_field": "speaker",
                    })
                
                if len(speaker_results) >= top_k:
                    break
                    
                offset = next_offset
                if not next_offset:
                    break
            
            logger.info(f"Speaker-only search found {len(speaker_results)} results (scanned {scanned})")
            
        except Exception as e:
            logger.error(f"ERROR during speaker-only search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Speaker search error: {str(e)}")

    # Strategy 0.5: EXACT PHRASE SEARCH - Find segments containing the exact query text
    # This runs BEFORE semantic search to prioritize exact transcript matches
    # When user searches "before Windows 95, 1984", we find segments with that exact text
    exact_phrase_results = []
    if use_scan_strategies and query_text and len(query_text.strip()) >= 5:
        try:
            exact_phrase = normalize_for_matching(query_text.strip())
            logger.info(f"Exact phrase search for: '{exact_phrase[:60]}'")
            
            offset = None
            scanned = 0
            max_scan_exact = min(max_scanned, 20000)  # Limit exact phrase scan
            
            while scanned < max_scan_exact and len(exact_phrase_results) < top_k:
                points, next_offset = qdrant_client.scroll(
                    collection_name=SEGMENTS_COLLECTION,
                    scroll_filter=search_filter,  # Apply video/language filters
                    limit=min(1000, max_scan_exact - scanned),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                
                for p in points:
                    scanned += 1
                    payload = p.payload or {}
                    text = normalize_for_matching(payload.get("text") or "")
                    
                    # Check if the exact phrase appears as whole words in the transcript
                    # Uses word boundaries to prevent 'dance' matching 'abundance' etc.
                    if whole_phrase_match(exact_phrase, text):
                        spk = payload.get("speaker", "")
                        diar_spk = payload.get("diarization_speaker", "")
                        video_title = payload.get("video_title", "")
                        
                        # Apply speaker filter if present
                        if speaker_filter:
                            speaker_combined = f"{spk} {diar_spk}".strip()
                            if not fuzzy_match_speaker(speaker_filter, speaker_combined, threshold=70):
                                continue
                        
                        exact_phrase_results.append({
                            "id": p.id,
                            "score": 0.99,  # Highest score for exact phrase match
                            "video_id": payload.get("video_id"),
                            "video_title": video_title,
                            "speaker": spk,
                            "diarization_speaker": diar_spk,
                            "start_time": payload.get("start_time", 0),
                            "end_time": payload.get("end_time", 0),
                            "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                            "text": payload.get("text", ""),
                            "text_length": payload.get("text_length", 0),
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "match_types": ["exact_phrase_match"],
                            "fuzzy_score": 1.0,
                            "matched_field": "text",
                            "matched_terms": [{"term": query_text.strip(), "field": "text", "score": 1.0, "type": "exact_phrase"}],
                            "matched_words_count": len(query_text.split()),
                            "is_multi_match": False,
                            "exact_phrase": True
                        })
                
                if len(exact_phrase_results) >= top_k:
                    break
                
                offset = next_offset
                if not next_offset:
                    break
            
            logger.info(f"Exact phrase search found {len(exact_phrase_results)} results (scanned {scanned})")
            
        except Exception as e:
            logger.warning(f"Exact phrase search error (non-fatal): {e}")

    # Strategy 1: Semantic search with optional query expansion
    if query_text:
        try:
            logger.info(f"Semantic search for: '{query_text[:120]}'")
            
            # Build query variations using LLM understanding
            # ALWAYS search with the ORIGINAL query first — it's closest to what the user typed
            query_variations = [query_text]
            
            if query_intent:
                # If LLM rewrote the semantic_query, add it as a variation (not replacement)
                llm_query = query_intent.get("semantic_query", "")
                if llm_query and llm_query.strip().lower() != query_text.strip().lower():
                    query_variations.append(llm_query)
                    logger.info(f"Added LLM semantic query: {llm_query[:80]}")
                
                # Add translated query for cross-language search
                translated = query_intent.get("semantic_query_translated", "")
                if translated and translated.strip() and translated.strip().lower() != query_text.strip().lower():
                    query_variations.append(translated)
                    logger.info(f"Added cross-language query: {translated[:80]}")
                
                # Add expanded terms as additional search queries
                expanded_terms = query_intent.get("expanded_terms", [])
                if expanded_terms:
                    # Join expanded terms into a single query phrase (more efficient)
                    expansion_query = " ".join(expanded_terms[:3])
                    if expansion_query.strip() and expansion_query != query_text:
                        query_variations.append(expansion_query)
                        logger.info(f"Added expansion query: {expansion_query[:80]}")
            elif use_query_expansion and openai_client and len(query_text.split()) >= 3:
                # Fallback to old-style expansion if LLM understanding not available
                try:
                    expanded = expand_query_with_gpt(query_text)
                    query_variations = expanded[:3]
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")
            
            # Limit query variations for speed - original query is most important
            # For short queries (1-2 words), SKIP expanded terms entirely — they cause
            # semantic drift (e.g. "mufti" → expanded to "religious scholar" → pulls in
            # unrelated religious content like "qari mansoor jamat")
            query_word_count = len(query_text.split()) if query_text else 0
            if query_word_count <= 2:
                # For short queries, only use original + translated (no expanded terms)
                translated = query_intent.get("semantic_query_translated", "") if query_intent else ""
                filtered = []
                for v in query_variations:
                    if v == query_text:
                        filtered.append(v)
                    elif translated and v == translated:
                        filtered.append(v)
                query_variations = filtered if filtered else [query_text]
                logger.info(f"Short query ({query_word_count} words): limited to {len(query_variations)} variations (no expansion)")
            else:
                query_variations = query_variations[:2]  # Max 2 variations for speed
            logger.info(f"Total query variations (capped): {len(query_variations)}")
            
            # Search with all query variations
            all_semantic_results = {}  # Use dict to deduplicate by ID
            
            for idx, q_var in enumerate(query_variations):
                # Use cached embedding for repeated queries
                query_vector = get_cached_embedding(q_var)
                
                # Adjust search parameters - balance speed vs recall
                search_params = SearchParams(
                    hnsw_ef=128 if single_word_query else 64,
                    exact=True if single_word_query else False
                )

                # Use a higher threshold for Qdrant retrieval to avoid irrelevant results.
                # We want to retrieve results that have at least some semantic similarity.
                # For queries like "bill gates" in a database without Bill Gates content,
                # this prevents retrieving completely unrelated Pakistan politics videos.
                retrieval_threshold = max(min_score * 0.85, 0.35)  # e.g. 0.35 * 0.85 = 0.30 (raised from 0.26)
                
                sem_search_response = qdrant_client.query_points(
                    collection_name=SEGMENTS_COLLECTION,
                    query=query_vector,
                    limit=top_k * 2 if idx == 0 else top_k,  # Reduced from 3x for speed
                    score_threshold=retrieval_threshold,  # Lower threshold — filter later
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params
                )
                sem_search_results = sem_search_response.points
                
                # Process results from this query variation
                for r in sem_search_results:
                    result_id = r.id
                    
                    # Skip if already processed with better score
                    if result_id in all_semantic_results:
                        # Keep the better score
                        if float(getattr(r, "score", 0.0)) > all_semantic_results[result_id].get("base_score", 0):
                            pass  # Update below
                        else:
                            continue
                    
                    payload = r.payload or {}
                    video_title = payload.get("video_title", "")
                    speaker = payload.get("speaker", "")
                    diarization_speaker = payload.get("diarization_speaker", "")
                    text = payload.get("text", "")
                    
                    # ELASTIC title filter with fuzzy matching
                    if title_filter:
                        if not fuzzy_match_text(title_filter, video_title, threshold=60):
                            continue
                    
                    # ELASTIC speaker filter with fuzzy matching (checks both speaker fields)
                    # For speaker_parallel_search: DON'T filter - we want transcript mentions too
                    # The speaker field matching is handled separately by Strategy 0
                    speaker_field_match = False
                    if speaker_filter:
                        speaker_combined = f"{speaker} {diarization_speaker}"
                        speaker_field_match = fuzzy_match_speaker(speaker_filter, speaker_combined, threshold=70)
                        
                        # Only filter by speaker if NOT a parallel search (i.e., explicit speaker-only filter from user)
                        # For speaker searches with query, we want BOTH speaker-tagged AND transcript mentions
                        if not speaker_parallel_search and not speaker_field_match:
                            continue
                    
                    # Calculate fuzzy boost score
                    fuzzy_boost = 0
                    if title_filter and video_title:
                        fuzzy_boost = max(fuzzy_boost, fuzz.partial_ratio(title_filter.lower(), video_title.lower()) / 100)
                    if speaker_filter:
                        speaker_combined = f"{speaker} {diarization_speaker}"
                        if speaker or diarization_speaker:
                            fuzzy_boost = max(fuzzy_boost, fuzz.ratio(speaker_filter.lower(), speaker_combined.lower()) / 100)
                    
                    base_score = float(getattr(r, "score", 0.0))
                    
                    # ── QUERY-PRESENCE VERIFICATION ──
                    # For short queries (1-2 words like "mufti"), semantic search can return
                    # loosely related results from the same domain (e.g. "qari", "jamat").
                    # We verify if the actual query words appear in the result text/metadata
                    # and penalize results where they don't.
                    query_words_in_result = False
                    if query_text:
                        text_lower = (text or "").lower()
                        title_lower = (video_title or "").lower()
                        speaker_lower = f"{speaker} {diarization_speaker}".lower()
                        summary_lower = (payload.get("summary_en") or "").lower()
                        all_text = f"{text_lower} {title_lower} {speaker_lower} {summary_lower}"
                        
                        # Check if ANY query word appears in the result (whole word or safe variant)
                        query_words = [w.lower().strip() for w in query_text.split() if len(w.strip()) >= 3]
                        for qw in query_words:
                            if whole_word_match(qw, all_text) or word_variant_match(qw, all_text):
                                query_words_in_result = True
                                break
                    
                    # Apply query-presence penalty for SHORT queries (1-2 words)
                    # Short queries are specific searches ("mufti", "drone", "bill gates") — user expects
                    # the word to actually appear in results, not just be "semantically related"
                    # EXCEPTION: Skip penalty for known person alias queries — we want ALL results about them
                    query_word_count = len(query_text.split()) if query_text else 0
                    presence_penalty = 0.0
                    if is_person_alias_query:
                        # Person alias queries: no presence penalty — semantic similarity is enough
                        # We want to find ALL content related to this person even if their name
                        # isn't literally in the transcript text
                        pass
                    elif query_word_count <= 2 and not query_words_in_result and not speaker_field_match:
                        # STRONGER penalty: results that don't contain the actual search term
                        # This prevents "bill gates" → "pakistan politics" false matches
                        # Also prevents "mufti" → "qari mansoor jamat" false matches
                        presence_penalty = 0.25  # Increased from 0.15 to 0.25
                    elif 3 <= query_word_count <= 4 and not query_words_in_result and not speaker_field_match:
                        # Medium queries (3-4 words): smaller penalty
                        presence_penalty = 0.15

                    # Reward close lexical matches (including typo/near-word matches)
                    # so that words near the query get visible score lift.
                    lexical_closeness = 0.0
                    if query_text:
                        all_text = f"{text or ''} {video_title or ''} {speaker or ''} {diarization_speaker or ''} {payload.get('summary_en') or ''}"
                        lexical_query_words = [w for w in query_text.split() if len(normalize_word(w)) >= 3]
                        if lexical_query_words:
                            close_threshold = 78 if len(lexical_query_words) <= 2 else 82
                            lexical_closeness = query_closeness_score(lexical_query_words, all_text, threshold=close_threshold)
                    
                    combined_score = calculate_combined_score(base_score, False, fuzzy_boost)
                    if lexical_closeness > 0:
                        combined_score += min(lexical_closeness * 0.12, 0.12)
                    combined_score = max(combined_score - presence_penalty, 0.0)
                    
                    # Results from expanded/translated query variations are lower confidence
                    # They are more likely to drift from user intent
                    if idx > 0:
                        combined_score *= 0.85  # 15% discount for non-original queries
                    
                    # Apply min_score filter AFTER combined scoring
                    if combined_score < min_score:
                        continue
                    
                    # Build match types list
                    match_types = ["semantic"]
                    if speaker_field_match:
                        match_types.append("speaker_field_match")
                    if query_words_in_result:
                        match_types.append("query_term_present")
                    if lexical_closeness >= 0.65:
                        match_types.append("query_close_match")
                    
                    all_semantic_results[result_id] = {
                        "id": r.id,
                        "score": combined_score,
                        "base_score": base_score,
                        "video_id": payload.get("video_id"),
                        "video_title": video_title,
                        "speaker": speaker,
                        "diarization_speaker": diarization_speaker,
                        "start_time": payload.get("start_time", 0),
                        "end_time": payload.get("end_time", 0),
                        "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                        "text": text,
                        "text_length": payload.get("text_length", 0),
                        "youtube_url": payload.get("youtube_url", ""),
                        "language": payload.get("language", ""),
                        "created_at": payload.get("created_at"),
                        "match_types": match_types,
                        "fuzzy_score": round(fuzzy_boost, 2),
                        "lexical_closeness": round(lexical_closeness, 3),
                        "matched_field": "speaker" if speaker_field_match else "semantic_vector",
                        "query_variation": q_var if idx > 0 else "original"
                    }
            
            # Convert dict to list
            semantic_results = list(all_semantic_results.values())
            logger.info(f"Found {len(semantic_results)} unique semantic results from {len(query_variations)} query variations")

        except Exception as e:
            logger.info(f"ERROR during semantic search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")

    # Strategy 2: Keyword search
    # For single-word queries, also use the query itself as a keyword fallback.
    search_words = list(words) if words else []
    if query_text and len(query_text.split()) == 1:
        search_words.append(query_text)
    
    # Clean up search words: normalize quotes/dashes, strip punctuation, remove too-short words and stop words
    search_words = sorted(set(
        normalize_word(w)
        for w in search_words
        if len(normalize_word(w)) >= 2 and normalize_word(w).lower() not in STOP_WORDS
    ))
    logger.info(f"Keywords after stop word filtering: {len(search_words)} words")
    
    # EXACT PHRASE MATCHING: Prepare the full original query for exact substring match
    # This ensures that when user searches for "before Windows 95, 1984" and that exact
    # phrase exists in a transcript, it gets the highest score (0.99)
    exact_phrase_query = None
    if query_text and len(query_text) >= 5:
        # Normalize the full query for exact phrase comparison
        exact_phrase_query = normalize_for_matching(query_text.strip())
    
    if use_scan_strategies and search_words:
        try:
            # Cap keyword search to avoid timeout - keyword matching is fast but needs limit
            max_scan_keyword = min(max_scanned, 3000)  # Reduced from 5000 for speed
            logger.info(f"Elastic keyword search for: {search_words[:10]} (max_scanned={max_scan_keyword})")
            words_lower = search_words  # already lowercased by normalize_word
            page_size = 500  # Smaller batches for faster response
            scanned = 0
            offset = None

            scroll_filter = search_filter

            # For single-word searches, pre-filter with Qdrant text index so matches are
            # retrieved directly instead of relying on early scroll windows.
            if len(words_lower) == 1:
                kw = words_lower[0]
                keyword_text_conditions = [
                    FieldCondition(key="text", match=MatchText(text=kw)),
                    FieldCondition(key="summary_en", match=MatchText(text=kw)),
                    FieldCondition(key="video_title", match=MatchText(text=kw)),
                    FieldCondition(key="speaker", match=MatchText(text=kw)),
                    FieldCondition(key="diarization_speaker", match=MatchText(text=kw)),
                ]
                if search_filter and getattr(search_filter, "must", None):
                    scroll_filter = Filter(must=list(search_filter.must), should=keyword_text_conditions)
                else:
                    scroll_filter = Filter(should=keyword_text_conditions)
                logger.info(f"Keyword search using MatchText prefilter for single word: '{kw}'")

            # Early termination when we have enough results
            target_results = max(top_k, 20)
            while scanned < max_scan_keyword and len(keyword_results) < target_results:
                points, next_offset = qdrant_client.scroll(
                    collection_name=SEGMENTS_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=min(page_size, max_scan_keyword - scanned),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not points:
                    break

                for p in points:
                    scanned += 1
                    payload = p.payload or {}
                    # Normalize text fields: lowercase + normalize curly quotes/dashes
                    text = normalize_for_matching(payload.get("text") or "")
                    speaker_field = (payload.get("speaker") or "")
                    video_title = payload.get("video_title", "")
                    diarization_speaker = (payload.get("diarization_speaker") or "")
                    video_filename = (payload.get("video_filename") or "")
                    
                    # ========== EXACT PHRASE MATCH CHECK (HIGHEST PRIORITY) ==========
                    # If the user's full query appears as-is in the transcript, give it top score
                    # This ensures "before Windows 95, 1984" matches exactly and ranks first
                    is_exact_phrase_match = False
                    if exact_phrase_query and len(exact_phrase_query) >= 5:
                        if whole_phrase_match(exact_phrase_query, text):
                            is_exact_phrase_match = True
                            # Add as exact match with highest score - this goes to the TOP
                            keyword_results.append({
                                "id": p.id,
                                "score": 0.99,  # Near-perfect score for exact phrase match
                                "video_id": payload.get("video_id"),
                                "video_title": video_title,
                                "speaker": speaker_field,
                                "diarization_speaker": diarization_speaker,
                                "start_time": payload.get("start_time", 0),
                                "end_time": payload.get("end_time", 0),
                                "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                                "text": payload.get("text", ""),
                                "text_length": payload.get("text_length", 0),
                                "youtube_url": payload.get("youtube_url", ""),
                                "language": payload.get("language", ""),
                                "created_at": payload.get("created_at"),
                                "match_types": ["keyword", "exact_phrase_match"],
                                "fuzzy_score": 1.0,
                                "matched_field": "text",
                                "matched_words_count": len(search_words),
                                "matched_terms": [{"term": query_text.strip(), "field": "text", "score": 1.0, "type": "exact_phrase"}],
                                "is_multi_match": False,  # Single exact phrase match
                                "field_weight": 3.0,  # High weight for exact phrase
                                "exact_phrase": True
                            })
                            continue  # Skip word-by-word matching for this segment
                    # ========== END EXACT PHRASE MATCH ==========
                    
                    # Track matched words and their scores
                    matched_words_count = 0
                    close_words_count = 0.0
                    best_fuzzy_score = 0
                    matched_field = ""
                    matched_terms = []  # Track exactly which terms matched
                    field_weights = {
                        "video_title": 2.5,      # Highest priority: title matches
                        "speaker": 1.8,          # High priority: speaker matches  
                        "diarization_speaker": 1.8,
                        "text": 1.0,             # Normal priority: text content
                        "video_filename": 0.8    # Lower priority: filename
                    }
                    best_field_weight = 0
                    
                    # STRICT keyword matching: check each word across fields
                    # Both search words AND field values are normalized (lowercase + straight quotes)
                    for w in words_lower:
                        word_found = False
                        
                        # Check each field individually to track which field matched
                        fields_to_check = [
                            ("text", text),  # already normalized above
                            ("speaker", normalize_for_matching(speaker_field)),
                            ("diarization_speaker", normalize_for_matching(diarization_speaker)),
                            ("video_title", normalize_for_matching(video_title)),
                        ]
                        
                        for field_name, field_value in fields_to_check:
                            if not field_value:
                                continue
                            
                            field_match_score = 0
                            
                            # Whole-word match (prevents 'tan' matching 'pakistan', 'dance' matching 'abundance')
                            # Both sides are normalized so "there've" matches "there've"
                            if whole_word_match(w, field_value):
                                field_match_score = 1.0
                            # Variant match for words like fail -> failed/failing/failure
                            elif word_variant_match(w, field_value):
                                field_match_score = 0.88
                            # Fuzzy match ONLY for longer words (4+ chars) against individual words
                            # Uses word-to-word comparison, NOT substring matching
                            elif len(w) >= 4:
                                fw_score = fuzzy_word_match(w, field_value, threshold=80)
                                if fw_score > 0:
                                    field_match_score = fw_score
                                elif len(w) >= 5:
                                    near_score = fuzzy_word_match(w, field_value, threshold=76)
                                    if near_score > 0:
                                        field_match_score = near_score * 0.65
                                        close_words_count += 1
                            
                            if field_match_score > 0:
                                word_found = True
                                matched_words_count += 1
                                matched_terms.append({"term": w, "field": field_name, "score": round(field_match_score, 2)})
                                
                                # Track best matching field and score
                                field_weight = field_weights.get(field_name, 1.0)
                                weighted_score = field_match_score * field_weight
                                
                                if weighted_score > best_fuzzy_score:
                                    best_fuzzy_score = weighted_score
                                    matched_field = field_name
                                    best_field_weight = field_weight
                                
                                break  # Found match for this word, move to next word
                    
                    if matched_words_count == 0 and close_words_count == 0:
                        continue
                    
                    # Flexible word matching: require a high proportion but not necessarily ALL words
                    # For short queries (2-3 words): require ALL words
                    # For medium queries (4-6 words): allow 1 missing word
                    # For long queries (7+ words): require at least 75% of words
                    total_words = len(words_lower)
                    if total_words <= 3:
                        min_required = total_words  # ALL must match
                    elif total_words <= 6:
                        min_required = total_words - 1  # Allow 1 miss
                    else:
                        min_required = max(3, int(total_words * 0.75))  # 75% must match
                    
                    if matched_words_count < min_required:
                        continue
                    
                    # ELASTIC title filter with fuzzy matching
                    if title_filter:
                        if not fuzzy_match_text(title_filter, video_title, threshold=60):
                            continue
                    
                    # ELASTIC speaker filter with fuzzy matching
                    # For person alias queries with parallel search, DON'T filter keywords by speaker
                    # We want to find segments MENTIONING the person, not just spoken BY them
                    if speaker_filter and not (is_person_alias_query and speaker_parallel_search):
                        speaker_check = f"{speaker_field} {diarization_speaker}"
                        if not fuzzy_match_speaker(speaker_filter, speaker_check, threshold=70):
                            continue
                    
                    # Calculate intelligent score based on:
                    # 1. Field weight (title > speaker > text)
                    # 2. Match quality (exact vs fuzzy)
                    # 3. Number of words matched
                    # 4. Proportion of search query matched
                    
                    # Calculate score based on match quality
                    word_coverage = (matched_words_count + (close_words_count * 0.6)) / max(len(words_lower), 1)
                    match_quality = best_fuzzy_score / max(best_field_weight, 1.0)
                    
                    # Score formula: quality * coverage — higher baseline
                    final_score = match_quality * word_coverage * 0.80
                    
                    # Boost for exact text matches
                    if match_quality >= 1.0:
                        final_score += 0.12
                    
                    # Strong boost for title matches — title matches are highly relevant
                    if matched_field == "video_title":
                        final_score = min(final_score + 0.20, 0.98)
                    
                    # Boost for speaker field matches
                    if matched_field in ("speaker", "diarization_speaker"):
                        final_score = min(final_score + 0.10, 0.95)
                    
                    # Must meet minimum score threshold
                    if final_score < min_score:
                        continue
                    
                    keyword_results.append({
                        "id": p.id,
                        "score": min(final_score, 1.0),
                        "video_id": payload.get("video_id"),
                        "video_title": video_title,
                        "speaker": speaker_field,
                        "diarization_speaker": diarization_speaker,
                        "start_time": payload.get("start_time", 0),
                        "end_time": payload.get("end_time", 0),
                        "duration": round((payload.get("end_time", 0) - payload.get("start_time", 0)), 2),
                        "text": payload.get("text", ""),
                        "text_length": payload.get("text_length", 0),
                        "youtube_url": payload.get("youtube_url", ""),
                        "language": payload.get("language", ""),
                        "created_at": payload.get("created_at"),
                        "match_types": ["keyword", f"matched_in_{matched_field}"],
                        "fuzzy_score": round(best_fuzzy_score / max(best_field_weight, 1.0), 2),
                        "matched_field": matched_field,
                        "matched_words_count": matched_words_count,
                        "matched_terms": matched_terms,  # List of {term, field, score} for each matched term
                        "is_multi_match": len(matched_terms) > 1,  # True if multiple terms matched
                        "field_weight": best_field_weight
                    })

                if len(keyword_results) >= top_k * 2:
                    break

                offset = next_offset
                if not next_offset:
                    break

        except Exception as e:
            logger.info(f"ERROR during keyword search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Keyword search error: {str(e)}")

    # Strategy 3: Title matching — search video titles for the query
    # Runs when either:
    # 1. User provides explicit title_filter (e.g., title:2023)
    # 2. User searches with query_text (automatic title search with lower priority)
    # This ensures searching "2023" finds videos with "2023" in the title
    title_query = title_filter or query_text or ""
    if use_scan_strategies and title_query and len(title_query.strip()) >= 2:  # Reduced from 3 to 2 for short queries like "2023"
        try:
            logger.info(f"Title matching search for: '{title_query[:120]}'")
            
            # Also check the translated query from LLM understanding
            title_queries = [title_query]
            if query_intent:
                translated = query_intent.get("semantic_query_translated", "")
                if translated and translated.strip().lower() != title_query.strip().lower():
                    title_queries.append(translated)
            
            # For person alias queries, add canonical name and key variants as title search terms
            if is_person_alias_query:
                person_data = PERSON_ALIASES[alias_person_key]
                canonical = person_data["canonical"]
                if canonical.lower() not in title_query.lower():
                    title_queries.append(canonical)
                # Add short forms that commonly appear in video titles
                for variant in person_data["speaker_variants"][:4]:
                    if variant.lower() not in [tq.lower() for tq in title_queries]:
                        title_queries.append(variant)
                logger.info(f"Person alias: expanded title queries to {len(title_queries)} variants")
            
            # Collect video_ids already in semantic/keyword results to avoid duplicating low-value segments
            existing_ids = set()
            for r in semantic_results:
                existing_ids.add(r["id"])
            for r in keyword_results:
                existing_ids.add(r["id"])
            for r in speaker_results:
                existing_ids.add(r["id"])
            
            # Track which videos matched by title to collect their segments
            matched_video_ids = set()
            matched_video_scores = {}  # video_id -> best title match score
            
            offset = None
            scanned = 0
            max_scan_title = min(max_scanned, 50000)
            seen_titles = {}  # video_id -> video_title (to avoid re-checking same video)
            
            while scanned < max_scan_title:
                points, next_offset = qdrant_client.scroll(
                    collection_name=SEGMENTS_COLLECTION,
                    scroll_filter=title_search_filter,  # Use title-specific filter (no language restriction for universal queries)
                    limit=min(1000, max_scan_title - scanned),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                
                for p in points:
                    scanned += 1
                    payload = p.payload or {}
                    vid = payload.get("video_id")
                    video_title = payload.get("video_title", "")
                    
                    # Skip if we've already determined this video's title match status
                    if vid in seen_titles:
                        # Already processed this video - skip (title match = 1 entry per video)
                        continue
                    
                    # First time seeing this video — check title match
                    seen_titles[vid] = video_title
                    
                    if not video_title:
                        continue
                    
                    # Check each title query variation
                    best_title_score = 0
                    has_real_overlap = False  # MUST have actual string overlap to be valid
                    
                    for tq in title_queries:
                        tq_lower = tq.lower().strip()
                        vt_lower = video_title.lower().strip()
                        
                        # Extract query words for overlap checking
                        # Use >= 4 chars to avoid short words like "ur", "is", "to"
                        # matching as substrings of unrelated title words ("future", etc.)
                        query_words_exact = [w for w in tq_lower.split() if len(w) >= 3]  # For exact match
                        query_words_substr = [w for w in tq_lower.split() if len(w) >= 4]  # For substring match
                        title_words_exact = [w for w in vt_lower.split() if len(w) >= 3]
                        title_words_substr = [w for w in vt_lower.split() if len(w) >= 4]
                        
                        # === STEP 1: Check for ACTUAL string overlap ===
                        # This prevents fuzzy algorithms from matching unrelated strings
                        overlap_found = False
                        overlap_score = 0
                        
                        # Check if full query is substring of title or vice versa
                        if tq_lower in vt_lower or vt_lower in tq_lower:
                            overlap_found = True
                            overlap_score = 0.95
                        else:
                            # Check word-level overlap (more reliable than fuzzy for title matching)
                            for qw in query_words_exact:
                                # Exact word match (both words >= 3 chars)
                                if qw in title_words_exact:
                                    overlap_found = True
                                    overlap_score = max(overlap_score, 0.85)
                                    break
                            if not overlap_found:
                                for qw in query_words_substr:
                                    # Query word (>= 4 chars) is substring of title word (>= 4 chars)
                                    # but only if it covers a significant portion (>= 60%) of the target word
                                    # e.g., "rehma" in "rehmaa" (5/6=83%) — OK
                                    # e.g., "mark" in "bookmark" (4/8=50%) — rejected
                                    if any(qw in tw and len(qw) / len(tw) >= 0.6 for tw in title_words_substr):
                                        overlap_found = True
                                        overlap_score = max(overlap_score, 0.80)
                                        break
                                    # Title word (>= 4 chars) is substring of query word (>= 4 chars)
                                    # Same coverage ratio requirement
                                    elif any(tw in qw and len(tw) / len(qw) >= 0.6 for tw in title_words_substr):
                                        overlap_found = True
                                        overlap_score = max(overlap_score, 0.75)
                                        break
                        
                        if overlap_found:
                            has_real_overlap = True
                            best_title_score = max(best_title_score, overlap_score)
                        else:
                            # === STEP 2: Fuzzy matching ONLY if overlap exists somewhere ===
                            # Use very strict fuzzy matching - only for handling typos
                            ratio = fuzz.ratio(tq_lower, vt_lower) / 100
                            
                            # Only accept fuzzy match if ratio is VERY high (>85%) 
                            # indicating actual similarity, not random matching
                            if ratio >= 0.85:
                                has_real_overlap = True
                                best_title_score = max(best_title_score, ratio * 0.9)  # Discount fuzzy score
                    
                    # CRITICAL: Skip if no real string overlap detected
                    # This prevents "rehmaa" from matching "Elon Musk" due to fuzzy algo quirks
                    if not has_real_overlap:
                        continue
                    
                    # Threshold for title match — 50% for longer queries, 65% for short ones (lowered for better recall)
                    threshold = 0.50 if len(title_query) >= 8 else 0.65
                    
                    # Log near-matches for debugging
                    if best_title_score >= 0.3:
                        logger.info(f"Title check: query='{title_query[:50]}' vs '{video_title[:50]}' => score={best_title_score:.2f} (threshold={threshold})")
                    
                    if best_title_score >= threshold:
                        matched_video_ids.add(vid)
                        matched_video_scores[vid] = min(best_title_score, 0.98)
                        
                        # For title matches, add ONE video-level entry (no segment details)
                        # Use this segment's metadata for video info only
                        title_results.append({
                            "id": f"title_match_{vid}",  # Special ID for title-only match
                            "score": round(matched_video_scores[vid], 4),
                            "video_id": vid,
                            "video_title": video_title,
                            "speaker": "",  # No specific speaker for video-level match
                            "diarization_speaker": "",
                            "start_time": 0,  # Represents entire video
                            "end_time": 0,
                            "duration": 0,
                            "text": "",  # No segment text - this is a title match
                            "text_length": 0,
                            "youtube_url": payload.get("youtube_url", ""),
                            "language": payload.get("language", ""),
                            "created_at": payload.get("created_at"),
                            "match_types": ["title_match"],
                            "fuzzy_score": round(best_title_score, 2),
                            "matched_field": "video_title",
                            "is_video_only": True,  # Flag to indicate this is video-level, not segment
                        })
                
                # Stop early if we found enough title-matched videos
                if len(title_results) >= top_k:
                    break
                    
                offset = next_offset
                if not next_offset:
                    break
            
            logger.info(f"Title matching found {len(title_results)} segments from {len(matched_video_ids)} videos (scanned {scanned} points, checked {len(seen_titles)} unique videos)")
            
        except Exception as e:
            logger.warning(f"Title matching search error (non-fatal): {e}")

    # Merge results - combine exact phrase, semantic, keyword, speaker, and title matches
    # EXACT PHRASE MATCHES GO FIRST - they have highest priority (score 0.99)
    # Track matched_terms from all sources to detect multi-match within same segment
    merged = {}
    
    def merge_matched_terms(existing_terms, new_terms):
        """Merge matched_terms lists, avoiding duplicates"""
        if not existing_terms:
            existing_terms = []
        if not new_terms:
            return existing_terms
        # Track existing term strings to avoid duplicates
        existing_set = {t.get("term", "") for t in existing_terms}
        for t in new_terms:
            if t.get("term", "") not in existing_set:
                existing_terms.append(t)
                existing_set.add(t.get("term", ""))
        return existing_terms
    
    # 1. Add exact phrase matches FIRST - they should always be on top
    for r in exact_phrase_results:
        r.setdefault("matched_terms", [])
        r.setdefault("is_multi_match", False)
        merged[r["id"]] = r
    
    # 2. Add speaker results
    for r in speaker_results:
        if r["id"] in merged:
            if "speaker_filter" not in merged[r["id"]]["match_types"]:
                merged[r["id"]]["match_types"].append("speaker_filter")
            # Merge matched_terms if speaker result has any
            merged[r["id"]]["matched_terms"] = merge_matched_terms(
                merged[r["id"]].get("matched_terms", []),
                r.get("matched_terms", [])
            )
            # Don't override exact phrase score (0.99)
        else:
            r.setdefault("matched_terms", [])
            r.setdefault("is_multi_match", False)
            merged[r["id"]] = r
    
    # 3. Add title results
    for r in title_results:
        if r["id"] in merged:
            if "title_match" not in merged[r["id"]]["match_types"]:
                merged[r["id"]]["match_types"].append("title_match")
            merged[r["id"]]["matched_terms"] = merge_matched_terms(
                merged[r["id"]].get("matched_terms", []),
                r.get("matched_terms", [])
            )
            merged[r["id"]]["score"] = max(merged[r["id"]]["score"], r["score"])
        else:
            r.setdefault("matched_terms", [])
            r.setdefault("is_multi_match", False)
            merged[r["id"]] = r
    
    for r in semantic_results:
        if r["id"] in merged:
            merged[r["id"]]["match_types"].append("semantic")
            merged[r["id"]]["matched_terms"] = merge_matched_terms(
                merged[r["id"]].get("matched_terms", []),
                r.get("matched_terms", [])
            )
            merged[r["id"]]["score"] = max(merged[r["id"]]["score"], r["score"])
        else:
            r.setdefault("matched_terms", [])
            r.setdefault("is_multi_match", False)
            merged[r["id"]] = r

    for r in keyword_results:
        if r["id"] in merged:
            if "keyword" not in merged[r["id"]]["match_types"]:
                merged[r["id"]]["match_types"].append("keyword")
            merged[r["id"]]["matched_terms"] = merge_matched_terms(
                merged[r["id"]].get("matched_terms", []),
                r.get("matched_terms", [])
            )
            merged[r["id"]]["score"] = max(merged[r["id"]]["score"], r["score"])
        else:
            r.setdefault("matched_terms", [])
            r.setdefault("is_multi_match", False)
            merged[r["id"]] = r
    
    # Update is_multi_match flag based on combined matched_terms count
    # Also boost score for multi-match segments (multiple terms found in same segment)
    for seg_id, seg in merged.items():
        matched_terms = seg.get("matched_terms", [])
        num_unique_terms = len(matched_terms)
        seg["matched_words_count"] = num_unique_terms
        seg["is_multi_match"] = num_unique_terms > 1
        
        # Boost score for multi-match: each additional term adds a small boost
        if num_unique_terms > 1 and seg.get("score", 0) < 0.98:
            multi_match_boost = min(0.05 * (num_unique_terms - 1), 0.15)  # Max 15% boost
            seg["score"] = round(min(seg["score"] + multi_match_boost, 0.98), 4)

    # Sort: EXACT PHRASE MATCHES FIRST, then TITLE MATCHES, then SPEAKER MATCHES, then by score, then by start time for stability
    # Priority order: exact_phrase (3) > title_match (2) > speaker (1) > summary/text (0)
    def match_priority(x):
        match_types = x.get("match_types", [])
        matched_field = x.get("matched_field", "")
        if "exact_phrase_match" in match_types:
            return 4
        if "title_match" in match_types or matched_field == "video_title":
            return 3
        if "speaker_filter" in match_types or "speaker_field_match" in match_types or matched_field == "speaker":
            return 2
        if matched_field == "video_summary":
            return 1
        return 0
    
    merged_list = sorted(
        merged.values(),
        key=lambda x: (
            match_priority(x),  # Priority: exact_phrase > title > speaker > summary > text
            x.get("score", 0),
            -x.get("start_time", 0)
        ),
        reverse=True
    )
    
    # ADVANCED: Apply LLM reranking if enabled (replaces Cohere English-only reranker)
    # Skip reranking for small result sets (not worth the latency) or when we have good matches
    # Protect title-matched and exact-phrase-matched results: they should keep a minimum score floor
    has_high_confidence_results = any(r.get("score", 0) >= 0.8 for r in merged_list[:5])
    should_rerank = use_reranking and query_text and len(query_text.split()) > 1 and len(merged_list) >= 15 and not has_high_confidence_results
    
    if should_rerank:
        logger.info(f"Applying LLM reranking to {len(merged_list)} results...")
        
        # Remember pre-rerank scores for protected match types so we can enforce a floor
        title_match_scores = {}
        exact_phrase_scores = {}
        for r in merged_list:
            if "title_match" in r.get("match_types", []):
                title_match_scores[r["id"]] = r["score"]
            if "exact_phrase_match" in r.get("match_types", []):
                exact_phrase_scores[r["id"]] = r["score"]
        
        merged_list = rerank_with_llm(query_text, merged_list, top_k=min(len(merged_list), top_k * 3))
        
        # Restore score floors for protected match types
        # Exact phrase matches should ALWAYS keep 0.99 score (they are confirmed transcript matches!)
        # Title matches should keep a floor proportional to their original score
        if merged_list:
            for r in merged_list:
                if r["id"] in exact_phrase_scores:
                    # Exact phrase matches are CONFIRMED matches - always restore 0.99 score
                    r["score"] = 0.99
                    # Ensure the match type is preserved
                    if "exact_phrase_match" not in r.get("match_types", []):
                        r["match_types"].append("exact_phrase_match")
                elif r["id"] in title_match_scores:
                    original = title_match_scores[r["id"]]
                    if r["score"] < original * 0.7:
                        r["score"] = round(max(r["score"], original * 0.80), 4)
            # Re-sort: priority order exact_phrase > title > speaker > others, then by score
            merged_list.sort(
                key=lambda x: (
                    2 if "exact_phrase_match" in x.get("match_types", []) else (1 if "title_match" in x.get("match_types", []) else 0),
                    x.get("score", 0)
                ),
                reverse=True
            )
            logger.info(f"LLM reranking complete, top score: {merged_list[0].get('score', 0):.4f}, exact-phrase-protected: {len(exact_phrase_scores)}, title-protected: {len(title_match_scores)}")
        else:
            logger.info("LLM reranking returned no results")
    elif use_reranking and query_text:
        logger.info(f"Skipping LLM reranking (results={len(merged_list)}, high_confidence={has_high_confidence_results}) for speed")
    
    # Group by video and collect ALL matching segments per video
    # CONSOLIDATE adjacent/overlapping segments into full segments
    # This prevents showing chunks of the same segment multiple times
    videos_seen = {}
    final_results = []
    
    # First, group all results by video_id
    video_segments = {}
    for result in merged_list:
        vid = result.get("video_id")
        if vid not in video_segments:
            video_segments[vid] = []
        video_segments[vid].append(result)
    
    # For each video, consolidate overlapping/adjacent segments
    consolidated_segments = []
    for vid, segments in video_segments.items():
        # Sort segments by start_time
        segments.sort(key=lambda x: x.get("start_time", 0))
        
        # Merge adjacent/overlapping segments (within 5 seconds)
        merged_segs = []
        for seg in segments:
            if not merged_segs:
                merged_segs.append({
                    **seg,
                    "segment_ids": [seg["id"]],
                    "match_count": 1,
                    "texts": [seg.get("text", "")],
                    "all_matched_terms": seg.get("matched_terms", []).copy(),  # Track all matched terms
                })
            else:
                last = merged_segs[-1]
                last_end = last.get("end_time", 0)
                seg_start = seg.get("start_time", 0)
                
                # If segments are adjacent or overlapping (within 5 sec gap)
                if seg_start <= last_end + 5:
                    # Consolidate into the existing segment
                    last["segment_ids"].append(seg["id"])
                    last["match_count"] += 1
                    last["end_time"] = max(last.get("end_time", 0), seg.get("end_time", 0))
                    last["duration"] = round(last["end_time"] - last["start_time"], 2)
                    last["score"] = max(last.get("score", 0), seg.get("score", 0))
                    # Append text if not duplicate
                    seg_text = seg.get("text", "")
                    if seg_text and seg_text not in " ".join(last["texts"]):
                        last["texts"].append(seg_text)
                    # Merge match types
                    for mt in seg.get("match_types", []):
                        if mt not in last.get("match_types", []):
                            last["match_types"].append(mt)
                    # Merge matched_terms from consolidated segment
                    for term in seg.get("matched_terms", []):
                        term_str = term.get("term", "")
                        existing_terms = {t.get("term", "") for t in last.get("all_matched_terms", [])}
                        if term_str and term_str not in existing_terms:
                            last["all_matched_terms"].append(term)
                    # Keep highest fuzzy score
                    last["fuzzy_score"] = max(last.get("fuzzy_score", 0), seg.get("fuzzy_score", 0))
                    # Keep highest LLM score if present
                    if seg.get("llm_relevance_score"):
                        last["llm_relevance_score"] = max(
                            last.get("llm_relevance_score") or 0, 
                            seg.get("llm_relevance_score", 0)
                        )
                else:
                    # New segment group
                    merged_segs.append({
                        **seg,
                        "segment_ids": [seg["id"]],
                        "match_count": 1,
                        "texts": [seg.get("text", "")],
                        "all_matched_terms": seg.get("matched_terms", []).copy(),
                    })
        
        # Finalize text for each consolidated segment
        for seg in merged_segs:
            # Join texts with proper spacing, removing duplicates
            seg["text"] = " ".join(seg["texts"])
            seg["text_length"] = len(seg["text"])
            # Update is_multi_match based on consolidated matched_terms
            all_terms = seg.get("all_matched_terms", [])
            seg["matched_terms"] = all_terms
            seg["matched_words_count"] = len(all_terms)
            seg["is_multi_match"] = len(all_terms) > 1
            # Boost score further for consolidated multi-match
            if seg["is_multi_match"] and seg.get("score", 0) < 0.98:
                consolidated_boost = min(0.03 * (seg["match_count"] - 1), 0.10)  # Bonus for consolidated segments
                seg["score"] = round(min(seg["score"] + consolidated_boost, 0.98), 4)
            del seg["texts"]  # Clean up temporary field
            if "all_matched_terms" in seg:
                del seg["all_matched_terms"]  # Clean up - use matched_terms instead
            consolidated_segments.append(seg)
    
    # Re-sort consolidated segments: priority order title > speaker > summary > text, then by score
    # This ensures the most relevant match types are always shown first
    def consolidated_priority(x):
        match_types = x.get("match_types", [])
        matched_field = x.get("matched_field", "")
        if "exact_phrase_match" in match_types:
            return 4
        if "title_match" in match_types or matched_field == "video_title":
            return 3
        if "speaker_filter" in match_types or "speaker_field_match" in match_types or matched_field == "speaker":
            return 2
        if matched_field == "video_summary":
            return 1
        return 0
    
    consolidated_segments.sort(
        key=lambda x: (
            consolidated_priority(x),
            x.get("score", 0),
            -x.get("start_time", 0)
        ),
        reverse=True
    )
    
    # Log exact phrase and title matches for debugging
    exact_phrase_count = sum(1 for s in consolidated_segments if "exact_phrase_match" in s.get("match_types", []))
    title_match_count = sum(1 for s in consolidated_segments if "title_match" in s.get("match_types", []))
    if exact_phrase_count > 0:
        logger.info(f"Exact phrase matches found: {exact_phrase_count} segments will be prioritized at top")
    if title_match_count > 0:
        logger.info(f"Title matches found: {title_match_count} videos matched by title")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY RELEVANCE VALIDATION - Check if ANY results are actually relevant
    # This prevents showing completely unrelated results (e.g., "bill gates" → Pakistan politics)
    # ═══════════════════════════════════════════════════════════════════════════
    if consolidated_segments and query_text and len(query_text.strip()) >= 3:
        query_word_count = len(query_text.split())
        # Skip validation if we have verified matches (exact phrase or title matches)
        has_verified_matches = exact_phrase_count > 0 or title_match_count > 0
        has_explicit_keyword_hits = len(keyword_results) > 0
        skip_validation_for_short_query = query_word_count <= 1
        
        if not has_verified_matches and not has_explicit_keyword_hits and not skip_validation_for_short_query and use_reranking and openai_client:
            # Validate if top results are actually relevant to the query
            relevance_check = validate_query_relevance(query_text, consolidated_segments[:5])
            
            # If ALL top results are irrelevant, return empty result set
            if not relevance_check["is_relevant"]:
                logger.warning(f"Query '{query_text[:50]}' returned no relevant results. Max relevance: {relevance_check['max_relevance']:.2f}. Reason: {relevance_check['explanation']}")
                return {
                    "query": query_text,
                    "words": words,
                    "speaker_filter": speaker_filter,
                    "collection": SEGMENTS_COLLECTION,
                    "total_speaker_hits": 0,
                    "total_semantic_hits": 0,
                    "total_keyword_hits": 0,
                    "total_title_hits": 0,
                    "total_exact_phrase_hits": 0,
                    "returned": 0,
                    "unique_videos": 0,
                    "filters_applied": {
                        "video_id": video_id_filter,
                        "speaker": speaker_filter,
                        "title": title_filter,
                        "language": language_filter,
                        "time_range": time_range,
                        "min_score": min_score,
                    },
                    "relevance_validation": {
                        "performed": True,
                        "passed": False,
                        "max_relevance": relevance_check["max_relevance"],
                        "explanation": relevance_check["explanation"]
                    },
                    "results": [],
                    "message": f"No relevant results found for '{query_text}'. The database may not contain content about this topic."
                }
            else:
                logger.info(f"Query relevance validation passed: {relevance_check['relevant_count']}/{len(consolidated_segments[:5])} results are relevant (max score: {relevance_check['max_relevance']:.2f})")
    
    # Apply limits (up to 10 segments per video, top_k total)
    for result in consolidated_segments:
        vid = result.get("video_id")
        if vid not in videos_seen:
            videos_seen[vid] = 0
        
        # Keep up to 10 segments per video for better context
        if videos_seen[vid] < 10:
            videos_seen[vid] += 1
            final_results.append(result)
        
        if len(final_results) >= top_k:
            break

    logger.info(f"Search completed: {len(speaker_results)} speaker + {len(semantic_results)} semantic + {len(keyword_results)} keyword + {len(title_results)} title = {len(final_results)} merged results from {len(videos_seen)} videos (consolidated from {len(merged_list)} raw matches)")

    return {
        "query": query_text,
        "words":  words,
        "speaker_filter": speaker_filter,
        "collection":  SEGMENTS_COLLECTION,
        "total_speaker_hits": len(speaker_results),
        "total_semantic_hits": len(semantic_results),
        "total_keyword_hits": len(keyword_results),
        "total_title_hits": len(title_results),
        "total_exact_phrase_hits": len(exact_phrase_results),  # Exact transcript matches
        "returned":  len(final_results),
        "unique_videos": len(videos_seen),
        "filters_applied": {
            "video_id": video_id_filter,
            "speaker": speaker_filter,
            "title": title_filter,
            "language": language_filter,
            "time_range": time_range,
            "min_score": min_score,
        },
        "results": [
            {
                "id": r["id"],
                "segment_ids": r.get("segment_ids", [r["id"]]),  # All segment IDs that were consolidated
                "match_count": r.get("match_count", 1),  # Number of segments consolidated
                "is_exact_phrase_match": "exact_phrase_match" in r.get("match_types", []),  # True if exact query text found in transcript
                "is_multi_match": r.get("is_multi_match", False),  # True if multiple terms matched same segment
                "matched_terms": r.get("matched_terms", []),  # List of {term, field, score} for each matched term
                "score": round(r["score"], 4),
                "match_types": r.get("match_types", []),
                "matched_field": r.get("matched_field", ""),
                "fuzzy_score": round(r.get("fuzzy_score", 0), 4),
                "matched_words_count": r.get("matched_words_count", 0),
                "video_id": r.get("video_id"),
                "video_title": r.get("video_title", ""),
                "speaker": r.get("speaker", ""),
                "diarization_speaker": r.get("diarization_speaker", ""),
                "start_time": r.get("start_time", 0),
                "end_time": r.get("end_time", 0),
                "duration": r.get("duration", 0),
                "text": r.get("text", ""),
                "text_length": r.get("text_length", 0),
                "summary_en": r.get("summary_en", ""),
                "youtube_url": r.get("youtube_url", ""),
                "language": r.get("language", ""),
                "created_at": r.get("created_at"),
                "llm_relevance_score": r.get("llm_relevance_score"),
                "youtube_url_timestamped": f"{r.get('youtube_url', '')}?t={int(r.get('start_time', 0))}" if r.get('youtube_url') else ""
            }
            for r in final_results
        ]
    }

@app.post("/search/incremental")
async def search_incremental(data: IncrementalSearchRequest, authorized: bool = Depends(verify_api_key)):
    """
    INCREMENTAL CURSOR-BASED SEARCH for fast pagination.
    
    First request (no cursor):
      - Executes full search using existing multi-strategy search
      - Generates search_session_id (UUID)
      - Caches all results for 30 minutes
      - Returns first batch_size results with cursor
    
    Subsequent requests (with cursor):
      - Retrieves cached results using search_session_id
      - Extracts next batch starting from cursor position
      - Returns batch with next cursor
    
    Returns:
      {
        "success": true,
        "results": [...],  # batch of results
        "cursor": {
          "next": "base64_cursor_string",  # null if no more results
          "has_more": bool,
          "total_available": int
        },
        "search_session_id": "uuid",
        "metadata": {...}
      }
    """
    start_time = time.time()
    
    # Extract parameters
    search_session_id = data.search_session_id
    cursor_str = data.cursor
    batch_size = data.batch_size
    
    # Decode cursor
    cursor = decode_cursor(cursor_str)
    
    # Try to get cached results
    cached_data = get_cached_results(search_session_id) if search_session_id else None
    
    if cached_data:
        # ═══════════════════════════════════════════════════════════════════════════
        # CACHED RESULTS PATH - Check if cache has enough, else expand query
        # ═══════════════════════════════════════════════════════════════════════════
        
        all_results = cached_data.get("results", [])
        query_params = cached_data.get("query_params", {})
        top_k_used = cached_data.get("top_k_used", 20)
        
        # Calculate if cache has enough results for this batch
        cursor_index = cursor.get("index", -1) if cursor else -1
        needed_index = cursor_index + batch_size * 3  # Heuristic: need ~3x batch_size results ahead
        
        # If cache doesn't have enough results, expand query
        if needed_index >= len(all_results) and len(all_results) >= top_k_used * 0.9:
            # We've consumed most of cached results - query for more
            new_top_k = min(top_k_used + 30, 200)  # Increase by 30, max 200
            
            logger.info(f"[INCREMENTAL SEARCH] Cache insufficient (has {len(all_results)}, need ~{needed_index}). Expanding query from top_k={top_k_used} to {new_top_k}...")
            
            # Execute expanded search
            search_request = SearchRequest(
                query=data.query,
                words=data.words,
                word=data.word,
                top_k=new_top_k,
                video_id=data.video_id,
                speaker=data.speaker,
                title=data.title,
                language=data.language,
                min_score=data.min_score,
                time_range=data.time_range,
                max_scanned=data.max_scanned,
                search_mode=data.search_mode,
                filter_type=data.filter_type,
                filter_year=data.filter_year,
                filter_month=data.filter_month,
                filter_date=data.filter_date
            )
            
            try:
                search_response = await search(search_request, authorized=True)
                all_results = search_response.get("results", [])
                
                # Update cache with expanded results
                cache_search_results(search_session_id, all_results, query_params, new_top_k)
                top_k_used = new_top_k
                
                logger.info(f"[INCREMENTAL SEARCH] Cache expanded: now {len(all_results)} results (top_k={new_top_k})")
            except Exception as e:
                logger.error(f"Failed to expand cache: {str(e)}")
                # Fall back to existing cached results
        else:
            logger.info(f"[INCREMENTAL SEARCH] Using cached results for session {search_session_id[:8]} (has {len(all_results)} results)...")
        
        # Extract batch from cached results
        batch, next_cursor, has_more = extract_batch_from_results(all_results, cursor, batch_size)
        next_cursor, has_more, expansion_pending = maybe_mark_expandable_boundary(
            batch=batch,
            next_cursor=next_cursor,
            has_more=has_more,
            all_results=all_results,
            top_k_used=top_k_used,
            max_top_k=200
        )
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "results": batch,
            "cursor": {
                "next": next_cursor,
                "has_more": has_more,
                "total_available": len(all_results)
            },
            "search_session_id": search_session_id,
            "metadata": {
                "elapsed_seconds": round(elapsed, 3),
                "cache_hit": True,
                "batch_size": len(batch),
                "batch_start": cursor.get("index", -1) + 1 if cursor else 0,
                "query": query_params.get("query", ""),
                "top_k_used": top_k_used,
                "expansion_pending": expansion_pending
            }
        }
    
    else:
        # ═══════════════════════════════════════════════════════════════════════════
        # FIRST REQUEST PATH - Execute full search and cache results
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Generate new search session ID
        if not search_session_id:
            search_session_id = str(uuid.uuid4())
        
        logger.info(f"[INCREMENTAL SEARCH] Executing new search for session {search_session_id[:8]}...")
        
        # Convert IncrementalSearchRequest to SearchRequest parameters
        # We'll execute the full search logic inline (calling the search function would be cleaner
        # but requires refactoring - for now, we'll call the existing search endpoint logic)
        
        # Create a SearchRequest object with SMALL top_k for fast initial response
        initial_top_k = 20  # Query only 20 results initially (~4-5 seconds)
        search_request = SearchRequest(
            query=data.query,
            words=data.words,
            word=data.word,
            top_k=initial_top_k,  # FAST: Query only 20 instead of 200
            video_id=data.video_id,
            speaker=data.speaker,
            title=data.title,
            language=data.language,
            min_score=data.min_score,
            time_range=data.time_range,
            max_scanned=data.max_scanned,
            search_mode=data.search_mode,
            filter_type=data.filter_type,
            filter_year=data.filter_year,
            filter_month=data.filter_month,
            filter_date=data.filter_date
        )
        
        # Execute the main search (reuse existing search logic)
        try:
            search_response = await search(search_request, authorized=True)
            
            # Extract results from search response
            all_results = search_response.get("results", [])
            
            # Cache the results with top_k tracking
            query_params = {
                "query": data.query,
                "speaker": data.speaker,
                "title": data.title,
                "video_id": data.video_id,
                "language": data.language,
                "search_mode": data.search_mode,
                "filter_type": data.filter_type
            }
            cache_search_results(search_session_id, all_results, query_params, initial_top_k)
            
            # Extract first batch
            batch, next_cursor, has_more = extract_batch_from_results(all_results, None, batch_size)
            next_cursor, has_more, expansion_pending = maybe_mark_expandable_boundary(
                batch=batch,
                next_cursor=next_cursor,
                has_more=has_more,
                all_results=all_results,
                top_k_used=initial_top_k,
                max_top_k=200
            )
            
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "results": batch,
                "cursor": {
                    "next": next_cursor,
                    "has_more": has_more,
                    "total_available": len(all_results)
                },
                "search_session_id": search_session_id,
                "metadata": {
                    "elapsed_seconds": round(elapsed, 3),
                    "cache_hit": False,
                    "batch_size": len(batch),
                    "batch_start": 0,
                    "query": data.query,
                    "total_found": search_response.get("total_found", 0),
                    "search_mode": data.search_mode,
                    "top_k_used": initial_top_k,
                    "expansion_pending": expansion_pending
                }
            }
            
        except Exception as e:
            logger.error(f"Incremental search failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Search execution failed: {str(e)}"
            )
    
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
    
    query_vector = get_cached_embedding(query_text)
    
    try:
        search_filter = Filter(
            should=[
                FieldCondition(key="video_id", match=MatchValue(value=vid))
                for vid in video_ids
            ]
        )
        
        search_response = qdrant_client.query_points(
            collection_name=SEGMENTS_COLLECTION,
            query=query_vector,
            limit=top_k * len(video_ids),
            score_threshold=min_score,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False
        )
        search_results = search_response.points
        
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

@app.post("/re-embed-all")
async def re_embed_all(authorized: bool = Depends(verify_api_key)):
    """
    Re-embed all existing segments with enriched context text and English summaries.
    Processes in batches of 200. This is an admin endpoint - run once after deployment.
    Requires API key.
    """
    try:
        logger.info("Starting re-embedding of all segments with enriched text...")
        
        total_processed = 0
        total_updated = 0
        total_errors = 0
        offset = None
        batch_size = 200
        
        while True:
            # Scroll through all points
            points, next_offset = qdrant_client.scroll(
                collection_name=SEGMENTS_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
            
            texts_to_embed = []
            points_to_update = []
            
            for p in points:
                total_processed += 1
                payload = p.payload or {}
                text = payload.get("text", "")
                speaker = payload.get("speaker", "UNKNOWN")
                video_title = payload.get("video_title", "")
                language = payload.get("language", "")
                
                if not text or len(text.strip()) < 3:
                    continue
                
                # Build enriched embedding text
                enriched_parts = []
                if speaker and speaker != "UNKNOWN":
                    enriched_parts.append(f"[Speaker: {speaker}]")
                if video_title:
                    enriched_parts.append(f"[Title: {video_title}]")
                if language:
                    enriched_parts.append(f"[Language: {language}]")
                enriched_parts.append(text)
                enriched_text = " ".join(enriched_parts)
                
                texts_to_embed.append(enriched_text)
                points_to_update.append({
                    "id": p.id,
                    "payload": payload,
                    "enriched_text": enriched_text
                })
            
            if not texts_to_embed:
                offset = next_offset
                if not next_offset:
                    break
                continue
            
            # Generate new embeddings
            try:
                if USE_OPENAI_EMBEDDINGS and openai_client:
                    vectors = get_openai_embeddings_batch(texts_to_embed)
                else:
                    results_list = list(get_fastembed_model().embed(texts_to_embed))
                    vectors = [r.tolist() if hasattr(r, 'tolist') else list(r) for r in results_list]
                
                # Generate English summaries for non-English content
                summaries = {}
                non_english_texts = []
                non_english_indices = []
                for i, pt in enumerate(points_to_update):
                    lang = pt["payload"].get("language", "")
                    if lang and lang.lower() not in ["english", "en"]:
                        non_english_texts.append(f"[{i}] {pt['payload'].get('text', '')[:300]}")
                        non_english_indices.append(i)
                
                if non_english_texts and openai_client:
                    try:
                        for batch_start in range(0, len(non_english_texts), 20):
                            batch = non_english_texts[batch_start:batch_start + 20]
                            summary_resp = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Translate/summarize each numbered transcript segment into a brief English summary (1 line each). Prefix with the segment number in brackets."},
                                    {"role": "user", "content": "\n".join(batch)}
                                ],
                                max_tokens=1000,
                                temperature=0.1
                            )
                            for line in summary_resp.choices[0].message.content.strip().split("\n"):
                                line = line.strip()
                                if line and "[" in line:
                                    try:
                                        idx_str = line.split("]")[0].replace("[", "").strip()
                                        summary_text = "]".join(line.split("]")[1:]).strip().lstrip("- :")
                                        summaries[int(idx_str)] = summary_text
                                    except (ValueError, IndexError):
                                        pass
                    except Exception as e:
                        logger.warning(f"Summary generation failed in re-embed: {str(e)}")
                
                # Upsert updated points
                new_points = []
                for i, pt in enumerate(points_to_update):
                    updated_payload = pt["payload"].copy()
                    updated_payload["enriched_text"] = pt["enriched_text"]
                    updated_payload["summary_en"] = summaries.get(i, updated_payload.get("summary_en", ""))
                    
                    new_points.append(PointStruct(
                        id=pt["id"],
                        vector=vectors[i],
                        payload=updated_payload
                    ))
                
                qdrant_client.upsert(
                    collection_name=SEGMENTS_COLLECTION,
                    points=new_points,
                    wait=True
                )
                total_updated += len(new_points)
                logger.info(f"Re-embedded batch: {total_updated}/{total_processed} processed")
                
            except Exception as e:
                total_errors += 1
                logger.error(f"Error re-embedding batch: {str(e)}")
            
            offset = next_offset
            if not next_offset:
                break
        
        return {
            "success": True,
            "total_processed": total_processed,
            "total_updated": total_updated,
            "total_errors": total_errors,
            "message": f"Re-embedded {total_updated} segments with enriched text and summaries"
        }
        
    except Exception as e:
        logger.error(f"Re-embed-all failed: {str(e)}")
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

class UpdatePayloadRequest(BaseModel):
    video_id: int = Field(..., gt=0, description="Video ID to update")
    video_title: Optional[str] = Field(default=None, max_length=500)
    video_filename: Optional[str] = Field(default=None, max_length=500)
    youtube_url: Optional[str] = Field(default=None, max_length=1000)
    language: Optional[str] = Field(default=None, max_length=50)
    video_created_at: Optional[str] = Field(default=None, description="ISO datetime")
    processing_status: Optional[str] = Field(default=None, max_length=50)
    approval_status: Optional[str] = Field(default=None, max_length=50)
    is_archived: Optional[bool] = Field(default=None)
    user_id: Optional[int] = Field(default=None)
    speakers_count: Optional[int] = Field(default=None)
    audio_duration_seconds: Optional[float] = Field(default=None)
    video_description: Optional[str] = Field(default=None, max_length=5000)
    video_summary: Optional[str] = Field(default=None, max_length=10000)
    video_summary_english: Optional[str] = Field(default=None, max_length=10000)
    video_summary_urdu: Optional[str] = Field(default=None, max_length=10000)


@app.post("/update-video-payload")
async def update_video_payload(data: UpdatePayloadRequest, authorized: bool = Depends(verify_api_key)):
    """
    Update payload metadata on existing Qdrant points for a video WITHOUT re-generating embeddings.
    Only the provided (non-None) fields are updated; existing payload fields are preserved.
    """
    try:
        video_id = data.video_id

        # Build payload dict from non-None fields (skip video_id itself)
        payload_update = {}
        for field_name, field_value in data.dict(exclude={"video_id"}).items():
            if field_value is not None:
                payload_update[field_name] = field_value

        if not payload_update:
            return {
                "success": True,
                "message": f"No fields to update for video {video_id}",
                "video_id": video_id,
                "updated_fields": [],
                "points_affected": 0,
            }

        # Use set_payload with a filter on video_id — updates ALL points for this video
        qdrant_client.set_payload(
            collection_name=SEGMENTS_COLLECTION,
            payload=payload_update,
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id),
                    )
                ]
            ),
        )

        # Count how many points were affected
        count_result = qdrant_client.count(
            collection_name=SEGMENTS_COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id),
                    )
                ]
            ),
            exact=True,
        )

        logger.info(f"Updated payload for video {video_id}: {list(payload_update.keys())} on {count_result.count} points")

        return {
            "success": True,
            "message": f"Updated {len(payload_update)} fields on {count_result.count} points for video {video_id}",
            "video_id": video_id,
            "updated_fields": list(payload_update.keys()),
            "points_affected": count_result.count,
        }

    except Exception as e:
        logger.error(f"Update payload failed for video {data.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-video-payload-batch")
async def update_video_payload_batch(
    data: List[UpdatePayloadRequest],
    authorized: bool = Depends(verify_api_key)
):
    """
    Batch update payload for multiple videos at once.
    Each item follows the same rules as /update-video-payload.
    """
    results = []
    total_points = 0
    errors = []

    for item in data:
        try:
            payload_update = {}
            for field_name, field_value in item.dict(exclude={"video_id"}).items():
                if field_value is not None:
                    payload_update[field_name] = field_value

            if not payload_update:
                results.append({"video_id": item.video_id, "updated_fields": 0, "points": 0, "status": "skipped"})
                continue

            qdrant_client.set_payload(
                collection_name=SEGMENTS_COLLECTION,
                payload=payload_update,
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=item.video_id),
                        )
                    ]
                ),
            )

            count_result = qdrant_client.count(
                collection_name=SEGMENTS_COLLECTION,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=item.video_id),
                        )
                    ]
                ),
                exact=True,
            )

            total_points += count_result.count
            results.append({
                "video_id": item.video_id,
                "updated_fields": len(payload_update),
                "points": count_result.count,
                "status": "updated",
            })

        except Exception as e:
            logger.error(f"Batch update failed for video {item.video_id}: {str(e)}")
            errors.append({"video_id": item.video_id, "error": str(e)})

    logger.info(f"Batch payload update complete: {len(results)} videos, {total_points} total points, {len(errors)} errors")

    return {
        "success": len(errors) == 0,
        "total_videos": len(results),
        "total_points_affected": total_points,
        "results": results,
        "errors": errors,
    }


@app.get("/health")
async def health():
    try:
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": {
                "segments":  SEGMENTS_COLLECTION,
                "legacy":  LEGACY_COLLECTION
            },
            "points_count": collection_info.points_count,
            "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', None),
            "features": {
                "openai_embeddings": USE_OPENAI_EMBEDDINGS,
                "fastembed_model": FASTEMBED_MODEL_NAME,
                "llm_understanding": os.getenv("USE_LLM_UNDERSTANDING", "true"),
                "llm_reranking": os.getenv("USE_RERANKING", "true"),
            }
        }
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
        return {
            "collection":  SEGMENTS_COLLECTION,
            "points_count": collection_info.points_count,
            "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', None),
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "Video Transcript LLM-Powered Search API",
        "version": "6.0 - LLM ACCURACY ENGINE",
        "ai_features": {
            "openai_embeddings": USE_OPENAI_EMBEDDINGS,
            "embedding_model": OPENAI_EMBEDDING_MODEL if USE_OPENAI_EMBEDDINGS else FASTEMBED_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "fastembed_fallback": FASTEMBED_MODEL_NAME,
            "openai_only_search": os.getenv("OPENAI_ONLY_SEARCH", "true"),
            "llm_query_understanding": os.getenv("USE_LLM_UNDERSTANDING", "true"),
            "llm_reranking": os.getenv("USE_RERANKING", "true"),
            "cross_language_search": "Urdu <-> English via LLM translation",
            "enriched_embeddings": "Speaker + Title + Language context in vectors",
        },
        "endpoints": {
            "POST /embed-video": "Embed video transcript with enriched context + English summaries",
            "POST /search": "LLM-powered: Intent parsing + Dual-language + GPT-4o-mini reranking",
            "POST /search-by-title": "Fuzzy search videos by title",
            "POST /search-multi-video": "Search across multiple specific videos",
            "POST /suggest": "Autocomplete suggestions for speakers/titles",
            "POST /re-embed-all": "Admin: Re-embed all segments with enriched text (run once)",
            "GET /video/{video_id}/segments": "Get all segments for a video",
            "DELETE /video/{video_id}/embeddings": "Delete all embeddings for a video",
            "GET /health": "Health check",
            "GET /stats": "Collection statistics"
        },
        "llm_pipeline": {
            "1_query_understanding": "GPT-4o-mini parses intent, detects language, extracts speaker, translates query",
            "2_dual_language_search": "Searches with original + translated query for cross-language recall",
            "3_semantic_search": "OpenAI text-embedding-3-large (3072-dim) vector search in Qdrant",
            "4_keyword_search": "Qdrant text index for exact keyword matching (no scroll)",
            "5_llm_reranking": "GPT-4o-mini scores top 30 results for true relevance (multilingual)",
            "6_enriched_embeddings": "Vectors include speaker name + video title + language context",
            "7_english_summaries": "Urdu segments get English summary for cross-language matching",
        },
        "environment_variables": {
            "OPENAI_API_KEY": "Required for embeddings, query understanding, and reranking",
            "USE_OPENAI_EMBEDDINGS": "true/false (default: true)",
            "OPENAI_ONLY_SEARCH": "true/false (default: true) - route content filters to semantic and skip scan-heavy fallbacks",
            "USE_LLM_UNDERSTANDING": "true/false (default: true) - GPT-4o-mini query parsing",
            "USE_RERANKING": "true/false (default: true) - GPT-4o-mini reranking",
            "USE_QUERY_EXPANSION": "true/false (default: true) - expanded via LLM understanding",
            "FASTEMBED_MODEL": "FastEmbed model name (default: BAAI/bge-small-en-v1.5)",
            "EMBEDDING_DIMENSION": "3072 for OpenAI large, 1536 for small",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    logger.info(f"Starting FastAPI Video Elastic Search Service on port {port}...")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Collections: {SEGMENTS_COLLECTION}, {LEGACY_COLLECTION}")
    logger.info("Features: Fuzzy search, typo tolerance, query caching")
    uvicorn.run(app, host="0.0.0.0", port=port)


