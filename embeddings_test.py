from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchText, SearchParams
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict
import re
import requests

app = FastAPI()

print("Loading sentence-transformers model...", flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully", flush=True)

print("Warming up model...", flush=True)
_ = model.encode("warmup text", show_progress_bar=False)
print("Model warmed up and ready", flush=True)

QDRANT_URL = "https://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzY5NjA5NTM4fQ.zKJKxYEPp7JHVxa6DuS4nMgp5Uy6_2NHFeFrJCMjrKY"

SEGMENTS_COLLECTION = "video_transcript_segments"
LEGACY_COLLECTION = "text_embeddings"

print("Connecting to Qdrant...", flush=True)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
print("Connected to Qdrant successfully", flush=True)

def create_segments_collection():
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if SEGMENTS_COLLECTION not in collection_names:
            print(f"Creating collection '{SEGMENTS_COLLECTION}'...", flush=True)
            qdrant_client.create_collection(
                collection_name=SEGMENTS_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
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
            print(f"Collection '{SEGMENTS_COLLECTION}' created successfully", flush=True)
        else:
            print(f"Collection '{SEGMENTS_COLLECTION}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)
        raise

def create_legacy_collection():
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if LEGACY_COLLECTION not in collection_names:
            print(f"Creating legacy collection '{LEGACY_COLLECTION}'...", flush=True)
            qdrant_client.create_collection(
                collection_name=LEGACY_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance. COSINE),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000)
            )
            print(f"Collection '{LEGACY_COLLECTION}' created successfully", flush=True)
    except Exception as e:
        print(f"Error managing legacy collection:  {str(e)}", flush=True)


def ensure_indexes_http():
    """Create required indexes using direct HTTP requests to avoid client version issues."""
    try: 
        print("Creating indexes via HTTP...", flush=True)
        
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
                    print(f"  ✓ Created index for '{field_name}'", flush=True)
                elif response.status_code == 400 and "already exists" in response. text. lower():
                    print(f"  ✓ Index for '{field_name}' already exists", flush=True)
                else:
                    print(f"  ✗ Failed to create index for '{field_name}':  {response.text}", flush=True)
                    
            except Exception as e:
                print(f"  ✗ Error creating index for '{field_name}': {str(e)}", flush=True)
        
        print("Index creation completed", flush=True)
        
    except Exception as e:
        print(f"Error in ensure_indexes_http: {str(e)}", flush=True)


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
async def embed_video(data: dict):
    try:
        video_id = data.get("video_id")
        if not video_id:
            raise HTTPException(status_code=400, detail="video_id is required")
        
        identification_segments = data.get("identification_segments", [])
        if not identification_segments:
            raise HTTPException(status_code=400, detail="identification_segments is required")
        
        video_title = data.get("video_title", "")
        video_filename = data.get("video_filename", "")
        youtube_url = data.get("youtube_url", "")
        language = data.get("language", "")
        
        print(f"Processing video {video_id} with {len(identification_segments)} segments", flush=True)
        
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
        
        print(f"Generating embeddings for {len(texts_to_embed)} segments in batch.. .", flush=True)
        batch_start_time = datetime.utcnow()
        vectors = model.encode(texts_to_embed, show_progress_bar=False, batch_size=32).tolist()
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        print(f"Batch embedding completed in {batch_duration:.2f} seconds", flush=True)
        
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
        
        print(f"Inserting {len(points)} points into Qdrant...", flush=True)
        
        try:
            qdrant_client.upsert(
                collection_name=SEGMENTS_COLLECTION,
                points=points,
                wait=True
            )
            print(f"Successfully inserted {len(points)} points", flush=True)
        except Exception as e:
            print(f"ERROR:  Qdrant insertion failed: {str(e)}", flush=True)
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
        print(f"Error in embed_video: {str(e)}", flush=True)
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
        print(f"Deleted existing embeddings for video {video_id}", flush=True)
    except Exception as e:
        print(f"Note: Could not delete embeddings (may not exist): {str(e)}", flush=True)

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
async def search(data: dict):
    """
    Enhanced search with proper handling of query vs filters. 
    
    - query: Text to search semantically in transcript content
    - words: Keywords to find in transcript text
    - speaker: Filter by speaker name (exact or partial match)
    - video_id: Filter by specific video
    - title: Filter by video title
    """
    query_text = data.get("query", "")
    words = data.get("words", [])
    word = data.get("word")
    top_k = int(data.get("top_k", 10))
    video_id_filter = data.get("video_id")
    speaker_filter = data. get("speaker")
    title_filter = data. get("title")
    language_filter = data. get("language")
    min_score = float(data.get("min_score", 0.3))  # Lower default threshold
    time_range = data. get("time_range")
    max_scanned = int(data.get("max_scanned", 10000))
    
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

    # Build Qdrant filter - only use exact filters
    filter_conditions = []
    
    if video_id_filter is not None:
        filter_conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=video_id_filter))
        )
    
    # Only use speaker filter for exact single-word matches
    if speaker_filter is not None and " " not in speaker_filter:
        filter_conditions.append(
            FieldCondition(key="speaker", match=MatchValue(value=speaker_filter))
        )
    
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

    # Strategy 1: Semantic search
    if query_text:
        try:
            print(f"Semantic search for: '{query_text[: 120]}'", flush=True)
            query_vector = model. encode(query_text).tolist()

            sem_search_results = qdrant_client. search(
                collection_name=SEGMENTS_COLLECTION,
                query_vector=query_vector,
                limit=top_k * 3,
                score_threshold=min_score,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

            for r in sem_search_results: 
                # Client-side title filter if needed
                if title_filter and title_filter. lower() not in r.payload. get("video_title", "").lower():
                    continue
                
                # Client-side speaker search (partial match in text or speaker field)
                if speaker_search_in_text: 
                    text_content = r.payload. get("text", "").lower()
                    speaker_field = r.payload. get("speaker", "").lower()
                    search_term = speaker_search_in_text. lower()
                    if search_term not in text_content and search_term not in speaker_field: 
                        continue
                
                semantic_results.append({
                    "id": r.id,
                    "score": float(getattr(r, "score", 0.0)),
                    "video_id": r. payload.get("video_id"),
                    "video_title": r.payload. get("video_title", ""),
                    "speaker": r. payload.get("speaker", ""),
                    "diarization_speaker": r.payload. get("diarization_speaker", ""),
                    "start_time": r. payload.get("start_time", 0),
                    "end_time":  r.payload.get("end_time", 0),
                    "duration": round((r.payload.get("end_time", 0) - r.payload.get("start_time", 0)), 2),
                    "text": r.payload.get("text", ""),
                    "text_length": r.payload.get("text_length", 0),
                    "youtube_url": r.payload. get("youtube_url", ""),
                    "language": r.payload.get("language", ""),
                    "created_at": r. payload.get("created_at"),
                    "match_types": ["semantic"]
                })

        except Exception as e:
            print(f"ERROR during semantic search: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")

    # Strategy 2: Keyword search (also search for speaker name in text)
    search_words = list(words) if words else []
    if speaker_search_in_text: 
        # Add speaker name words to keyword search
        search_words.extend(speaker_search_in_text.split())
    
    if search_words:
        try:
            print(f"Keyword search for words: {search_words} (max_scanned={max_scanned})", flush=True)
            words_lower = [w.lower() for w in search_words]
            page_size = 1000
            scanned = 0
            offset = None

            scroll_filter = search_filter

            while scanned < max_scanned and len(keyword_results) < top_k * 2:
                points, next_offset = qdrant_client. scroll(
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
                    text = (p.payload.get("text") or "").lower()
                    speaker_field = (p. payload.get("speaker") or "").lower()
                    combined_text = f"{text} {speaker_field}"
                    
                    # Check if ANY word matches (OR logic for names)
                    if any(w in combined_text for w in words_lower):
                        # Client-side title filter
                        if title_filter and title_filter.lower() not in p.payload.get("video_title", "").lower():
                            continue
                        
                        keyword_results.append({
                            "id": p.id,
                            "score": 1.0,
                            "video_id": p.payload.get("video_id"),
                            "video_title":  p.payload.get("video_title", ""),
                            "speaker": p. payload.get("speaker", ""),
                            "diarization_speaker": p. payload.get("diarization_speaker", ""),
                            "start_time":  p.payload.get("start_time", 0),
                            "end_time": p.payload. get("end_time", 0),
                            "duration":  round((p.payload.get("end_time", 0) - p.payload. get("start_time", 0)), 2),
                            "text": p. payload.get("text", ""),
                            "text_length": p.payload. get("text_length", 0),
                            "youtube_url": p.payload.get("youtube_url", ""),
                            "language":  p.payload.get("language", ""),
                            "created_at": p. payload.get("created_at"),
                            "match_types": ["keyword"]
                        })

                if len(keyword_results) >= top_k * 2:
                    break

                offset = next_offset
                if not next_offset:
                    break

        except Exception as e: 
            print(f"ERROR during keyword search: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail=f"Keyword search error: {str(e)}")

    # Merge results
    merged = {}
    
    for r in semantic_results: 
        merged[r["id"]] = r

    for r in keyword_results:
        if r["id"] in merged:
            if "keyword" not in merged[r["id"]]["match_types"]: 
                merged[r["id"]]["match_types"].append("keyword")
            merged[r["id"]]["score"] = max(merged[r["id"]]["score"], 0.95)
        else:
            merged[r["id"]] = r

    merged_list = sorted(
        merged. values(),
        key=lambda x: (x. get("score", 0), -x.get("start_time", 0)),
        reverse=True
    )[:top_k]

    return {
        "query": query_text,
        "words":  words,
        "speaker_searched_in_text": speaker_search_in_text,
        "collection":  SEGMENTS_COLLECTION,
        "total_semantic_hits": len(semantic_results),
        "total_keyword_hits": len(keyword_results),
        "returned":  len(merged_list),
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
                "score": round(r["score"], 4),
                "match_types": r. get("match_types", []),
                "video_id": r.get("video_id"),
                "video_title": r.get("video_title", ""),
                "speaker": r.get("speaker", ""),
                "diarization_speaker": r. get("diarization_speaker", ""),
                "start_time": r.get("start_time", 0),
                "end_time":  r.get("end_time", 0),
                "duration": r.get("duration", 0),
                "text": r. get("text", ""),
                "text_length": r. get("text_length", 0),
                "youtube_url": r.get("youtube_url", ""),
                "language": r.get("language", ""),
                "created_at": r.get("created_at"),
                "youtube_url_timestamped": f"{r. get('youtube_url', '')}?t={int(r.get('start_time', 0))}" if r.get('youtube_url') else ""
            }
            for r in merged_list
        ]
    }
    
@app.post("/search-by-title")
async def search_by_title(data: dict):
    """
    Search for videos by title (returns unique videos, not segments).
    
    Params:
      - title: partial or full video title
      - limit: max number of videos to return
    """
    title = data.get("title", "")
    limit = int(data.get("limit", 10))
    
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    
    try:
        print(f"Searching videos by title: '{title}'", flush=True)
        
        # Scroll through collection to find matching titles
        title_lower = title.lower()
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
                video_title = p.payload.get("video_title", "")
                video_id = p.payload.get("video_id")
                
                if title_lower in video_title.lower() and video_id not in videos:
                    videos[video_id] = {
                        "video_id": video_id,
                        "video_title": video_title,
                        "youtube_url": p.payload.get("youtube_url", ""),
                        "language": p.payload.get("language", "")
                    }
                
                if len(videos) >= limit:
                    break
            
            offset = next_offset
            if not next_offset:
                break
        
        return {
            "title_query": title,
            "total_videos_found": len(videos),
            "scanned_segments": scanned,
            "videos": list(videos.values())[:limit]
        }
        
    except Exception as e:
        print(f"ERROR in search_by_title: {str(e)}", flush=True)
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
    
    print(f"Searching across {len(video_ids)} videos for: '{query_text[: 100]}'", flush=True)
    
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
async def get_video_segments(video_id: int, limit: int = 100, offset: int = 0):
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
                    "segment_index": p.payload.get("segment_index"),
                    "speaker": p.payload.get("speaker"),
                    "start_time":  p.payload.get("start_time"),
                    "end_time": p.payload.get("end_time"),
                    "duration": p.payload.get("duration"),
                    "text": p.payload.get("text"),
                    "text_length": p.payload.get("text_length"),
                }
                for p in points
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/video/{video_id}/embeddings")
async def delete_video_embeddings(video_id:  int):
    try:
        delete_existing_embeddings(video_id)
        return {
            "success": True,
            "message":  f"Deleted all embeddings for video {video_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            "vectors_count": collection_info.vectors_count
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
            "vectors_count": collection_info. vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance. name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service":  "Video Transcript Semantic Search API",
        "version":  "3.0",
        "endpoints": {
            "POST /embed-video": "Embed entire video transcript",
            "POST /search": "Enhanced semantic + keyword + title search",
            "POST /search-by-title": "Search videos by title",
            "POST /search-multi-video": "Search across multiple specific videos",
            "GET /video/{video_id}/segments": "Get all segments for a video",
            "DELETE /video/{video_id}/embeddings": "Delete all embeddings for a video",
            "GET /health": "Health check",
            "GET /stats": "Collection statistics"
        },
        "search_features": [
            "Semantic search using embeddings",
            "Multi-keyword search (AND/OR logic)",
            "Video title search (fuzzy matching)",
            "Speaker filtering",
            "Time range filtering",
            "Query parsing (e.g., 'speaker:John title:intro AI')",
            "Combined search strategies",
            "Score boosting for keyword matches"
        ],
        "example_queries": {
            "semantic": {"query": "machine learning algorithms"},
            "keyword": {"words": ["neural", "network"], "query": "AI"},
            "speaker": {"query": "AI", "speaker": "John"},
            "title": {"query": "python", "title": "introduction"},
            "parsed": {"query": "speaker:John title:intro machine learning"},
            "multi_video": {"query": "AI", "video_ids": [1, 2, 3]}
        }
    }

if __name__ == "__main__": 
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"Starting FastAPI Video Embedding Service on port {port}.. .", flush=True)
    print(f"Qdrant URL: {QDRANT_URL}", flush=True)
    print(f"Collections: {SEGMENTS_COLLECTION}, {LEGACY_COLLECTION}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
