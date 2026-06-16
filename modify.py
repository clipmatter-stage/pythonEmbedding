import sys
import re

file_path = "embeddings_test.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add redis and rq imports around line 26
import_addition = """import json as json_module
from redis import Redis
from rq import Queue

# ============== REDIS QUEUE INITIALIZATION ==============
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(REDIS_URL)
task_queue = Queue('video_processing', connection=redis_conn)"""
content = content.replace("import json as json_module", import_addition)

# 2. Add webhook_url to EmbedVideoRequest
request_addition = """class EmbedVideoRequest(BaseModel):
    video_id: int = Field(..., gt=0, description="Video ID must be positive")
    identification_segments: List[dict] = Field(..., min_length=1)
    webhook_url: Optional[str] = Field(default=None, description="Optional webhook URL to receive status callbacks")"""
content = content.replace(
    'class EmbedVideoRequest(BaseModel):\n    video_id: int = Field(..., gt=0, description="Video ID must be positive")\n    identification_segments: List[dict] = Field(..., min_length=1)',
    request_addition
)

# 3. Replace embed_video with process_video_task and new embed_video
# We will find the exact string of embed_video
embed_video_start_str = '''@app.post("/embed-video")
async def embed_video(data: EmbedVideoRequest, authorized: bool = Depends(verify_api_key)):
    """Embed video transcript segments. Requires API key if configured."""
    try:
        video_id = data.video_id'''

# Let's verify if embed_video_start_str is in content
if embed_video_start_str not in content:
    print("Could not find embed_video start!")
    sys.exit(1)

# The end of embed_video is just before `def delete_existing_embeddings`
# Let's split by `@app.post("/embed-video")`
parts = content.split('@app.post("/embed-video")')
before_embed = parts[0]
after_embed_start = parts[1]

# Now split by `def delete_existing_embeddings(video_id: int):`
after_parts = after_embed_start.split('def delete_existing_embeddings(video_id: int):')
embed_video_body = after_parts[0]
after_embed = 'def delete_existing_embeddings(video_id: int):' + after_parts[1]

# Create the new content
new_worker_and_endpoint = '''def process_video_task(data_dict: dict):
    """Background worker task to process video embeddings and upsert to Qdrant."""
    webhook_url = data_dict.get('webhook_url')
    video_id = data_dict.get('video_id')
    try:
        identification_segments = data_dict.get('identification_segments', [])
        video_title = data_dict.get('video_title', '')
        video_filename = data_dict.get('video_filename', '')
        youtube_url = data_dict.get('youtube_url', '')
        language = data_dict.get('language', '')
        
        video_created_at = data_dict.get('video_created_at') or ""
        processing_status = data_dict.get('processing_status') or "completed"
        approval_status = data_dict.get('approval_status') or "approved"
        is_archived = data_dict.get('is_archived', False)
        user_id = data_dict.get('user_id')
        speakers_count = data_dict.get('speakers_count', 0)
        audio_duration_seconds = data_dict.get('audio_duration_seconds', 0)
        video_description = data_dict.get('video_description') or ""
        video_summary = data_dict.get('video_summary') or ""
        video_summary_english = data_dict.get('video_summary_english') or ""
        video_summary_urdu = data_dict.get('video_summary_urdu') or ""
        
        logger.info(f"[WORKER] Processing video {video_id} with {len(identification_segments)} segments")
        
        batch_number = 1
        total_batches = 1
        batch_info = data_dict.get('batch_info')
        if batch_info:
            batch_number = batch_info.get("batch_number", 1)
            total_batches = batch_info.get("total_batches", 1)
            logger.info(f"[WORKER] Batch {batch_number}/{total_batches} for video {video_id}")
        
        if batch_number == 1:
            delete_existing_embeddings(video_id)
        else:
            logger.info(f"[WORKER] Skipping delete for batch {batch_number} (only delete on batch 1)")
        
        points = []
        segments_embedded = 0
        segments_without_text = 0
        texts_to_embed = []
        segment_metadata = []
        
        for idx, segment in enumerate(identification_segments):
            segment_index = segment.get("segment_index", idx)
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            diarization_speaker = segment.get("diarizationSpeaker", "")
            match_type = segment.get("match", "")
            confidence = segment.get("confidence", 0)
            
            if not text or len(text.strip()) < 3:
                segments_without_text += 1
                continue
            
            enriched_parts = []
            if speaker and speaker != "UNKNOWN":
                enriched_parts.append(f"[Speaker: {speaker}]")
            if video_title:
                enriched_parts.append(f"[Title: {video_title}]")
            if language:
                enriched_parts.append(f"[Language: {language}]")
            enriched_parts.append(text)
            
            enriched_text = " ".join(enriched_parts)
            
            if len(enriched_text) > 6000:
                enriched_text = enriched_text[:6000] + "..."
                logger.debug(f"Truncated segment {segment_index} from {len(text)} to 6000 chars")
            
            texts_to_embed.append(enriched_text)
            segment_metadata.append({
                'idx': segment_index,
                'speaker': speaker,
                'diarization_speaker': diarization_speaker,
                'match_type': match_type,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'text': text,
                'enriched_text': enriched_text
            })
        
        if not texts_to_embed:
            raise ValueError(f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}")
        
        logger.info(f"[WORKER] Generating embeddings for {len(texts_to_embed)} segments in batch...")
        batch_start_time = datetime.utcnow()
        
        if USE_OPENAI_EMBEDDINGS and openai_client:
            logger.info(f"[WORKER] Using OpenAI {OPENAI_EMBEDDING_MODEL} for batch embedding")
            vectors = get_openai_embeddings_batch(texts_to_embed)
        else:
            logger.info("[WORKER] Using FastEmbed for batch embedding")
            results_list = list(get_fastembed_model().embed(texts_to_embed))
            vectors = [r.tolist() if hasattr(r, 'tolist') else list(r) for r in results_list]
        
        summaries_en = {}
        if openai_client and language and language.lower() not in ["english", "en"]:
            try:
                logger.info(f"[WORKER] Generating English summaries for {len(segment_metadata)} segments...")
                for batch_start in range(0, len(segment_metadata), 20):
                    batch_end = min(batch_start + 20, len(segment_metadata))
                    batch_texts = []
                    for m in segment_metadata[batch_start:batch_end]:
                        batch_texts.append(f"[{m['idx']}] {m['text'][:300]}")
                    
                    prompt_text = "\\n".join(batch_texts)
                    summary_response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Translate/summarize each numbered transcript segment into a brief English summary (1 line each). Return one summary per line, prefixed with the segment number in brackets. Keep it concise."},
                            {"role": "user", "content": prompt_text}
                        ],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    
                    for line in summary_response.choices[0].message.content.strip().split("\\n"):
                        line = line.strip()
                        if line and "[" in line:
                            try:
                                idx_str = line.split("]")[0].replace("[", "").strip()
                                summary_text = "]".join(line.split("]")[1:]).strip().lstrip("- :")
                                summaries_en[int(idx_str)] = summary_text
                            except (ValueError, IndexError):
                                pass
            except Exception as e:
                logger.warning(f"[WORKER] Summary generation failed (non-critical): {str(e)}")
        
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        logger.info(f"[WORKER] Batch embedding completed in {batch_duration:.2f} seconds")
        
        for i, metadata in enumerate(segment_metadata):
            vector = vectors[i]
            id_string = f"video_{video_id}_seg_{metadata['idx']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))
            
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
                "match_type": metadata['match_type'],
                "start_time": metadata['start_time'],
                "end_time": metadata['end_time'],
                "duration": metadata['end_time'] - metadata['start_time'],
                "text": metadata['text'],
                "text_length": len(metadata['text']),
                "confidence": metadata['confidence'],
                "summary_en": summary_en,
                "enriched_text": metadata.get('enriched_text', metadata['text']),
                "created_at": datetime.utcnow().isoformat(),
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
            raise ValueError(f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}")
        
        logger.info(f"[WORKER] Inserting {len(points)} points into Qdrant...")
        
        QDRANT_BATCH_SIZE = 150
        for i in range(0, len(points), QDRANT_BATCH_SIZE):
            sub_batch = points[i:i + QDRANT_BATCH_SIZE]
            logger.info(f"[WORKER] Upserting Qdrant sub-batch {i // QDRANT_BATCH_SIZE + 1}: {len(sub_batch)} points")
            qdrant_client.upsert(
                collection_name=SEGMENTS_COLLECTION,
                points=sub_batch,
                wait=True
            )
        logger.info(f"[WORKER] Successfully inserted {len(points)} points")
        
        if webhook_url:
            payload_success = {
                "status": "completed",
                "video_id": video_id,
                "segments_embedded": segments_embedded,
                "message": "Successfully indexed in Qdrant"
            }
            try:
                requests.post(webhook_url, json=payload_success, timeout=10)
                logger.info(f"[WORKER] Webhook success sent to {webhook_url}")
            except Exception as we:
                logger.error(f"[WORKER] Webhook failed: {str(we)}")
                
    except Exception as e:
        logger.error(f"[WORKER] Error processing video {video_id}: {str(e)}")
        if webhook_url:
            payload_error = {
                "status": "failed",
                "video_id": video_id,
                "error": str(e)
            }
            try:
                requests.post(webhook_url, json=payload_error, timeout=10)
            except Exception as we:
                logger.error(f"[WORKER] Webhook error notification failed: {str(we)}")
        raise e

@app.post("/embed-video", status_code=202)
async def embed_video(data: EmbedVideoRequest, authorized: bool = Depends(verify_api_key)):
    """Enqueue video transcript embedding task. Requires API key if configured."""
    try:
        data_dict = data.model_dump()
        job = task_queue.enqueue(process_video_task, data_dict, job_timeout='1h')
        
        return {
            "status": "queued",
            "video_id": data.video_id,
            "job_id": job.id,
            "message": "Video is processing in the background"
        }
    except Exception as e:
        logger.error(f"Error enqueueing task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

'''

content = before_embed + new_worker_and_endpoint + after_embed

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("File updated successfully.")
