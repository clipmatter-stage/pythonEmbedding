"""Strictly authenticated, bounded snapshots and payload-only timestamp repair.
No embeddings, point deletion or whole-video replacement is performed here.
"""
import hashlib
import json
import math

VERSION='index_operations_v1'
FIELDS={'start_time','end_time','duration','timestamp_unit'}

def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class IndexOperationError(RuntimeError):pass

def patch_timestamps(client,collection,video_id,changes,*,dry_run=True):
    if not changes or len(changes)>5:raise IndexOperationError('Select 1–5 points per request')
    ids=[c['point_id'] for c in changes]
    if len(set(map(str,ids)))!=len(ids):raise IndexOperationError('Duplicate point identity')
    points=client.retrieve(collection_name=collection,ids=ids,with_payload=True,with_vectors=False,timeout=5)
    by_id={str(p.id):p for p in points};before={}
    for change in changes:
        point=by_id.get(str(change['point_id']));payload=point.payload if point else None
        if not payload or payload.get('video_id')!=video_id:raise IndexOperationError('Point/video mismatch')
        if payload_hash(payload)!=change['expected_payload_sha256']:raise IndexOperationError('Payload changed; re-audit')
        update=change['payload']
        if set(update)!=FIELDS or update['timestamp_unit']!='seconds':raise IndexOperationError('Only complete seconds intervals may be patched')
        if any(type(update[k]) not in (int,float) or not math.isfinite(update[k]) for k in ['start_time','end_time','duration']):raise IndexOperationError('Invalid numeric interval')
        if not 0<=update['start_time']<update['end_time'] or abs(update['end_time']-update['start_time']-update['duration'])>.001:raise IndexOperationError('Invalid interval')
        before[str(point.id)]=dict(payload)
    changed=[]
    if not dry_run:
        try:
            for c in changes:
                # Include an uncertain/timed-out write in rollback as well.
                changed.append(c['point_id'])
                client.set_payload(collection_name=collection,payload=c['payload'],points=[c['point_id']],wait=True,timeout=5)
            actual=client.retrieve(collection_name=collection,ids=ids,with_payload=True,with_vectors=False,timeout=5)
            now={str(p.id):p.payload for p in actual}
            if any(any(now.get(str(c['point_id']),{}).get(k)!=v for k,v in c['payload'].items()) for c in changes):raise IndexOperationError('Post-write verification failed')
        except Exception as exc:
            restored=True
            for point_id in changed:
                old=before[str(point_id)]
                try:
                    client.set_payload(collection_name=collection,payload={k:old[k] for k in FIELDS if k in old},points=[point_id],wait=True,timeout=5)
                    missing=list(FIELDS-set(old))
                    if missing:client.delete_payload(collection_name=collection,keys=missing,points=[point_id],wait=True,timeout=5)
                except Exception:restored=False
            raise IndexOperationError('Repair failed; rollback '+('applied, verify snapshots' if restored else 'incomplete, restore snapshots')) from exc
    return {'operation_version':VERSION,'video_id':video_id,'dry_run':dry_run,'points':len(changes),'vectors_changed':False,
            'before_hashes':{k:payload_hash(v) for k,v in before.items()}}

def register_index_operations(app,client,collection,redis,api_key):
    import hmac
    from typing import Optional
    from fastapi import Depends,HTTPException,Request
    from pydantic import BaseModel,Field
    from qdrant_client import models
    async def authorized(request:Request):
        if not api_key:raise HTTPException(503,detail={'code':'operations_not_configured'})
        if not hmac.compare_digest(request.headers.get('x-api-key',''),api_key):raise HTTPException(401,detail='Unauthorized')
    class PatchRequest(BaseModel):
        video_id:int=Field(gt=0)
        changes:list[dict]=Field(min_length=1,max_length=5)
        dry_run:bool=True
    @app.get('/operations/points')
    async def snapshot(video_id:Optional[int]=None,offset:Optional[str]=None,limit:int=50,with_vectors:bool=False,auth=Depends(authorized)):
        if not 1<=limit<=100 or (video_id is not None and video_id<1):raise HTTPException(422,detail='Invalid snapshot scope')
        filter_=models.Filter(must=[models.FieldCondition(key='video_id',match=models.MatchValue(value=video_id))]) if video_id else None
        points,next_offset=client.scroll(collection_name=collection,scroll_filter=filter_,offset=int(offset) if offset and offset.isdigit() else offset,
            limit=limit,with_payload=True,with_vectors=with_vectors,timeout=10)
        return {'operation_version':VERSION,'collection':collection,'next_offset':next_offset,
                'points':[{'id':p.id,'payload':p.payload,'payload_sha256':payload_hash(p.payload or {}),**({'vector':p.vector} if with_vectors else {})} for p in points]}
    @app.post('/operations/patch-timestamps')
    async def patch(data:PatchRequest,auth=Depends(authorized)):
        try:
            with redis.lock(f'index-commit:{data.video_id}',timeout=120,blocking_timeout=3):
                return patch_timestamps(client,collection,data.video_id,data.changes,dry_run=data.dry_run)
        except IndexOperationError as exc:raise HTTPException(409,detail={'code':'index_repair_rejected','message':str(exc)})
        except Exception:raise HTTPException(503,detail={'code':'index_operations_unavailable','retryable':True})
