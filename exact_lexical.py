"""Versioned exact-search keyword prefilter and checkpointed payload-only repair.
No vector writes, text edits, point deletion or producer relabeling.
"""
import hashlib,json,re
from transcript_search_contract import VERSION,normalize,alias_with_spans
INDEX_VERSION='exact_terms_v1:'+VERSION
GUARD_FIELDS=['text','video_id','start_time','end_time','duration','timestamp_unit','speaker','transcript_speaker','diarization_speaker','segment_index','processing_status','approval_status','is_archived','indexing_run_id','source_snapshot_sha256','source_segment_id','embedding_model','embedding_dimensions','recovery_contract']
GUARD_VERSION='exact_payload_guard_v1'
KEYS=['exact_terms_version','exact_normalized_terms_v1','exact_alias_terms_v1','exact_source_sha256']
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def derived_fields(text):
    text=str(text or '')
    return {'exact_terms_version':INDEX_VERSION,'exact_source_sha256':hashlib.sha256(text.encode()).hexdigest(),'exact_normalized_terms_v1':sorted(set(re.findall(r'\w+',normalize(text)))),'exact_alias_terms_v1':sorted(set(re.findall(r'\w+',alias_with_spans(text)[0])))}
def prefilter(client,collection,base,query,contract,models):
    if contract not in {'normalized','alias'}:return base,False
    words=re.findall(r'\w+',alias_with_spans(query)[0]if contract=='alias'else normalize(query))
    if not words:return base,False
    schema=client.get_collection(collection).payload_schema
    field='exact_alias_terms_v1'if contract=='alias'else 'exact_normalized_terms_v1'
    if any(getattr(getattr(schema.get(k),'data_type',None),'value',getattr(schema.get(k),'data_type',None))!='keyword' for k in ['exact_terms_version',field]):return base,False
    covered=models.Filter(must=[*base.must,models.FieldCondition(key='exact_terms_version',match=models.MatchValue(value=INDEX_VERSION))])
    if client.count(collection_name=collection,count_filter=base,exact=True,timeout=10).count!=client.count(collection_name=collection,count_filter=covered,exact=True,timeout=10).count:return base,False
    return models.Filter(must=[*base.must,*[models.FieldCondition(key=field,match=models.MatchValue(value=w))for w in sorted(set(words))]]),True

def read_video(client,collection,video_id,models):
    out=[];offset=None
    while True:
        rows,offset=client.scroll(collection_name=collection,scroll_filter=models.Filter(must=[models.FieldCondition(key='video_id',match=models.MatchValue(value=video_id))]),offset=offset,limit=200,with_payload=GUARD_FIELDS+KEYS,with_vectors=False,timeout=10);out.extend(rows)
        if len(out)>5000:raise ValueError('Explicit video point bound exceeded')
        if offset is None:return out

def plan_video(client,collection,video_id,models):
    points=read_video(client,collection,video_id,models);items=[]
    for p in points:
        payload=p.payload or {};new=derived_fields(payload.get('text'));before={k:payload[k]for k in KEYS if k in payload}
        if before==new:continue
        after={**payload,**new};items.append({'id':str(p.id),'before_fields':before,'before_guard_sha256':digest(payload),'after_guard_sha256':digest(after),'source_sha256':new['exact_source_sha256']})
    return {'version':INDEX_VERSION,'guard_version':GUARD_VERSION,'video_id':video_id,'total_points':len(points),'items':items,'plan_sha256':digest(items)}

def apply_video(client,collection,plan,models,rollback=False):
    if plan.get('guard_version')!=GUARD_VERSION or plan.get('version')!=INDEX_VERSION or digest(plan.get('items'))!=plan.get('plan_sha256'):raise ValueError('Checkpoint version/hash mismatch')
    rows=read_video(client,collection,plan['video_id'],models);points={str(p.id):p for p in rows}
    if len(points)!=plan['total_points']:raise ValueError('Point coverage changed')
    if len({i.get('id')for i in plan['items']})!=len(plan['items']):raise ValueError('Duplicate checkpoint IDs')
    if any(not set(i.get('before_fields',{})).issubset(KEYS)for i in plan['items']):raise ValueError('Checkpoint contains nonlexical fields')
    operations=[];already_current=0
    for item in plan['items']:
        p=points.get(item['id']);payload=p.payload if p else None
        if payload is None:raise ValueError('Checkpoint point missing')
        expected=item['after_guard_sha256']if rollback else item['before_guard_sha256']
        already=item['before_guard_sha256']if rollback else item['after_guard_sha256']
        if digest(payload)==already:
            already_current+=1;continue
        if digest(payload)!=expected or derived_fields(payload.get('text'))['exact_source_sha256']!=item['source_sha256']:raise ValueError('Source or checkpoint payload changed')
        fields=item['before_fields']if rollback else derived_fields(payload.get('text'))
        if fields:operations.append(models.SetPayloadOperation(set_payload=models.SetPayload(payload=fields,points=[p.id])))
        missing=[k for k in KEYS if k not in fields]
        if missing:operations.append(models.DeletePayloadOperation(delete_payload=models.DeletePayload(keys=missing,points=[p.id])))
    for i in range(0,len(operations),100):client.batch_update_points(collection_name=collection,update_operations=operations[i:i+100],wait=True,timeout=30)
    after={str(p.id):p for p in read_video(client,collection,plan['video_id'],models)}
    for item in plan['items']:
        expected=item['before_guard_sha256']if rollback else item['after_guard_sha256']
        if item['id']not in after or digest(after[item['id']].payload)!=expected:raise ValueError('Payload verification failed; retain checkpoint and inspect uncertain writes')
    return {'video_id':plan['video_id'],'updated_points':len(plan['items'])-already_current,'already_current_points':already_current,'verified_points':len(plan['items']),'rollback':rollback,'verified':True,'vectors_written':0}

def register_operations(app,client,redis,collection,models,authorized):
    from fastapi import Depends,HTTPException
    from pydantic import BaseModel,Field
    class Apply(BaseModel):
        checkpoint:dict
        rollback:bool=False
    @app.get('/operations/exact-lexical-plan')
    async def plan(video_id:int,auth=Depends(authorized)):
        if video_id<1:raise HTTPException(422,detail='Positive video ID required')
        import asyncio
        return await asyncio.to_thread(plan_video,client,collection,video_id,models)
    @app.post('/operations/exact-lexical-apply')
    async def apply(data:Apply,auth=Depends(authorized)):
        import asyncio
        def execute():
            video_id=data.checkpoint.get('video_id')
            if type(video_id)is not int or video_id<1 or len(data.checkpoint.get('items',[]))>5000:raise ValueError('Invalid checkpoint scope')
            with redis.lock(f'index-commit:{video_id}',timeout=180,blocking_timeout=3):return apply_video(client,collection,data.checkpoint,models,data.rollback)
        try:return await asyncio.to_thread(execute)
        except ValueError as e:raise HTTPException(409,detail={'code':'exact_index_checkpoint_rejected','message':str(e)})
        except Exception:raise HTTPException(503,detail={'code':'exact_index_update_unavailable','retryable':True})
    @app.post('/operations/exact-lexical-indexes')
    async def indexes(auth=Depends(authorized)):
        import asyncio
        def create():
            for key in ['exact_terms_version','exact_normalized_terms_v1','exact_alias_terms_v1']:
                client.create_payload_index(collection_name=collection,field_name=key,field_schema=models.PayloadSchemaType.KEYWORD,wait=True,timeout=30)
            return {'version':INDEX_VERSION,'fields':['exact_terms_version','exact_normalized_terms_v1','exact_alias_terms_v1'],'vectors_written':0}
        return await asyncio.to_thread(create)

def protect_text_mutation(name,args,kwargs):
    """Keep exact fields coherent for known managed text writers, including batches."""
    args=list(args);kwargs=dict(kwargs)
    def update_payload(payload):
        return {**payload,**derived_fields(payload['text'])}if isinstance(payload,dict)and 'text'in payload else payload
    argument='payload'if name in {'set_payload','overwrite_payload'}else 'points'if name=='upsert'else 'update_operations'if name=='batch_update_points'else None
    if argument is None:return tuple(args),kwargs
    value=kwargs.get(argument,args[1]if len(args)>1 else None)
    if argument=='payload':updated=update_payload(value)
    elif isinstance(value,list):
        updated=[]
        for item in value:
            if argument=='points' and getattr(item,'payload',None)and 'text'in item.payload:item=item.model_copy(update={'payload':update_payload(item.payload)})
            elif argument=='update_operations'and getattr(item,'set_payload',None):
                operation=item.set_payload;item=item.model_copy(update={'set_payload':operation.model_copy(update={'payload':update_payload(operation.payload)})})
            updated.append(item)
    else:return tuple(args),kwargs
    if argument in kwargs:kwargs[argument]=updated
    elif len(args)>1:args[1]=updated
    return tuple(args),kwargs
