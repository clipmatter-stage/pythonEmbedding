"""Deterministic exact search: Redis-backed immutable continuation, bounded reads.
Point-ID order. A passage is one indexed utterance; never silently merge page edges.
Only app-managed writes are revisioned; external writers must invalidate the revision.
"""
from speaker_identity import view as identity_view, FIELDS as IDENTITY_FIELDS, matches_identity
import copy,hashlib,json,os,secrets,time
from transcript_search_contract import CompiledMatcher,VERSION,valid_interval
PROTOCOL='exact_cursor_v1'
TTL=1800
MAX_POINTS=1000
PAGE_SIZE=100
MAX_BYTES=512000
FIELDS=IDENTITY_FIELDS+['video_id','video_title','speaker','diarization_speaker','transcript_speaker','start_time','end_time','timestamp_unit','text','youtube_url','language','video_created_at']
class CursorError(Exception):
    def __init__(self,code,message,status=409):self.code=code;self.message=message;self.status=status

class RevisionedClient:
    MUTATIONS={'upsert','delete','set_payload','overwrite_payload','delete_payload','update_vectors','delete_vectors','batch_update_points'}
    def __init__(self,client,redis):self.client=client;self.redis=redis
    def __getattr__(self,name):
        method=getattr(self.client,name)
        if name not in self.MUTATIONS:return method
        def write(*args,**kwargs):
            from exact_lexical import protect_text_mutation
            args,kwargs=protect_text_mutation(name,args,kwargs)
            with self.redis.lock('exact-index-write',timeout=300,blocking_timeout=30):
                token=secrets.token_hex(16);self.redis.set('exact:index:writing',token,ex=300);self.redis.set('exact:index:revision',token)
                kwargs['wait']=True
                kwargs['timeout']=min(int(kwargs.get('timeout') or 120),120)
                try:return method(*args,**kwargs)
                except Exception:
                    self.redis.set('exact:index:uncertain',token)
                    raise
                finally:
                    self.redis.set('exact:index:revision',secrets.token_hex(16));self.redis.delete('exact:index:writing')
        return write

def revision(redis):
    if redis.exists('exact:index:uncertain'):raise CursorError('search_index_unverified','An index write had an uncertain outcome. Operator verification is required.',503)
    if redis.exists('exact:index:writing'):raise CursorError('search_index_changed','The index is being updated. Run the search again.')
    value=redis.get('exact:index:revision') or b'initial'
    return value.decode() if isinstance(value,bytes) else value

def binding(data):
    keys=['query','transcript_contract','video_id','language','speaker','time_range','filter_year','filter_month','filter_date','allowed_video_ids','authorization_scope','scope_revision']
    return hashlib.sha256(json.dumps({'rules':VERSION,'protocol':PROTOCOL,'application_revision':os.getenv('RAILWAY_GIT_COMMIT_SHA','development'),**{k:data.get(k) for k in keys}},sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def exact_page(data,client,redis,collection,models,clock=time.time):
    started=time.perf_counter();bound=binding(data);token=data.get('exact_cursor');size=min(max(int(data.get('exact_page_size') or PAGE_SIZE),1),PAGE_SIZE)
    matcher=CompiledMatcher(data.get('query',''),data.get('transcript_contract','normalized'))
    allowed=data.get('allowed_video_ids')
    if allowed is not None and (not isinstance(allowed,list) or len(allowed)>10000):raise CursorError('search_scope_invalid','Invalid authorized scope',422)
    conditions=[models.FieldCondition(key=k,match=models.MatchValue(value=v)) for k,v in [('processing_status','completed'),('approval_status','approved'),('is_archived',False)]]
    if allowed is not None:
        conditions.append(models.FieldCondition(key='video_id',match=models.MatchAny(any=allowed or [-1])))
    for key in ['video_id','language','speaker']:
        if data.get(key) is not None:conditions.append(models.FieldCondition(key=key,match=models.MatchValue(value=data[key])))
    filter_=models.Filter(must=conditions)
    current=revision(redis)
    from exact_lexical import prefilter
    if hasattr(client,'get_collection'):
        scan_filter,indexed=prefilter(client,collection,filter_,data['query'],data.get('transcript_contract','normalized'),models)
    else:scan_filter,indexed=filter_,False
    count=client.count(collection_name=collection,count_filter=filter_,exact=True,timeout=10).count
    if token:
        if not isinstance(token,str) or not token.startswith('ex1_') or len(token)!=52:raise CursorError('search_session_expired','Invalid or expired continuation.')
        raw=redis.get('exact:token:'+token)
        if raw is None:raise CursorError('search_session_expired','Search expired after 30 minutes. Run it again.')
        state=json.loads(raw)
        if state['binding']!=bound or state['size']!=size:raise CursorError('search_session_mismatch','Search or authorization scope changed. Run it again.')
        if clock()>=state['expires']:raise CursorError('search_session_expired','Search expired. Run it again.')
        if state['revision']!=current or state['index_count']!=count:raise CursorError('search_index_changed','Index changed. Run the search again.')
    else:
        state={'binding':bound,'size':size,'revision':current,'index_count':count,'offset':None,'scanned':0,'occurrences':0,'passages':0,'videos':[],'expires':clock()+TTL,'session':secrets.token_hex(16)}
        # Tokens retain no user-readable offset/filter. Store only after a successful page.
        token='ex1_'+secrets.token_hex(24)
    with redis.lock('exact:lock:'+token,timeout=90,blocking_timeout=3):
        cache_key='exact:response:'+bound+':'+hashlib.sha256(current.encode()).hexdigest()+':'+token
        cached=redis.get(cache_key)
        if cached:
            if revision(redis)!=current:raise CursorError('search_index_changed','Index changed. Run the search again.')
            result=json.loads(cached);result['search_contract']['metrics']['response_cache_hit']=True;return result
        state=copy.deepcopy(state);results=[];scanned=0;examined=0;calls=0;capacity=size;done=False;normalization=0.;read_seconds=0.;bytes_used=0
        deadline=time.perf_counter()+18
        while examined<MAX_POINTS and len(results)<size and time.perf_counter()<deadline:
            limit=min(capacity,size-len(results),MAX_POINTS-examined)
            t=time.perf_counter();points,next_offset=client.scroll(collection_name=collection,scroll_filter=scan_filter,offset=state['offset'],limit=limit,with_payload=FIELDS,with_vectors=False,timeout=10);read_seconds+=time.perf_counter()-t;calls+=1;examined+=len(points)
            if not points and next_offset is not None:raise CursorError('search_read_incomplete','Index read could not advance. Retry this page.',503)
            batch=[]
            for point in points:
                row={'id':str(point.id),**identity_view(point.payload or {})};text=row.get('text') or '';scanned+=1
                if data.get('speaker')and not matches_identity(data['speaker'],row):continue
                date=str(row.get('video_created_at') or '');prefix=data.get('filter_date') or (str(data['filter_year'])+(f"-{int(data['filter_month']):02d}" if data.get('filter_month') else '') if data.get('filter_year') else '')
                tr=data.get('time_range') or {}
                if not valid_interval(row):
                    if matcher.spans(text):raise CursorError('search_source_timing_invalid','A matching source passage has invalid timestamps and requires review.',503)
                    continue
                if (prefix and not date.startswith(prefix)) or (tr.get('start') is not None and row['start_time']<tr['start']) or (tr.get('end') is not None and row['end_time']>tr['end']):continue
                t=time.perf_counter();spans=matcher.spans(text);normalization+=time.perf_counter()-t
                if not spans:continue
                row.update(match_spans=spans,occurrence_count=len(spans),segment_ids=[row['id']],score=1,match_types=['transcript_'+data.get('transcript_contract','normalized')],matched_field='text')
                batch.append(row)
            batch_bytes=len(json.dumps(batch,ensure_ascii=False).encode())
            if batch_bytes>MAX_BYTES:
                if len(points)<=1:raise CursorError('search_page_too_large','A matching utterance exceeds the safe response size.',413)
                scanned-=len(points);capacity=max(1,limit//2);continue
            if bytes_used+batch_bytes>MAX_BYTES:
                # Nothing from this entire batch was emitted; retain its input offset.
                scanned-=len(points);break
            results.extend(batch);bytes_used+=batch_bytes;state['offset']=next_offset
            if next_offset is None:done=True;break
        if revision(redis)!=current:raise CursorError('search_index_changed','Index changed while reading. Run the search again.')
        state['scanned']+=scanned;state['occurrences']+=sum(r['occurrence_count'] for r in results);state['passages']+=len(results);state['videos']=sorted(set(state['videos'])|{r['video_id'] for r in results})
        if len(state['videos'])>10000:raise CursorError('search_scope_too_large','Search scope exceeds the bounded authorization registry.',503)
        counts={'matching_videos':len(state['videos']),'displayed_passages':state['passages'],'matched_transcript_occurrences':state['occurrences'],'merged_passages':None}
        next_token=None if done else 'ex1_'+secrets.token_hex(24)
        metadata={'authority':'python_contract_v1','representation':VERSION,'protocol':PROTOCOL,'contract':data.get('transcript_contract','normalized'),'status':('completed' if state['passages'] else 'no_matches') if done else 'partial','exhaustive':done,'counts_final':done,'counts':counts,'scanned_segments':state['scanned'],'scope':'authorized_eligible_indexed_segments','index_revision':current,'lexical_index_used':indexed,'eligible_index_points':count,'passage_unit':'indexed_source_utterance','merged_passages_note':'Cross-utterance merging is not performed; counts are not merged-window totals.','expires_at':state['expires'],'metrics':{'points_examined':examined,'read_calls':calls,'retrieval_ms':round(read_seconds*1000,2),'normalization_ms':round(normalization*1000,2),'elapsed_ms':round((time.perf_counter()-started)*1000,2),'result_payload_bytes':bytes_used,'provider_calls':0,'response_cache_hit':False}}
        result={'results':results,'returned':len(results),'query':data['query'],'search_mode':'simple','filter_type':'text','search_contract':metadata,'cursor':{'next':next_token,'has_more':not done},'search_session_id':state['session']}
        ttl=max(1,int(state['expires']-clock()))
        # Response and next state are committed together; retries replay the same page.
        pipe=redis.pipeline(transaction=True)
        pipe.set('exact:token:'+token,json.dumps(state if not data.get('exact_cursor') else json.loads(raw)),ex=ttl)
        if next_token:pipe.set('exact:token:'+next_token,json.dumps(state),ex=ttl)
        pipe.set(cache_key,json.dumps(result,ensure_ascii=False),ex=ttl);pipe.execute()
        return result

def register_exact_search(app,client,redis,collection,models,authorize,api_key=None):
    from fastapi import Depends,HTTPException,Request
    import hmac
    from pydantic import BaseModel,Field
    from typing import Optional
    async def strict_auth(request:Request,allowed=Depends(authorize)):
        if not api_key:raise HTTPException(503,detail='Exact search authentication is not configured')
        if not hmac.compare_digest(request.headers.get('x-api-key',''),api_key):raise HTTPException(401,detail='Unauthorized')
    class ExactRequest(BaseModel):
        query:str=Field(min_length=1,max_length=1000)
        transcript_contract:str=Field(default='normalized',pattern='^(literal|normalized|alias)$')
        exact_cursor:Optional[str]=Field(default=None,max_length=100)
        exact_page_size:int=Field(default=100,ge=1,le=100)
        video_id:Optional[int]=None
        language:Optional[str]=None
        speaker:Optional[str]=None
        time_range:Optional[dict]=None
        filter_year:Optional[int]=None
        filter_month:Optional[int]=None
        filter_date:Optional[str]=None
        allowed_video_ids:Optional[list[int]]=Field(default=None,max_length=10000)
        authorization_scope:Optional[str]=None
        scope_revision:Optional[str]=None
    class InvalidateRequest(BaseModel):
        expected_revision:str=Field(max_length=100)
        confirmed_stable:bool
    @app.post('/operations/exact-invalidate')
    async def invalidate(data:InvalidateRequest,authorized:bool=Depends(strict_auth)):
        with redis.lock('exact-index-write',timeout=300,blocking_timeout=3):
            current=redis.get('exact:index:revision') or b'initial'
            current=current.decode() if isinstance(current,bytes) else current
            if not data.confirmed_stable or current!=data.expected_revision or redis.exists('exact:index:writing'):raise HTTPException(409,detail='Index stability/revision not confirmed')
            redis.set('exact:index:revision',secrets.token_hex(16));redis.delete('exact:index:uncertain')
            return {'invalidated':True,'revision':revision(redis)}
    @app.get('/operations/exact-capabilities')
    async def capabilities(authorized:bool=Depends(strict_auth)):
        import os
        return {'protocol':PROTOCOL,'rules_version':VERSION,'revision_tracking':isinstance(client,RevisionedClient),'revision':revision(redis),'application_revision':os.getenv('RAILWAY_GIT_COMMIT_SHA','unknown'),'cursor_ttl_seconds':TTL,'request_max_points':MAX_POINTS,'response_max_passages':PAGE_SIZE}
    @app.post('/search/exact')
    async def exact(data:ExactRequest,authorized:bool=Depends(strict_auth)):
        import asyncio
        try:return await asyncio.to_thread(exact_page,data.model_dump(),client,redis,collection,models)
        except CursorError as e:raise HTTPException(e.status,detail={'code':e.code,'message':e.message,'retryable':True})
        except ValueError as e:raise HTTPException(422,detail={'code':'invalid_exact_query','message':str(e)})
        except HTTPException:raise
        except Exception:raise HTTPException(503,detail={'code':'exact_search_unavailable','message':'Exact search is temporarily unavailable. Retry this page.','retryable':True})


async def exact_response(data,client,redis,collection,models):
    import asyncio
    from fastapi import HTTPException
    try:return await asyncio.to_thread(exact_page,data,client,redis,collection,models)
    except CursorError as e:raise HTTPException(e.status,detail={'code':e.code,'message':e.message,'retryable':True})
    except ValueError as e:raise HTTPException(422,detail={'code':'invalid_exact_query','message':str(e)})
    except HTTPException:raise
    except Exception:raise HTTPException(503,detail={'code':'exact_search_unavailable','message':'Exact search is temporarily unavailable. Retry this page.','retryable':True})
