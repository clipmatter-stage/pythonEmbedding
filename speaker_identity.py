"""Identity metadata is distinct from anonymous labels and legacy predictions."""
import copy,hashlib,json,re
VERSION='speaker_identity_v2'
UNKNOWN='Unknown Speaker'
FIELDS=['identity_version','identity_status','identity_source','identity_key','identity_review','identity_enrollment_fingerprint','identity_calibration_sha256','identity_acceptance_policy','identity_run_id','identity_name','identity_revision']
def supported(row):
    if row.get('identity_version')!=VERSION or not row.get('identity_key'):return False
    review=row.get('identity_review')or{}
    if row.get('identity_status')=='verified' and row.get('identity_source')=='manual':
        return isinstance(review,dict) and all(review.get(k)for k in ['review_id','reviewer_id','reviewed_at']) and bool(re.fullmatch('[a-f0-9]{64}',str(review.get('evidence_sha256',''))))
    return row.get('identity_status')=='accepted' and all(row.get(k)for k in ['identity_enrollment_fingerprint','identity_run_id']) and bool(row.get('identity_calibration_sha256') or (row.get('identity_source')=='provider' and row.get('identity_acceptance_policy')=='pyannote_match_v1'))
def unknown_label(row):
    label=str(row.get('transcript_speaker') or row.get('diarization_speaker') or row.get('source_speaker') or row.get('speaker') or '')
    if re.fullmatch('[A-Z]{1,3}',label):return 'Unknown '+label
    match=re.fullmatch(r'(?:SPEAKER[_ -]?)?(\d+)',label,re.I)
    if not match or int(match[1])>10000:return UNKNOWN
    n=int(match[1]);suffix=''
    while True:
        suffix=chr(65+n%26)+suffix;n=n//26-1
        if n<0:break
    return 'Unknown '+suffix

def view(row):
    out=dict(row)
    name=(row.get('identity_name')or row.get('speaker')or UNKNOWN)if supported(row)else UNKNOWN
    if supported(row)and row.get('identity_key')=='hafiz_naeem_ur_rehman':name='Hafiz Naeem Ur Rehman'
    out['source_identity_group']=row.get('source_identity_group')or hashlib.sha256(json.dumps([row.get('video_id'),row.get('transcript_speaker')or row.get('diarization_speaker'),row.get('speaker')],ensure_ascii=False).encode()).hexdigest()
    out.update(speaker=name,speaker_display_name=unknown_label(row) if name==UNKNOWN else name,identity_name=name,identity_key=row.get('identity_key')if supported(row)else None,identity_status=row.get('identity_status')if supported(row)else 'unknown',identity_version=VERSION)
    return out

def patch_identity(client,collection,video_id,changes,*,rollback=False,dry_run=True):
    from index_operations import payload_hash
    if not 1<=len(changes)<=50:raise ValueError('Select 1-50 explicit point IDs')
    ids=[x['point_id']for x in changes]
    if len(set(map(str,ids)))!=len(ids):raise ValueError('Duplicate points')
    points={str(p.id):p for p in client.retrieve(collection_name=collection,ids=ids,with_payload=True,with_vectors=False,timeout=10)}
    before={};updates=[]
    for c in changes:
        p=points.get(str(c['point_id']));payload=p.payload if p else None
        if not payload or payload.get('video_id')!=video_id:raise ValueError('Point scope mismatch')
        fields=dict(c.get('fields',{}));delete=list(c.get('delete_fields',[]))
        if not fields and not delete or not set(fields).issubset(FIELDS) or not set(delete).issubset(FIELDS)or set(fields)&set(delete):raise ValueError('Identity fields only')
        # Remote Qdrant omits null payload values whereas the local client can
        # retain them. Express clearing explicitly so hashes match both stores.
        delete.extend(k for k,v in fields.items() if v is None)
        fields={k:v for k,v in fields.items() if v is not None}
        after={k:v for k,v in {**payload,**fields}.items()if k not in delete}
        if payload_hash(payload)==c.get('post_payload_sha256'):continue
        if payload_hash(payload)!=c['expected_payload_sha256']:raise ValueError('Source/run/payload changed')
        if not rollback:
            if fields.get('identity_version')!=VERSION or fields.get('identity_status')not in {'unknown','accepted','verified'}:raise ValueError('Invalid identity decision')
            if fields.get('identity_status')!='unknown':
                if not supported(after):raise ValueError('Unreviewed identity')
                if payload.get('embedding_input_contract')!='transcript_context_text_only_v1':raise ValueError('Embedded identity provenance unavailable; staged replacement required')
            elif fields.get('identity_name')!=UNKNOWN or fields.get('identity_key')is not None:raise ValueError('Unknown must not claim identity')
        before[str(p.id)]={'payload_sha256':payload_hash(payload),'fields':{k:payload[k]for k in FIELDS if k in payload}}
        updates.append((p.id,fields,delete,payload_hash(after)))
    if not dry_run:
        for point_id,fields,delete,_ in updates:
            if fields:client.set_payload(collection_name=collection,payload=fields,points=[point_id],wait=True,timeout=20)
            if delete:client.delete_payload(collection_name=collection,keys=delete,points=[point_id],wait=True,timeout=20)
        actual={str(p.id):p.payload for p in client.retrieve(collection_name=collection,ids=ids,with_payload=True,with_vectors=False,timeout=10)}
        if any(payload_hash(actual[str(i)])!=h for i,_,_,h in updates):raise ValueError('Uncertain identity write; retain checkpoint and inspect')
    return {'version':VERSION,'video_id':video_id,'dry_run':dry_run,'updated_points':len(updates),'verified':not dry_run,'vectors_written':0,'before':before,'after_hashes':{**{str(c['point_id']):c['post_payload_sha256']for c in changes if c.get('post_payload_sha256')},**{str(i):h for i,_,_,h in updates}}}

def register(app,client,collection,redis,authorized):
    import asyncio
    from fastapi import Depends,HTTPException
    from pydantic import BaseModel,Field
    class Patch(BaseModel):
        video_id:int=Field(gt=0)
        changes:list[dict]=Field(min_length=1,max_length=50)
        dry_run:bool=True
        rollback:bool=False
    @app.post('/operations/patch-speaker-identities')
    async def patch(data:Patch,auth=Depends(authorized)):
        def execute():
            with redis.lock(f'index-commit:{data.video_id}',timeout=180,blocking_timeout=3):return patch_identity(client,collection,data.video_id,data.changes,dry_run=data.dry_run,rollback=data.rollback)
        try:return await asyncio.to_thread(execute)
        except ValueError as e:raise HTTPException(409,detail={'code':'speaker_identity_patch_rejected','message':str(e)})
        except Exception:raise HTTPException(503,detail={'code':'speaker_identity_patch_unavailable','retryable':True})

class SearchIdentityClient:
    """Read projection for search only; operation checkpoints always use raw client."""
    def __init__(self,client):self.client=client
    def __getattr__(self,name):
        method=getattr(self.client,name)
        if name not in {'scroll','retrieve','query_points','search'}:return method
        def read(*args,**kwargs):
            result=method(*args,**kwargs)
            def safe(p):return p.model_copy(update={'payload':view(p.payload or{})})if hasattr(p,'model_copy')else type(p)(**{**vars(p),'payload':view(p.payload or{})})
            if name=='scroll':return [safe(p)for p in result[0]],result[1]
            if name=='query_points':return result.model_copy(update={'points':[safe(p)for p in result.points]})
            return [safe(p)for p in result]
        return read

def speaker_search(client,collection,request,matches):
    from contextual_search import source_filter
    from transcript_search_contract import valid_interval
    from qdrant_client import models
    speaker=(request.speaker or request.query or'').strip()
    if not speaker:raise ValueError('Speaker name required; Unknown Speaker is a distinct filter')
    unknown=speaker.casefold()in {'unknown','unknown speaker'}
    filters=source_filter(request)
    # Strict-mode Qdrant rejects filtering on fields without payload indexes.
    # Preserve the bounded eligible scan until both derived identity indexes
    # exist; supported() still decides every returned identity below.
    indexed=False
    if not unknown and hasattr(client,'get_collection'):
        schema=client.get_collection(collection_name=collection).payload_schema or {}
        indexed=all(k in schema for k in ('identity_version','identity_status'))
    if indexed:filters.must.extend([models.FieldCondition(key='identity_version',match=models.MatchValue(value=VERSION)),models.FieldCondition(key='identity_status',match=models.MatchAny(any=['accepted','verified']))])
    offset=None;rows=[];scanned=0;limit=min(request.max_scanned,50000);next_offset=None
    while scanned<limit and len(rows)<request.top_k:
        points,next_offset=client.scroll(collection_name=collection,scroll_filter=filters,offset=offset,limit=min(200,limit-scanned),with_payload=True,with_vectors=False,timeout=10)
        for p in points:
            scanned+=1;r=view(p.payload or{})
            if not valid_interval(r):continue
            if (r['speaker']==UNKNOWN if unknown else matches_identity(speaker,r)):
                rows.append({**r,'id':str(p.id),'segment_ids':[str(p.id)],'score':1,'matched_field':'speaker','match_types':['verified_identity_filter'if not unknown else'unknown_identity_filter'],'duration':r['end_time']-r['start_time']})
        if next_offset is None:break
        offset=next_offset
    partial=next_offset is not None or len(rows)>request.top_k
    return {'results':rows[:request.top_k],'returned':len(rows[:request.top_k]),'search_mode':'simple','filter_type':'speaker','speaker_filter':speaker,'unique_videos':len({r['video_id']for r in rows[:request.top_k]}),'metadata':{'identity_version':VERSION,'identity_scope':'supported_identity_or_explicit_unknown','scanned':scanned,'matched_passages_in_scanned_scope':len(rows),'partial_reasons':['bounded_identity_scan']if partial else[],'status':'partial'if partial else'completed'if rows else'no_matches','exhaustive':not partial,'counts_final':not partial}}

def matches_identity(query,row):
    r=view(row);q=' '.join(str(query or'').casefold().replace('-',' ').split())
    if q in {'unknown','unknown speaker'}:return r['speaker']==UNKNOWN
    if not supported(row):return False
    if q in {'hnr','hafiz naeem','hafiz naeem rehman','hafiz naeem ur rehman','hafiz naeem ur rahman','حافظ نعیم','حافظ نعیم الرحمن'}:return row.get('identity_key')=='hafiz_naeem_ur_rehman'
    import re
    return bool(q)and set(re.findall(r'\w+',q)).issubset(set(re.findall(r'\w+',r['speaker'].casefold())))
