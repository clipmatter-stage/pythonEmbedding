"""Verified additive recovery inside the existing staged embedding worker.
Existing points are never overwritten. Replacements require separate activation safeguards.
"""
import hashlib,json,math
VERSION='transcript_recovery_v1'
class RecoveryRejected(RuntimeError):pass

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def baseline_digest(points):
    return hashlib.sha256(canonical(sorted([{'id':str(p.id),'payload_sha256':hashlib.sha256(canonical(p.payload).encode()).hexdigest()} for p in points],key=lambda p:p['id'])).encode()).hexdigest()
SHORT_LEXICAL={'کا','کو','ان','کی','کے','ہے','ہی','ہم','وہ','یہ','AI','IT','US','UK','a','I'}
def lexical(text):
    if not isinstance(text,str):return False
    value=text.strip()
    return any(c.isalnum() for c in value) and (len(value)>=3 or value.isnumeric() or value in SHORT_LEXICAL)
def valid_vector(vector,dimension):
    return (isinstance(vector,list) and len(vector)==dimension and all(type(x) in (int,float) and math.isfinite(x) for x in vector) and any(x!=0 for x in vector))
def validate_point(point,contract,video_id,run_id,model,dimensions):
    p=point.payload
    if not valid_vector(point.vector,dimensions):raise RecoveryRejected('Invalid/nonfinite/zero embedding vector')
    if p.get('video_id')!=video_id or p.get('indexing_run_id')!=run_id:raise RecoveryRejected('Video/generation mismatch')
    if p.get('embedding_model')!=model or p.get('embedding_dimensions')!=dimensions:raise RecoveryRejected('Embedding space mismatch')
    if p.get('source_snapshot_sha256')!=contract['source_sha256'] or p.get('source_segment_id')!=f"{video_id}:{p.get('segment_index')}":raise RecoveryRejected('Source identity mismatch')
    if p.get('recovery_contract')!=VERSION or p.get('user_id')!=contract['user_id']:raise RecoveryRejected('Recovery/access metadata mismatch')
    if p.get('processing_status')!='completed' or p.get('approval_status')!='approved' or p.get('is_archived') is not False:raise RecoveryRejected('Ineligible payload')
    a=p.get('start_time');b=p.get('end_time')
    if p.get('timestamp_unit')!='seconds' or any(type(x) not in (int,float) or not math.isfinite(x) for x in [a,b]) or not 0<=a<b:raise RecoveryRejected('Invalid seconds interval')
    if not lexical(p.get('text')) or not isinstance(p.get('speaker'),str) or not p.get('speaker'):raise RecoveryRejected('Invalid transcript/speaker field')
    if abs(p.get('duration',-1)-(b-a))>.001:raise RecoveryRejected('Duration mismatch')

def snapshot(client,collection,filter_):
    result=[];offset=None
    for _ in range(100):
        page,offset=client.scroll(collection_name=collection,scroll_filter=filter_,limit=100,offset=offset,with_payload=True,with_vectors=True,timeout=10)
        result.extend(page)
        if offset is None:return result
    raise RecoveryRejected('Per-video snapshot bound exceeded')

def stable_payload(payload):return {k:v for k,v in payload.items() if k!='created_at'}

def commit_recovery(client,collection,points,contract,*,video_id,run_id,model,dimensions,filter_,is_current,delete_owned):
    """Caller holds index-commit lock. All vectors are prepared before this function."""
    if contract.get('version')!=VERSION or not is_current():raise RecoveryRejected('Retired/stale recovery generation')
    expected=contract.get('source_indices',[])
    if not points or len(points)>2000 or len(points)!=len(expected) or len(set(expected))!=len(expected):raise RecoveryRejected('Incomplete staged recovery')
    if set(p.payload.get('segment_index') for p in points)!=set(expected) or len({str(p.id) for p in points})!=len(points):raise RecoveryRejected('Duplicate/missing source identities')
    for p in points:validate_point(p,contract,video_id,run_id,model,dimensions)
    before=snapshot(client,collection,filter_);own=[p for p in before if p.payload.get('indexing_run_id')==run_id];prior=[p for p in before if p.payload.get('indexing_run_id')!=run_id]
    if baseline_digest(prior)!=contract['baseline_payload_digest']:raise RecoveryRejected('Index changed since checkpoint')
    ids={str(p.id) for p in points};prior_ids={str(p.id) for p in prior}
    if ids&prior_ids:raise RecoveryRejected('Recovery cannot overwrite existing points')
    if any(str(p.id) not in ids for p in own):raise RecoveryRejected('Unplanned points in recovery generation')
    wanted={str(p.id):p for p in points}
    for p in own:
        validate_point(p,contract,video_id,run_id,model,dimensions)
        if stable_payload(p.payload)!=stable_payload(wanted[str(p.id)].payload):raise RecoveryRejected('Resumed generation content differs')
    missing=[p for p in points if str(p.id) not in {str(x.id) for x in own}]
    try:
        for i in range(0,len(missing),100):
            if not is_current():raise RecoveryRejected('Stale generation before write')
            client.upsert(collection_name=collection,points=missing[i:i+100],wait=True,timeout=20)
        actual=snapshot(client,collection,filter_)
        if not is_current() or {str(p.id) for p in actual}!=prior_ids|ids:raise RecoveryRejected('Post-write coverage/generation mismatch')
        current_prior=[p for p in actual if str(p.id) in prior_ids]
        if baseline_digest(current_prior)!=contract['baseline_payload_digest']:raise RecoveryRejected('Existing payload changed during recovery')
        for p in actual:
            if str(p.id) in ids:
                validate_point(p,contract,video_id,run_id,model,dimensions)
                if stable_payload(p.payload)!=stable_payload(wanted[str(p.id)].payload):raise RecoveryRejected('Post-write content mismatch')
    except Exception:
        # Callback must never report completion. Guarded cleanup cannot delete newer-run data.
        delete_owned(list(ids),run_id)
        raise
    return {'recovery_contract':VERSION,'coverage_verified':True,'new_points':len(points),'total_points':len(actual),'resumed_existing_points':len(own)}
