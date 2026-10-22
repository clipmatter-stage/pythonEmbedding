"""One authenticated callback transport for both API/worker entry points.
Credentials never enter logs, results, URLs, or canary jobs.
"""

import base64,json,hashlib,hmac,os,time,urllib.parse,inspect
import requests

def issue_callback_token(key,claims,ttl=14400):
    if not key:raise ValueError('Callback signing key is not configured')
    encoded=base64.urlsafe_b64encode(json.dumps({**claims,'exp':int(time.time())+ttl},separators=(',',':')).encode()).decode().rstrip('=')
    return encoded+'.'+hmac.new(key.encode(),('callback-v1.'+encoded).encode(),hashlib.sha256).hexdigest()

def attach_callback_token(data,key):
    if data.get('webhook_url') and data.get('run_id'):
        data['_callback_token']=issue_callback_token(key,{'scope':'embedding','video_id':data['video_id'],'run_id':data['run_id']})
    return data

def post_callback(url,payload,key,callback_token=None):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or parsed.username or parsed.password:raise ValueError('Callbacks require an HTTPS destination')
    headers={'Accept':'application/json'}
    if callback_token:headers['Authorization']='Bearer '+callback_token
    else:headers['X-API-Key']=key or ''
    response=requests.post(url,json=payload,headers=headers,timeout=20,allow_redirects=False)
    if 300<=response.status_code<400:
        target=urllib.parse.urlsplit(urllib.parse.urljoin(url,response.headers.get('Location','')))
        if target.scheme!='https' or target.netloc!=parsed.netloc or target.path!=parsed.path:raise ValueError('Callback redirect changed origin or endpoint')
        response=requests.post(target.geturl(),json=payload,headers=headers,timeout=20,allow_redirects=False)
    return response

def deliver(url,payload,key,error_type,max_attempts=5,callback_token=None):
    for attempt in range(max_attempts):
        try:
            response=post_callback(url,payload,key,callback_token)
            if 200<=response.status_code<300:return
            if response.status_code in (401,403,409,422):raise error_type('Callback rejected: HTTP '+str(response.status_code))
        except ValueError:raise error_type('Callback destination policy rejected')
        except requests.RequestException:pass
        if attempt+1<max_attempts:time.sleep(min(2**attempt,8))
    raise error_type('Callback delivery unavailable after bounded attempts')

def worker_canary(data,key,module,revision_tracking=False):
    nonce=data['challenge'];signature=hmac.new((key or '').encode(),nonce.encode(),hashlib.sha256).hexdigest()
    from rq import get_current_job
    current_job=get_current_job()
    result={'index_revision_tracking':revision_tracking,'worker_module':module,'worker_name':getattr(current_job,'worker_name',None),'speaker_identity_version':'speaker_identity_v2','search_pipeline':os.getenv('SEARCH_PIPELINE','fast'),'grounding_citation_mode':os.getenv('GROUNDING_CITATION_MODE','span_id'),'worker_revision':os.getenv('RAILWAY_GIT_COMMIT_SHA','unknown'),'api_key_configured':bool(key),'configured_key_matches_expected':hmac.compare_digest(signature,data['expected_signature']),'transport_sha256':hashlib.sha256(inspect.getsource(post_callback).encode()).hexdigest(),'provider_calls':0}
    try:
        legacy=post_callback(data['callback_url'],{'probe_nonce':nonce,'probe_signature':signature},key)
        try:legacy_body=legacy.json()
        except ValueError:legacy_body={}
        result['legacy_key_probe']={'http':legacy.status_code,**{k:legacy_body[k] for k in ['authenticated','header_present','body_proof_matches'] if k in legacy_body}}
        response=post_callback(data['callback_url'],{'probe_nonce':nonce,'probe_signature':signature},key,data.get('_callback_token'))
        result['http']=response.status_code
        try:body=response.json()
        except ValueError:body={}
        result['destination_observation']={k:body[k] for k in ['authenticated','authentication_method','header_present','bearer_present','body_proof_matches','probe_nonce'] if k in body}
    except Exception as exc:result['http']=None;result['transport_error_type']=type(exc).__name__
    return result

def register_callback_probe(app,queue,redis,api_key,worker_function):
    from fastapi import Depends,HTTPException,Request
    from pydantic import BaseModel,Field
    async def auth(request:Request):
        if not api_key or not hmac.compare_digest(request.headers.get('x-api-key',''),api_key):raise HTTPException(401,detail='Unauthorized')
    class Probe(BaseModel):
        callback_url:str=Field(max_length=500)
        challenge:str=Field(pattern='^[a-f0-9]{32}$')
        expected_signature:str=Field(pattern='^[a-f0-9]{64}$')
    @app.get('/operations/callback-queue-audit')
    async def queue_audit(authorized=Depends(auth)):
        from rq.job import Job
        from rq import Worker
        from rq.registry import StartedJobRegistry,FailedJobRegistry
        registries={'queued':queue,'started':StartedJobRegistry(queue=queue),'failed':FailedJobRegistry(queue=queue)}
        output={'registries':{},'workers':[],'api_key_configured':bool(api_key),'inspection_limit_per_registry':100}
        for state,registry in registries.items():
            ids=registry.get_job_ids(offset=0,length=100)if state=='queued'else registry.get_job_ids(start=0,end=99,cleanup=False);rows=[]
            for job_id in ids:
                try:
                    job=Job.fetch(job_id,connection=redis);data=job.args[0]if job.args and isinstance(job.args[0],dict)else {}
                    rows.append({'job_id':job_id,'function':job.func_name,'video_id':data.get('video_id'),'run_id':data.get('run_id'),'operation':data.get('operation'),'has_scoped_callback_token':bool(data.get('_callback_token')),'retained_callback_401_hint':bool(job.exc_info and '401'in job.exc_info and 'callback'in job.exc_info.lower())})
                except Exception:rows.append({'job_id':job_id,'inspection_failed':True})
            total=redis.llen(registry.key)if state=='queued'else redis.zcard(registry.key)
            output['registries'][state]={'total':total,'inspected':len(rows),'complete':len(rows)>=total,'jobs':rows}
        for worker in Worker.all(connection=redis):output['workers'].append({'name':worker.name,'queues':worker.queue_names(),'state':str(worker.get_state()),'revision':'unverified_until_processing_canary'})
        return output
    @app.post('/operations/callback-probe')
    async def submit(data:Probe,authorized=Depends(auth)):
        url=urllib.parse.urlsplit(data.callback_url)
        allowed=set(os.getenv('CALLBACK_ALLOWED_HOSTS','clip.digitalmatters.pk').split(','))
        if url.scheme!='https' or url.hostname not in allowed or url.port not in (None,443) or url.path!='/api/embeddings/callback/probe' or url.query or url.fragment or url.username or url.password:raise HTTPException(422,detail='Canary destination is not allowed')
        job_data={'operation':'callback_probe',**data.model_dump(),'_callback_token':issue_callback_token(api_key,{'scope':'probe','nonce':data.challenge},600)}
        job=queue.enqueue(worker_function,job_data,job_timeout=60,result_ttl=600,failure_ttl=600)
        return {'job_id':job.id}
    @app.get('/operations/callback-probe/{job_id}')
    async def result(job_id:str,authorized=Depends(auth)):
        from rq.job import Job
        job=Job.fetch(job_id,connection=redis)
        if job.func_name!=worker_function or not job.args or job.args[0].get('operation')!='callback_probe':raise HTTPException(403,detail='Not a callback canary')
        status=job.get_status();status=getattr(status,'value',status)
        return {'status':status,'result':job.result if status=='finished' else None}
