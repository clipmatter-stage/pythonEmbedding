"""Diverse semantic candidates -> original nearby turns -> one grounded rerank.
No embedding-space change or new identity assertions. Exact matching is separate.
"""
import copy,hashlib,json,time
from collections import defaultdict
from qdrant_client import models
from contextual_search import source_filter,source_passage
from grounded_validation import VERSION,SYSTEM,validate_grounded,candidate_id
from semantic_passage_evidence import ValidationUnavailable
from transcript_search_contract import CONFIG,normalize,valid_interval
FIELDS=['video_id','text','speaker','transcript_speaker','diarization_speaker','start_time','end_time','timestamp_unit','video_title','youtube_url','language','video_created_at']
def diverse(rows,limit,key="retrieval_score"):
    groups=defaultdict(list)
    for row in sorted(rows,key=lambda r:-r.get(key,r['score'])):groups[row['video_id']].append(row)
    out=[]
    for depth in range(max([len(x)for x in groups.values()]or[0])):
        for group in groups.values():
            if depth<len(group):out.append(group[depth])
            if len(out)>=limit:return out
    return out

def semantic_gloss(query):
    # Only explicit, reviewed transliterations; preserve the complete original intent.
    value=normalize(query);extras=[]
    for canonical,aliases in CONFIG['aliases'].items():
        normalized={normalize(x)for x in [canonical,*aliases]}
        if value in normalized:extras.extend([canonical,*aliases])
    return query if not extras else query+' ('+'; '.join(dict.fromkeys(extras))+')'

class GroundedSearch:
    def __init__(self,base,judge,model,revision,read_cache,write_cache,embedding_contract):
        self.base=base;self.judge=judge;self.model=model;self.revision=revision;self.read_cache=read_cache;self.write_cache=write_cache;self.embedding_contract=embedding_contract
    def search(self,request,video_ids=None):
        started=time.monotonic();epoch=self.revision();params=request.model_dump()if hasattr(request,'model_dump')else vars(request).copy()
        for k in ['cursor','exact_cursor','search_session_id','batch_size']:params.pop(k,None)
        key='grounded:'+hashlib.sha256(json.dumps([VERSION,hashlib.sha256(SYSTEM.encode()).hexdigest(),self.model,self.embedding_contract,epoch,params,video_ids],sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
        cached=self.read_cache(key)
        if cached is not None:
            if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
            result=copy.deepcopy(cached);meta=result['metadata']['passage_validation'];meta['cache_hit']=True;meta['cached_provider_calls']=meta['provider_calls'];meta['provider_calls']=0;meta['provider_usage']=[];result['ranking']['embedding_usage']=[];result['ranking']['timings']['total_seconds']=round(time.monotonic()-started,3);return result
        original=' '.join([request.query or '',request.word or '',*(request.words or [])]).strip()
        changes={'query':semantic_gloss(original),'word':None,'words':[],'top_k':self.base.settings.candidate_limit}
        expanded=request.model_copy(update=changes)if hasattr(request,'model_copy')else __import__('types').SimpleNamespace(**{**vars(request),**changes})
        from search_observability import capture_embeddings
        with capture_embeddings() as embedding_usage:response=self.base.search(expanded,video_ids)
        retrieved=time.monotonic();rows=diverse(response['results'],32);trace=[{'point_id':str(r['id']),'video_id':r['video_id'],'stage':'candidate_retrieval','retrieval_score':r.get('retrieval_score',r['score']),'selected_for_context':str(r['id'])in {str(x['id'])for x in rows}}for r in response['results']]
        filters=source_filter(request,video_ids);groups=defaultdict(list)
        for row in rows:groups[row['video_id']].append(row)
        contexts=[];warnings=[];context_reads=0
        context_deadline=time.monotonic()+10
        for video,focus in groups.items():
            if time.monotonic()>=context_deadline:
                warnings.append("context_budget_reached");contexts.extend(focus);continue
            windows=[models.FieldCondition(key='start_time',range=models.Range(gte=max(0,r['start_time']-20),lte=r['end_time']+20))for r in focus]
            window=models.Filter(must=[*filters.must,models.FieldCondition(key='video_id',match=models.MatchValue(value=video)),models.Filter(should=windows)])
            points,offset=self.base.client.scroll(collection_name=self.base.collection,scroll_filter=window,limit=500,with_payload=FIELDS,with_vectors=False,timeout=max(1,min(8,int(context_deadline-time.monotonic()))));context_reads+=1
            neighbors=[{'id':str(p.id),**(p.payload or {})}for p in points]
            if offset is not None:warnings.append('context_window_truncated')
            for row in focus:
                expanded=source_passage(row,neighbors,20)
                passage={**row,**expanded}if expanded else row.copy()
                if not expanded:passage['source_utterances']=[{k:row.get(k)for k in ['id','text','start_time','end_time','speaker']}]
                passage['speaker_identity_status']='unverified_source_label'
                passage['context_speaker_continuity']='anonymous_label' if row.get('transcript_speaker') or row.get('diarization_speaker') else 'unverified'
                passage['segment_ids']=list(map(str,passage.get('segment_ids')or[row['id']]))
                if len(passage['text'])>6000:warnings.append('source_passage_exceeds_validation_budget');continue
                contexts.append(passage)
        contexts=list({candidate_id(r):r for r in contexts}.values());context_done=time.monotonic()
        accepted,metadata=validate_grounded(original,contexts,self.judge)
        if warnings:metadata.update(status='partial',retryable=True,context_warnings=sorted(set(warnings)))
        if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
        accepted=[r for r in accepted if r['score']>=request.min_score];accepted=diverse(accepted,request.top_k,key="score")
        # Grouping occurs AFTER diversity/output selection, so one video cannot consume the output budget.
        final=defaultdict(list)
        for row in sorted(accepted,key=lambda r:-r['score']):final[row['video_id']].append(row)
        result={**response,'query':original,'results':[r for group in final.values()for r in group],'returned':len(accepted),'unique_videos':len(final),'metadata':{'passage_validation':metadata},'ranking':{**response['ranking'],'engine':'grounded_contextual','score_type':'grounded_relevance','evidence_verified':True,'candidate_trace':trace,'candidate_scope':'bounded_dense_or_hybrid_pool','context_reads':context_reads,'embedding_usage':embedding_usage,'model':self.model,'rules_version':VERSION,'index_revision':epoch,'timings':{'retrieval_seconds':round(retrieved-started,3),'context_seconds':round(context_done-retrieved,3),'validation_seconds':metadata['elapsed_seconds'],'total_seconds':round(time.monotonic()-started,3)}}}
        if metadata['status']in{'completed','no_matches'}:self.write_cache(key,result,300)
        return result

def summary_search(engine,request):
    candidates=request.summary_candidates
    if candidates is None:raise ValueError('Summary search requires authorized database candidate IDs')
    fields={c['video_id']:c['matched_summary_fields']for c in candidates};ids=sorted(fields)
    result=engine.search(request,ids or [-1]);metadata=result['metadata']['passage_validation']
    for row in result['results']:row.update(matched_summary_fields=fields[row['video_id']],matched_field='summary')
    total=engine.base.client.count(collection_name=engine.base.collection,count_filter=source_filter(request,ids or [-1]),exact=True,timeout=8).count
    bounded=total>result['ranking']['candidates_retrieved'] or result['ranking']['candidates_retrieved']>metadata['candidate_count']
    status='partial'if bounded or metadata['status']=='partial'else 'completed'if result['results']else 'no_matches'
    result.update(search_mode='simple',filter_type='summary',search_contract={'authority':'python_contract_v1','representation':VERSION,'contract':'summary','status':status,'exhaustive':not bounded and metadata['status']!='partial','counts':{'matching_videos':result['unique_videos'],'displayed_passages':result['returned'],'matched_transcript_occurrences':None},'summary_candidate_videos':len(ids),'summary_candidate_point_count':total,'scope':'authorized_summary_candidates_bounded_semantic_passages','passage_validation':metadata})
    return result
