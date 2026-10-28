"""Diverse semantic candidates -> original nearby turns -> one grounded rerank.
No embedding-space change or new identity assertions. Exact matching is separate.
"""
from speaker_constraint import passes_constraint
from speaker_identity import view as identity_view, FIELDS as IDENTITY_FIELDS, supported as identity_supported
import ast,copy,hashlib,json,time,os,re
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from qdrant_client import models
from contextual_search import source_filter
from grounded_validation import VERSION,SYSTEM,validate_grounded,candidate_id
from semantic_passage_evidence import ValidationUnavailable
from transcript_search_contract import CONFIG,normalize,valid_interval,GROUNDED_STRATEGY_VERSION,semantic_query_text
# Cache only successful, revision-bound snapshots; transient partials remain retryable.
SNAPSHOT_TTL=None  # Durable snapshots have no time-based expiry.
JUDGMENT_TTL=604800

@lru_cache(maxsize=1)
def search_rules_fingerprint():
    # Invalidate for search logic changes, independently of deployment commit IDs.
    files=('grounded_search.py','grounded_validation.py','fast_search.py','contextual_search.py',
           'grounded_title_search.py','grounded_summary_search.py','transcript_search_contract.py',
           'search_contract_v1.json','semantic_query_decomposition.py','speaker_identity.py',
           'semantic_passage_evidence.py','exact_lexical.py','embeddings_service.py','manual_metadata.py','asr_search.py',
           'intent_retrieval.py','speaker_constraint.py')
    digest=hashlib.sha256()
    for name in files:
        source=(Path(__file__).parent/name).read_bytes()
        if name=='embeddings_service.py':
            functions={'understand_query','grounded_retrieval_plan','grounded_title_intent',
                       'get_contextual_search','search','search_incremental','search_pipeline','search_engine',
                       'get_semantic_query_embedding'}
            tree=ast.parse(source)
            selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in functions]
            # Source text is stable across the local and production Python versions.
            source='\n'.join(ast.get_source_segment(source.decode(),node)for node in selected).encode()
        digest.update(name.encode());digest.update(source)
    return digest.hexdigest()

def legacy_snapshot_commits(engine):
    # A one-time migration is allowed only for the exact reviewed search rules.
    path=Path(__file__).parent/'search_snapshot_migration.json'
    if not path.exists():return []
    migration=json.loads(path.read_text())
    if migration.get('compatible_rules_fingerprint')!=search_rules_fingerprint():return []
    if len(engine.embedding_contract)>2 and engine.embedding_contract[2]!=migration.get('planner_model'):return []
    return migration.get('legacy_deployments',[])

def snapshot_key(engine,epoch,params,video_ids,initial,maximum,legacy_commit=None):
    contract=engine.embedding_contract[:2]if legacy_commit else engine.embedding_contract
    parts=['stable_snapshot_v1'if legacy_commit else 'persistent_snapshot_v1',
           GROUNDED_STRATEGY_VERSION,engine.title_indexed,engine.summary_indexed_fields,initial,maximum,
           VERSION,hashlib.sha256(SYSTEM.encode()).hexdigest(),legacy_commit or search_rules_fingerprint(),
           asdict(engine.base.settings),engine.base.sparse_enabled,engine.model,
           getattr(engine.judge,'cache_contract',None),contract,epoch,params,video_ids]
    return 'grounded:'+hashlib.sha256(json.dumps(parts,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def stable_snapshot(metadata,title,summary):
    return (metadata['status'] in {'completed','no_matches'}
        and not (set(title.get('warnings',[])) - {'title_retrieval_bounded'})
        and not (set(summary.get('warnings',[])) - {'summary_retrieval_bounded'}))

FIELDS=IDENTITY_FIELDS+['video_id','segment_index','text','speaker','transcript_speaker','diarization_speaker','start_time','end_time','timestamp_unit','video_title','youtube_url','language','video_created_at']
def diverse(rows,limit,key="retrieval_score"):
    groups=defaultdict(list)
    for row in sorted(rows,key=lambda r:(-r.get(key,r['score']),str(r['video_id']),str(r['id']))):groups[row['video_id']].append(row)
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

def coherent_passage(focus, neighbors):
    """Retain original turns; temporal context never establishes an identity."""
    from contextual_search import timestamp_url
    current=focus.copy()
    ordered=sorted([focus,*[r for r in neighbors if str(r['id'])!=str(focus['id'])
                    and r.get('video_id')==focus.get('video_id')and valid_interval(r)]],key=lambda r:(r['start_time'],r['end_time']))
    center=next(i for i,r in enumerate(ordered)if str(r['id'])==str(focus['id']))
    turns=[focus];label=focus.get('transcript_speaker')or focus.get('diarization_speaker')
    anonymous_unknown=str(focus.get('speaker')or'').casefold()in{'','unknown','unknown speaker'}and not label
    indexed_sequence=anonymous_unknown and type(focus.get('segment_index'))is int
    directions=()if anonymous_unknown and not indexed_sequence else (-1,1)
    for direction in directions:
        index=center+direction
        while 0<=index<len(ordered)and len(turns)<32:
            row=ordered[index]
            if ((row.get('speaker')!=focus.get('speaker') or row.get('source_identity_group')!=focus.get('source_identity_group'))or(row.get('transcript_speaker')or row.get('diarization_speaker'))!=label
                    or not str(row.get('text')or'').strip()):break
            if indexed_sequence:
                edge=turns[0]if direction<0 else turns[-1]
                if type(row.get('segment_index'))is not int or row['segment_index']!=edge['segment_index']+direction:break
            gap=turns[0]['start_time']-row['end_time']if direction<0 else row['start_time']-turns[-1]['end_time']
            proposed=sorted([*turns,row],key=lambda r:r['start_time'])
            if (not 0<=gap<=1 or proposed[-1]['end_time']-proposed[0]['start_time']>60
                    or sum(len(r['text'])+1 for r in proposed)>2400):break
            turns=proposed;index+=direction
    current.update(text=' '.join(r['text'].strip()for r in turns),start_time=turns[0]['start_time'],
                   end_time=turns[-1]['end_time'],segment_ids=[str(r['id'])for r in turns],
                   source_utterances=[{k:r.get(k)for k in ['id','segment_index','text','start_time','end_time','speaker','transcript_speaker','diarization_speaker']}for r in turns])
    current.update(duration=current['end_time']-current['start_time'],match_count=len(turns),
                   youtube_url_timestamped=timestamp_url(current.get('youtube_url',''),current['start_time']))
    return current

def contextual_candidate(focus, neighbors, named=()):
    """Recover a full nearby original turn without joining different speakers.

    A replacement has its own point ID, source label, times and quotations.
    Named-speaker constraints still apply to that source turn independently.
    """
    passage=coherent_passage(focus,neighbors)
    if len(passage['text'].split())>=8 or passage['end_time']-passage['start_time']>6:
        return passage
    options=[]
    for row in neighbors:
        if (str(row.get('id'))==str(focus['id'])or row.get('video_id')!=focus.get('video_id')
                or not valid_interval(row)or len(str(row.get('text')or'').split())<8):continue
        if named and normalize(row.get('speaker',''))not in named:continue
        gap=max(0,focus['start_time']-row['end_time'],row['start_time']-focus['end_time'])
        if gap>20:continue
        options.append((gap,abs(row['start_time']-focus['start_time']),str(row['id']),row))
    if not options:return passage
    recovered=coherent_passage(min(options,key=lambda item:item[:3])[3],neighbors)
    recovered['context_retrieval_anchor']={'point_id':str(focus['id']),
        'selection':'nearby_original_turn','retrieval_score':focus.get('retrieval_score',focus.get('score'))}
    return recovered

def ranked_diverse(rows,limit):
    # Give each retrieved video one verification slot before second passages.
    if limit<=0:return []
    return diverse(rows,limit)

class GroundedSearch:
    def __init__(self,base,judge,model,revision,read_cache,write_cache,embedding_contract,title_intent=None,title_indexed=False,summary_indexed_fields=(),snapshot_lock=None,metadata_search=None):
        self.metadata_search=metadata_search
        self.snapshot_lock=snapshot_lock
        self.title_intent=title_intent;self.title_indexed=title_indexed;self.summary_indexed_fields=tuple(summary_indexed_fields)
        self.base=base;self.base.payload_fields=FIELDS;self.judge=judge;self.model=model;self.revision=revision;self.read_cache=read_cache;self.write_cache=write_cache;self.embedding_contract=embedding_contract
    def search(self,request,video_ids=None):
        from asr_search import canonical_query
        original=semantic_query_text(request);canonical=canonical_query(original)
        if canonical!=original:
            changes={'query':canonical,'word':None,'words':[]}
            request=request.model_copy(update=changes)if hasattr(request,'model_copy')else __import__('types').SimpleNamespace(**{**vars(request),**changes})
        if self.snapshot_lock is None:result=self._search(request,video_ids)
        else:
            params=request.model_dump()if hasattr(request,'model_dump')else vars(request).copy()
            for key in ['cursor','exact_cursor','search_session_id','batch_size']:params.pop(key,None)
            identity=hashlib.sha256(json.dumps([params,video_ids,self.model,self.revision()],sort_keys=True,default=str).encode()).hexdigest()
            with self.snapshot_lock(identity):result=self._search(request,video_ids)
        result=copy.deepcopy(result);result['query']=original
        validation=result.get('metadata',{}).get('passage_validation',{})
        if 'original_query'in validation:validation['original_query']=original
        if canonical!=original:validation['canonical_query']=canonical
        return result

    def _search(self,request,video_ids=None):
        started=time.monotonic();epoch=self.revision();initial=int(os.getenv('GROUNDING_INITIAL_CANDIDATES','12'));maximum=int(os.getenv('GROUNDING_MAX_CANDIDATES','24'))
        if not 1<=initial<=maximum<=32:raise ValueError('Grounding limits must satisfy 1 <= initial <= maximum <= 32')
        params=request.model_dump()if hasattr(request,'model_dump')else vars(request).copy()
        for k in ['cursor','exact_cursor','search_session_id','batch_size']:params.pop(k,None)
        cache_key=snapshot_key(self,epoch,params,video_ids,initial,maximum)
        cached=self.read_cache(cache_key)
        if cached is None:
            for commit in legacy_snapshot_commits(self):
                previous=self.read_cache('legacy:'+snapshot_key(self,epoch,params,video_ids,initial,maximum,commit))
                if previous is not None:
                    if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
                    self.write_cache(cache_key,previous,SNAPSHOT_TTL);cached=previous;break
        if cached is not None:
            if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
            result=copy.deepcopy(cached);meta=result['metadata']['passage_validation'];meta['cache_hit']=True;meta['cached_provider_calls']=meta['provider_calls'];meta['provider_calls']=0;meta['provider_usage']=[];result['ranking']['embedding_usage']=[];result['ranking']['timings']['total_seconds']=round(time.monotonic()-started,3);return result
        original=semantic_query_text(request)
        metadata_rows=self.metadata_search(request)if self.metadata_search else []
        from grounded_title_search import title_candidates,merge_title_candidates
        from grounded_summary_search import summary_candidates,merge_summary_candidates
        title_rows=[];title_metadata={'matching_videos':0,'elapsed_seconds':0,'warnings':[],'exhaustive':True}
        summary_rows=[];summary_metadata={'matching_videos':0,'elapsed_seconds':0,'warnings':[],'exhaustive':True}
        title_plan=self.title_intent(original)if self.title_intent else {'queries':[original],'person_only':False}
        metadata_enabled=getattr(request,'search_mode','semantic')=='semantic'and not getattr(request,'filter_type',None)
        # Both payload-only lookups have their own bounded deadline and run in
        # parallel so summary discovery does not add a second serial scan.
        if metadata_enabled:
            with ThreadPoolExecutor(max_workers=2)as executor:
                title_future=executor.submit(title_candidates,self.base,request,title_plan['queries'],video_ids,self.title_indexed)
                summary_future=executor.submit(summary_candidates,self.base,request,title_plan['queries'],video_ids,self.summary_indexed_fields)
                try:title_rows,title_metadata=title_future.result()
                except Exception:title_metadata.update(exhaustive=False,warnings=['title_retrieval_unavailable'])
                try:summary_rows,summary_metadata=summary_future.result()
                except Exception:summary_metadata.update(exhaustive=False,warnings=['summary_retrieval_unavailable'])
        numeric_metadata=original.strip().isdigit()and bool(metadata_rows)
        if (title_plan.get('person_only')or numeric_metadata)and metadata_enabled:
            if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
            _,metadata=validate_grounded(original,[],self.judge)
            metadata.update(status='completed'if numeric_metadata else 'not_applicable',reason='numeric_metadata_lookup'if numeric_metadata else 'name_only_metadata_lookup',exhaustive=False)
            combined=merge_summary_candidates(title_rows,summary_rows,request.top_k)
            result={'query':original,'words':request.words,'speaker_filter':request.speaker,'search_mode':'semantic',
                'results':combined,'returned':len(combined),
                'unique_videos':len(combined),
                'metadata':{'passage_validation':metadata,'title_retrieval':title_metadata,'summary_retrieval':summary_metadata},
                'ranking':{'engine':'grounded_metadata_lookup'if summary_rows else 'grounded_title_lookup','score_type':'metadata_lexical_match',
                    'evidence_verified':False,'candidate_sources':['video_title','video_summary'],
                    'candidates_retrieved':len({r['video_id']for r in [*title_rows,*summary_rows]}),'embedding_usage':[],'index_revision':epoch,
                    'timings':{'title_seconds':title_metadata['elapsed_seconds'],'summary_seconds':summary_metadata['elapsed_seconds'],'total_seconds':round(time.monotonic()-started,3)}}}
            from manual_metadata import merge_metadata
            result=merge_metadata(result,metadata_rows,request.top_k)
            if stable_snapshot(metadata,title_metadata,summary_metadata):self.write_cache(cache_key,result,SNAPSHOT_TTL)
            return result
        changes={'query':semantic_gloss(original),'word':None,'words':[],'top_k':self.base.settings.candidate_limit}
        expanded=request.model_copy(update=changes)if hasattr(request,'model_copy')else __import__('types').SimpleNamespace(**{**vars(request),**changes})
        from search_observability import capture_embeddings
        semantic_started=time.monotonic()
        with capture_embeddings() as embedding_usage:response=self.base.search(expanded,video_ids)
        retrieved=time.monotonic()
        if response.get('speaker_only'):
            return self._speaker_speeches(request,response,original,title_rows,title_metadata,summary_rows,summary_metadata,started,epoch,cache_key)
        decomposition=response.get('query_decomposition')or{}
        speaker=request.speaker or response.get('speaker_filter')
        # Only deterministic decomposition may remove an actor from the judge's
        # request. A title, anonymous diarization ID or fuzzy name is no proof.
        named_constraint=bool(speaker and not re.match(r'^(?:speaker[ _-]*\d+|unknown(?: speaker)?)$',speaker,re.I))
        judging_query=decomposition.get('topic')if named_constraint and decomposition.get('decomposed')else original
        variants=self.base.speaker_values(speaker)if speaker and self.base.speaker_values else []
        named={normalize(name)for name in [speaker,*variants]if name}
        pool=response['results']
        if speaker:pool=[r for r in pool if passes_constraint(r,speaker,variants) or (named and normalize(r.get('speaker') or '') in named)]
        rows=ranked_diverse(pool,initial)
        chosen={str(row['id'])for row in rows}
        rows+=ranked_diverse([row for row in pool if str(row['id'])not in chosen],maximum-len(rows))if len(rows)<maximum else []
        from asr_search import reserve_candidates
        reserved=reserve_candidates(original,pool)
        reserved_ids={str(row['id'])for row in reserved}
        rows=[*reserved,*[row for row in rows if str(row['id'])not in reserved_ids]][:maximum]
        trace=[{'point_id':str(r['id']),'video_id':r['video_id'],'stage':'candidate_retrieval',
                'retrieval_score':r.get('retrieval_score',r['score']),
                'selected_for_context':str(r['id'])in{str(x['id'])for x in rows}}for r in response['results']]
        filters=source_filter(request,video_ids)
        contexts=[];warnings=[w for w in response.get('ranking',{}).get('retrieval_warnings',[])
            if w in {'multilingual_expansion_unavailable','translated_embedding_unavailable'}]
        context_reads=0;accepted=[];stages=[];seen=set()
        def judgment_key(query,row):
            value=[VERSION,hashlib.sha256(SYSTEM.encode()).hexdigest(),GROUNDED_STRATEGY_VERSION,
                self.model,getattr(self.judge,'cache_contract',None),epoch,query,candidate_id(row),
                row.get('speaker'),row.get('transcript_speaker'),row.get('diarization_speaker')]
            return 'grounded_judgment:'+hashlib.sha256(json.dumps(value,ensure_ascii=False,default=str).encode()).hexdigest()
        # Cache entries are considered only for freshly retrieved, eligible
        # passages, and are checked again by the same quote/role validator.
        read_judgment=None if warnings else lambda q,row:self.read_cache(judgment_key(q,row))
        write_judgment=None if warnings else lambda q,row,j:self.write_cache(judgment_key(q,row),j,JUDGMENT_TTL)
        context_seconds=validation_seconds=0.;deadline=time.monotonic()+60
        for offset in range(0,len(rows),initial):
            calls_left=6-sum(m['provider_calls']for m in stages)
            if calls_left<=0:warnings.append('validation_call_budget_reached');break
            if offset and len({r['video_id']for r in accepted})>=min(request.top_k,10):break
            if time.monotonic()>=deadline:warnings.append('validation_budget_reached');break
            focus=rows[offset:offset+initial];tick=time.monotonic()
            windows=[models.Filter(must=[models.FieldCondition(key='video_id',match=models.MatchValue(value=r['video_id'])),
                models.FieldCondition(key='start_time',range=models.Range(gte=max(0,r['start_time']-20),lte=r['end_time']+20))])for r in focus]
            window=models.Filter(must=[*filters.must,models.Filter(should=windows)])
            points,next_offset=self.base.client.scroll(collection_name=self.base.collection,scroll_filter=window,
                limit=500,with_payload=FIELDS,with_vectors=False,timeout=6);context_reads+=1
            if next_offset is not None:warnings.append('context_window_truncated')
            neighbors=[{'id':str(p.id),**identity_view(p.payload or{})}for p in points];batch=[]
            for row in focus:
                passage=contextual_candidate(row,neighbors,named)
                if str(passage['id'])!=str(row['id']):
                    trace.append({'point_id':str(passage['id']),'video_id':passage['video_id'],
                        'stage':'nearby_original_turn','anchor_point_id':str(row['id']),'selected_for_context':True})
                passage['speaker_identity_status']='supported_identity'if identity_supported(passage) else 'unverified_source_label'
                passage['context_speaker_continuity']='anonymous_label'if row.get('transcript_speaker')or row.get('diarization_speaker')else 'unverified'
                passage['segment_ids']=list(map(str,passage.get('segment_ids')or[row['id']]))
                passage_key=candidate_id(passage)
                if passage_key in seen:continue
                seen.add(passage_key)
                if len(passage['text'])>6000:warnings.append('source_passage_exceeds_validation_budget');continue
                batch.append(passage)
            context_seconds+=time.monotonic()-tick
            if not batch:continue
            if time.monotonic()>=deadline:warnings.append('validation_budget_reached');break
            contexts.extend(batch)
            try:
                passed,stage=validate_grounded(judging_query,batch,self.judge,batch_size=6,max_calls=min(3,calls_left),
                    deadline_seconds=max(.1,deadline-time.monotonic()),concurrency=2,
                    read_judgment=read_judgment,write_judgment=write_judgment)
            except ValidationUnavailable:
                if not accepted and not title_rows and not summary_rows and not metadata_rows:raise
                warnings.append('expansion_validation_unavailable');break
            accepted.extend(r for r in passed if r['score']>=request.min_score);stages.append(stage)
            validation_seconds+=stage['elapsed_seconds']
        # Empty candidate sets are evaluated without a provider call.
        if not stages:
            _,metadata=validate_grounded(judging_query,[],self.judge)
        else:
            metadata=copy.deepcopy(stages[0])
            for key in ['candidate_count','evaluated_count','provider_calls','unresolved_count','elapsed_seconds','judgment_cache_hits']:
                metadata[key]=sum(m[key]for m in stages)
            for key in ['trace','provider_usage','failure_reasons','retried_errors']:
                metadata[key]=[x for m in stages for x in m[key]]
        checked_ids={str(identity) for row in contexts for identity in row.get('segment_ids',[row['id']])}
        metadata['evaluated_source_segments']=len(checked_ids)
        metadata.update(status='partial'if any(m['status']=='partial'for m in stages)or warnings else 'completed'if accepted else 'no_matches',
                        retryable=any(m['retryable']for m in stages)or bool(warnings),
                        context_warnings=warnings,original_query=original,grounding_query=judging_query,
                        speaker_constraint=speaker,speaker_identity_authority='source_metadata_only',
                        initial_candidate_limit=initial,maximum_candidate_limit=maximum,
                        retrieval_candidates=len(response['results']),selected_candidates=len(rows),
                        has_unchecked_candidates=bool({str(r['id'])for r in pool}-checked_ids),exhaustive=False)
        if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
        accepted=[r for r in accepted if r['score']>=request.min_score];accepted=diverse(accepted,request.top_k,key="score")
        # Grouping occurs AFTER diversity/output selection, so one video cannot consume the output budget.
        final=defaultdict(list)
        for row in sorted(accepted,key=lambda r:(-r['score'],str(r['video_id']),str(r['id']))):final[row['video_id']].append(row)
        result={**response,'query':original,'results':[r for group in final.values()for r in group],'returned':len(accepted),'unique_videos':len(final),'metadata':{'passage_validation':metadata},'ranking':{**response['ranking'],'engine':'grounded_contextual','score_type':'grounded_relevance','evidence_verified':True,'candidate_trace':trace,'candidate_scope':'bounded_dense_or_hybrid_pool','context_reads':context_reads,'embedding_usage':embedding_usage,'model':self.model,'rules_version':VERSION,'index_revision':epoch,'timings':{'retrieval_seconds':round(retrieved-semantic_started,3),'context_seconds':round(context_seconds,3),'validation_seconds':round(validation_seconds,3),'total_seconds':round(time.monotonic()-started,3)}}}
        merged=merge_title_candidates(result['results'],title_rows,len(result['results'])+len(title_rows))
        merged=merge_summary_candidates(merged,summary_rows,request.top_k)
        if not merged and any(w in {'multilingual_expansion_unavailable','translated_embedding_unavailable'}for w in warnings):
            raise ValidationUnavailable('Query expansion unavailable',reason='query_expansion_unavailable')
        result.update(results=merged,returned=len(merged),unique_videos=len({r['video_id']for r in merged}))
        result['metadata']['title_retrieval']=title_metadata
        result['metadata']['summary_retrieval']=summary_metadata
        result['ranking'].update(candidate_sources=['video_title','video_summary','transcript']if metadata_enabled else ['transcript'],
            evidence_verified=bool(merged)and all(r['transcript_verified']for r in merged),
            transcript_verified_passages=sum(r['transcript_verified']for r in merged),
            title_only_videos=sum(bool(r.get('is_video_only')and r.get('title_evidence'))for r in merged),
            summary_only_videos=sum(bool(r.get('is_video_only')and r.get('summary_evidence'))for r in merged))
        if any(r.get('is_video_only')for r in merged):result['ranking']['score_type']='mixed_source_relevance'
        result['ranking']['timings']['title_seconds']=title_metadata['elapsed_seconds']
        result['ranking']['timings']['summary_seconds']=summary_metadata['elapsed_seconds']
        from manual_metadata import merge_metadata
        result=merge_metadata(result,metadata_rows,request.top_k)
        if stable_snapshot(metadata,title_metadata,summary_metadata):self.write_cache(cache_key,result,SNAPSHOT_TTL)
        return result

    def _speaker_speeches(self, request, response, original, title_rows, title_metadata, summary_rows, summary_metadata, started, epoch, cache_key):
        """Return that speaker's transcript turns without judging speech boilerplate."""
        rows=[]
        for row in response.get('results') or []:
            if float(row.get('score') or 0)<request.min_score:continue
            passage=dict(row)
            passage['speaker_identity_status']='supported_identity' if identity_supported(passage) else 'unverified_source_label'
            passage['segment_ids']=list(map(str, passage.get('segment_ids') or [passage['id']]))
            passage.update(evidence_verified=False, transcript_verified=False, evidence_source='speaker_speech')
            rows.append(passage)
        rows=rows[:request.top_k]
        if self.revision()!=epoch:raise ValidationUnavailable('Search index changed',reason='search_index_changed')
        _,metadata=validate_grounded(original,[],self.judge)
        scan=(response.get('metadata') or {}).get('speaker_scan') or {}
        bounded=not scan.get('exhaustive', True)
        metadata.update(status='partial' if bounded and rows else 'completed' if rows else 'no_matches',
            reason='speaker_speech_lookup', exhaustive=not bounded, retryable=bounded,
            original_query=original, grounding_query=original, speaker_constraint=response.get('speaker_filter'),
            speaker_identity_authority='source_metadata_only', provider_calls=0,
            candidate_count=len(rows), evaluated_count=len(rows))
        result={'query':original,'words':request.words,'speaker_filter':response.get('speaker_filter'),
            'search_mode':'semantic','results':rows,'returned':len(rows),
            'unique_videos':len({r['video_id'] for r in rows}),
            'metadata':{'passage_validation':metadata,'title_retrieval':title_metadata,'summary_retrieval':summary_metadata},
            'ranking':{'engine':'speaker_speech_lookup','score_type':'speaker_speech','evidence_verified':False,
                'candidate_sources':['transcript'],'embedding_usage':[],
                'candidates_retrieved':(response.get('ranking') or {}).get('candidates_retrieved', len(rows)),
                'index_revision':epoch,'model':self.model,'rules_version':VERSION,
                'timings':{'retrieval_seconds':0,'context_seconds':0,'validation_seconds':0,
                    'total_seconds':round(time.monotonic()-started,3),
                    'title_seconds':title_metadata.get('elapsed_seconds',0),
                    'summary_seconds':summary_metadata.get('elapsed_seconds',0)}}}
        if stable_snapshot(metadata,title_metadata,summary_metadata):self.write_cache(cache_key,result,SNAPSHOT_TTL)
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
    result.update(search_mode='simple',filter_type='summary',search_contract={'authority':'python_contract_v1','representation':VERSION,'contract':'summary','status':status,'exhaustive':not bounded and metadata['status']!='partial','partial_reasons':(['bounded_semantic_evaluation']if bounded else[])+metadata.get('partial_reasons',[]),'counts':{'matching_videos':result['unique_videos'],'displayed_passages':result['returned'],'matched_transcript_occurrences':None},'summary_candidate_videos':len(ids),'summary_candidate_point_count':total,'scope':'authorized_summary_candidates_bounded_semantic_passages','passage_validation':metadata})
    return result
