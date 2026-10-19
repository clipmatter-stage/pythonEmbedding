"""Versioned literal/normalized/explicit-alias transcript evidence contracts.
Source offsets are Python Unicode code points. Retrieval never establishes evidence.
"""
import hashlib
import json
import math
import pathlib
import os
import re
import unicodedata
from semantic_passage_evidence import consolidate_candidates, validate_passages, RetrievalBudgetReached
CONFIG=json.loads(pathlib.Path(__file__).with_name('search_contract_v1.json').read_text())
VERSION=CONFIG['version']

def normalized_with_spans(text):
    chars=[];spans=[]
    for match in re.finditer(r'.[\u0300-\u036f]*',str(text),re.DOTALL):
        normalized=unicodedata.normalize('NFKC',match.group()).casefold().translate(str.maketrans({'ي':'ی','ى':'ی','ك':'ک'}))
        for char in normalized:
            value=char if char.isalnum() or char=='_' or unicodedata.category(char).startswith('M') else ' '
            if value==' ' and (not chars or chars[-1]==' '):continue
            chars.append(value);spans.append((match.start(),match.end()))
    if chars and chars[-1]==' ':chars.pop();spans.pop()
    value=''.join(chars)
    for acronym,letters in CONFIG['acronyms'].items():
        pattern=r'(?<!\w)'+' +'.join(map(re.escape,letters))+r'(?!\w)'
        for m in reversed(list(re.finditer(pattern,value))):
            letter_offsets=[i for i in range(m.start(),m.end()) if value[i]!=' ']
            replacement=[spans[i] for i in letter_offsets]
            spans[m.start():m.end()]=replacement
            value=value[:m.start()]+acronym+value[m.end():]

    # Orthographic plural of this acronym; this is not topic expansion.
    for m in reversed(list(re.finditer(r'(?<!\w)ipps(?!\w)',value))):
        original=spans[m.start():m.end()];spans[m.start():m.end()]=original[:2]+[(original[2][0],original[3][1])]
        value=value[:m.start()]+'ipp'+value[m.end():]
    return value,spans

def normalize(text):return normalized_with_spans(text)[0]

def query_variants(query,contract):
    value=normalize(query)
    if contract=='alias':
        for key,aliases in CONFIG['aliases'].items():
            values={normalize(x) for x in aliases}|{key}
            if value in values:return sorted(values)
    return [value]

def match_spans(text,query,contract='normalized'):
    return CompiledMatcher(query,contract).spans(text)

def indexed_text_fields(text):
    from exact_lexical import derived_fields
    return {'search_contract_version':VERSION,'transcript_normalized_v1':normalize(text),**derived_fields(text)}

def title_score(title,query):
    title=normalize(title.replace('_',' '));query=normalize(query.replace('_',' '))
    if not query:return 0
    return 1 if title==query else .9 if set(query.split()).issubset(title.split()) else 0

def valid_interval(row):
    a=row.get('start_time');b=row.get('end_time')
    return (row.get('timestamp_unit','seconds')=='seconds' and type(a) in (int,float) and type(b) in (int,float)
            and math.isfinite(a) and math.isfinite(b) and 0<=a<b)

def transcript_results(rows,query,contract='normalized'):
    hits=[dict(r) for r in rows if valid_interval(r) and match_spans(r.get('text',''),query,contract)]
    sources={str(r['id']):dict(r) for r in hits}
    results=consolidate_candidates(hits)
    for r in results:
        r['match_spans']=match_spans(r['text'],query,contract)
        r['occurrence_count']=len(r['match_spans'])
        r['source_utterances']=[sources[str(i)] for i in r['segment_ids']]
        r['match_types']=['transcript_'+contract];r['matched_field']='text';r['score']=1
    return results

def search_session_binding(request):
    fields=['query','words','word','search_mode','filter_type','transcript_contract','video_id','speaker','title','language','time_range','filter_year','filter_month','filter_date','query_decomposition_mode','summary_candidates','min_score','allowed_video_ids','authorization_scope','scope_revision','index_revision']
    data={key:getattr(request,key,None) for key in fields}
    data.update(rules_version=VERSION,build=os.getenv('RAILWAY_GIT_COMMIT_SHA','development'),pipeline=os.getenv('SEARCH_PIPELINE','grounded'),judge_model=os.getenv('SEARCH_JUDGE_MODEL',os.getenv('OPENAI_TEXT_MODEL','gpt-6-luna')))
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def contract_search(request,reader,collection,models,embed,judge,*,use_normalized_index=False,verifier=None):
    kind=request.filter_type;contract=getattr(request,'transcript_contract','normalized');query=(getattr(request,'title',None) if kind=='title' else None) or request.query or ' '.join(request.words or [])
    if not query.strip():raise ValueError('A search query is required')
    maximum=min(getattr(request,'max_scanned',10000),100000);top=min(getattr(request,'top_k',1000),1000)
    conditions=[models.FieldCondition(key=k,match=models.MatchValue(value=v)) for k,v in [('processing_status','completed'),('approval_status','approved'),('is_archived',False)]]
    for name,key in [('video_id','video_id'),('language','language')]:
        if getattr(request,name,None) is not None:conditions.append(models.FieldCondition(key=key,match=models.MatchValue(value=getattr(request,name))))
    base=models.Filter(must=conditions);scan_filter=base;normalized_index_used=False
    # Only use indexed normalization after proving full eligible coverage, twice around the read.
    coverage=None
    if use_normalized_index and kind=='text' and contract!='literal' and hasattr(reader,'count'):
        coverage=reader.count(collection_name=collection,count_filter=base,exact=True).count
        covered=reader.count(collection_name=collection,count_filter=models.Filter(must=conditions+[models.FieldCondition(key='search_contract_version',match=models.MatchValue(value=VERSION))]),exact=True).count
        if coverage==covered:
            should=[models.FieldCondition(key='transcript_normalized_v1',match=models.MatchText(text=value)) for value in query_variants(query,contract)]
            scan_filter=models.Filter(must=conditions,should=should);normalized_index_used=True
    fields=['video_id','video_title','speaker','diarization_speaker','transcript_speaker','start_time','end_time','timestamp_unit','text','youtube_url','language','video_created_at','search_contract_version']
    summaries=['video_summary','video_summary_english','video_summary_urdu','summary_en','summary_ur']
    if kind=='summary':fields+=summaries
    rows=[];offset=None;scanned=0;exhaustive=False;summary_candidates=getattr(request,'summary_candidates',None)
    if kind!='summary' or summary_candidates is None:
        try:
            while scanned<maximum:
                points,offset=reader.scroll(collection_name=collection,scroll_filter=scan_filter,limit=min(200,maximum-scanned),offset=offset,with_payload=fields,with_vectors=False)
                scanned+=len(points)
                for p in points:
                    row={'id':str(p.id),**(p.payload or {})};date=str(row.get('video_created_at') or '')
                    date_filter=getattr(request,'filter_date',None)
                    prefix=date_filter or (f'{request.filter_year:04d}'+(f'-{request.filter_month:02d}' if getattr(request,'filter_month',None) else '') if getattr(request,'filter_year',None) else '')
                    if prefix and not date.startswith(prefix):continue
                    tr=getattr(request,'time_range',None) or {}
                    if tr.get('start') is not None and row.get('start_time',-1)<tr['start']:continue
                    if tr.get('end') is not None and row.get('end_time',float('inf'))>tr['end']:continue
                    rows.append(row)
                if offset is None:exhaustive=True;break
                if not points:break
        except RetrievalBudgetReached:pass
    if normalized_index_used:
        after=reader.count(collection_name=collection,count_filter=base,exact=True).count
        if after!=coverage:
            # A legacy writer raced the coverage proof. Re-run safely without the optimization.
            return contract_search(request,reader,collection,models,embed,judge,use_normalized_index=False,verifier=verifier)
    passage_validation=None
    if kind=='text':results=transcript_results(rows,query,contract)
    elif kind=='title':
        videos={}
        for r in rows:
            score=title_score(r.get('video_title',''),query)
            if score and r.get('video_id') not in videos:videos[r['video_id']]={**r,'score':score,'is_video_only':True,'text':'','matched_field':'video_title','match_types':['simple_title_filter']}
        results=sorted(videos.values(),key=lambda r:(-r['score'],str(r['video_id'])))
    elif kind=='summary':
        if summary_candidates is None:
            candidates={}
            for r in rows:
                matched=[f for f in summaries if summary_score(r.get(f,'') or '',query)]
                if matched:candidates.setdefault(r['video_id'],set()).update(matched)
            summary_candidates=[{'video_id':k,'matched_summary_fields':sorted(v)} for k,v in candidates.items()]
        passages=[];fields_by_video={};query_vector=embed(query,8) if summary_candidates else None
        if getattr(request,'summary_candidates',None) is not None:exhaustive=True
        if len(summary_candidates)>60:exhaustive=False
        for candidate in summary_candidates[:60]:
            filter_=models.Filter(must=conditions+[models.FieldCondition(key='video_id',match=models.MatchValue(value=candidate['video_id']))])
            try:response=reader.query_points(collection_name=collection,query=query_vector,query_filter=filter_,limit=30,with_payload=True,with_vectors=False)
            except RetrievalBudgetReached:exhaustive=False;break
            fields_by_video[candidate['video_id']]=candidate['matched_summary_fields']
            passages.extend({'id':str(p.id),**(p.payload or {}),'score':p.score} for p in response.points if valid_interval(p.payload or {}))
        results,passage_validation=validate_passages(query,passages,judge,verifier=verifier)
        for r in results:r['matched_summary_fields']=fields_by_video[r['video_id']];r['matched_field']='summary'
        if passage_validation['status']=='partial':exhaustive=False
    else:raise ValueError('Unsupported search contract')
    truncated=len(results)>top;results=results[:top];exhaustive=exhaustive and not truncated
    counts={'matching_videos':len({r['video_id'] for r in results}),'displayed_passages':0 if kind=='title' else len(results),'matched_transcript_occurrences':sum(r['occurrence_count'] for r in results) if kind=='text' else None}
    metadata={'authority':'python_contract_v1','representation':VERSION,'contract':contract if kind=='text' else kind,'exhaustive':exhaustive,'status':'partial' if not exhaustive else 'completed' if results else 'no_matches','retryable':not exhaustive,'counts':counts,'scanned_segments':scanned,'normalized_index_used':normalized_index_used,'scope':'eligible_indexed_segments'}
    if passage_validation:metadata['passage_validation']=passage_validation
    return {'results':results,'returned':len(results),'search_contract':metadata,'search_mode':'simple','filter_type':kind,'query':query}


def alias_with_spans(text):
    value,spans=normalized_with_spans(text)
    variants={normalize(alias):normalize(key) for key,aliases in CONFIG['aliases'].items() for alias in [key,*aliases]}
    pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in sorted(variants,key=len,reverse=True))+r')(?!\w)')
    for m in reversed(list(pattern.finditer(value))):
        key=variants[m.group()];source=(spans[m.start()][0],spans[m.end()-1][1]);spans[m.start():m.end()]=[source]*len(key);value=value[:m.start()]+key+value[m.end():]
    return value,spans

class CompiledMatcher:
    def __init__(self,query,contract):
        if contract not in {'literal','normalized','alias'}:raise ValueError('Invalid transcript contract')
        self.contract=contract
        value=query.strip() if contract=='literal' else alias_with_spans(query)[0] if contract=='alias' else normalize(query)
        if not value:raise ValueError('Empty normalized query')
        self.pattern=re.compile((r'(?<!\w)' if re.match(r'\w',value[0]) else '')+re.escape(value)+(r'(?!\w)' if re.match(r'\w',value[-1]) else ''))
    def spans(self,text):
        if self.contract=='literal':value=text;mapping=None
        else:value,mapping=alias_with_spans(text) if self.contract=='alias' else normalized_with_spans(text)
        out=[]
        for m in self.pattern.finditer(value):
            start=m.start() if mapping is None else mapping[m.start()][0];end=m.end() if mapping is None else mapping[m.end()-1][1]
            out.append({'start':start,'end':end,'text':text[start:end]})
        return out


def summary_score(text,query):
    body=alias_with_spans(text)[0];wanted=alias_with_spans(query)[0]
    return bool(wanted) and set(wanted.split()).issubset(body.split())
