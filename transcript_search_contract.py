"""Versioned lexical contracts. Spans are Unicode code-point [start,end)."""
import json
import math
import pathlib
import re
import unicodedata
from collections import OrderedDict
from semantic_passage_evidence import consolidate_candidates, normalize_multilingual, validate_passages, RetrievalBudgetReached

CONFIG=json.loads(pathlib.Path(__file__).with_name('search_contract_v1.json').read_text())
VERSION=CONFIG['version']
_CHAR_TABLE=str.maketrans({'ي':'ی','ى':'ی','ك':'ک','’':"'",'‘':"'",'“':'"','”':'"',
    '،':',','؟':'?','۔':'.','–':'-','—':'-','\u200c':' ','\u200d':' '})
SUMMARY_FIELDS=('video_summary','video_summary_english','video_summary_urdu','summary_en','summary_ur')


def normalized_with_spans(text):
    chars=[];positions=[]
    # Normalize a base plus combining marks together and preserve original spans.
    for match in re.finditer(r'[^\W_][\u0300-\u036f]*|.',text,re.UNICODE|re.DOTALL):
        value=unicodedata.normalize('NFKC',match.group()).translate(_CHAR_TABLE).casefold()
        for char in value:
            chars.append(char);positions.append((match.start(),match.end()))
    tokens=[]
    for match in re.finditer(r'\w+', ''.join(chars),re.UNICODE):
        tokens.append((match.group(),positions[match.start()][0],positions[match.end()-1][1]))
    output=[];i=0
    while i<len(tokens):
        for acronym,letters in CONFIG['acronyms'].items():
            if [t[0] for t in tokens[i:i+len(letters)]]==letters:
                output.append((acronym,tokens[i][1],tokens[i+len(letters)-1][2]));i+=len(letters);break
        else:
            output.append(tokens[i]);i+=1
    normalized='';mapping=[]
    for token,start,end in output:
        if normalized:normalized+=' ';mapping.append((start,start))
        normalized+=token;mapping.extend([(start,end)]*len(token))
    return normalized,mapping


def match_spans(text,query,contract='normalized'):
    if contract not in {'literal','normalized','alias'}:raise ValueError('Unknown transcript contract')
    if not query or not query.strip():return []
    if contract=='literal':
        haystack=text;needles=[query];mapping=None
    else:
        haystack,mapping=normalized_with_spans(text)
        needle=normalized_with_spans(query)[0];needles=[needle]
        if contract=='alias':
            for equivalents in CONFIG['aliases'].values():
                normalized=[normalized_with_spans(x)[0] for x in equivalents]
                if needle in normalized:needles=normalized;break
    spans={}
    for needle in needles:
        if not needle:continue
        prefix=r'(?<!\w)' if re.match(r'\w',needle[0],re.UNICODE) else ''
        suffix=r'(?!\w)' if re.match(r'\w',needle[-1],re.UNICODE) else ''
        for m in re.finditer(prefix+re.escape(needle)+suffix,haystack,re.UNICODE):
            start,end=(m.start(),m.end()) if mapping is None else (mapping[m.start()][0],mapping[m.end()-1][1])
            spans[(start,end)]={'start':start,'end':end,'text':text[start:end]}
    return [spans[key] for key in sorted(spans)]


def title_key(text):return normalized_with_spans(text)[0]


def title_score(title,query):
    key=title_key(title);wanted=title_key(query)
    if not wanted:return 0
    if key==wanted:return 1.0
    return .9 if all(term in key.split() for term in wanted.split()) else 0


def valid_interval(row):
    try:
        start=float(row['start_time']);end=float(row['end_time'])
        return math.isfinite(start) and math.isfinite(end) and 0<=start<end and row.get('timestamp_unit','seconds')=='seconds'
    except (KeyError,TypeError,ValueError):return False


def transcript_results(rows,query,contract):
    matched=[];sources={}
    for row in rows:
        if not valid_interval(row):continue
        occurrences=match_spans(row.get('text',''),query,contract)
        if not occurrences:continue
        item={**row,'score':1.0,'match_types':['transcript_'+contract],'matched_field':'text'}
        matched.append(item);sources[(str(row['video_id']),str(row['id']))]=(row,occurrences)
    results=consolidate_candidates(matched)
    for result in results:
        offset=0;occurrences=[];utterances=[]
        for sid in result['segment_ids']:
            original,spans=sources[(str(result['video_id']),str(sid))]
            utterances.append({'id':sid,'text':original['text'],'start_time':original['start_time'],'end_time':original['end_time']})
            occurrences.extend({**span,'start':span['start']+offset,'end':span['end']+offset,
                'source_id':sid,'source_start_time':original['start_time'],'source_end_time':original['end_time']} for span in spans)
            offset+=len(original['text'])+1
        result.update(match_spans=occurrences,occurrence_count=len(occurrences),source_utterances=utterances)
    return results


def date_matches(payload,request):
    value=str(payload.get('video_created_at') or '')[:10]
    if request.filter_date:return value==request.filter_date[:10]
    if request.filter_year:
        prefix=str(request.filter_year)+(f'-{request.filter_month:02}' if request.filter_month else '')
        return value.startswith(prefix)
    return True


def contract_search(request,reader,collection,models,embedding,judge,*,use_normalized_index=False):
    """Bounded existing-payload reader, with no writes or embedding-space changes."""
    query=(request.title or request.query) if request.filter_type=='title' else (request.query or ' '.join(request.words))
    if not query or not query.strip():raise ValueError('A search query is required')
    conditions=[models.FieldCondition(key=k,match=models.MatchValue(value=v)) for k,v in
        [('processing_status','completed'),('approval_status','approved'),('is_archived',False)]]
    if request.video_id:conditions.append(models.FieldCondition(key='video_id',match=models.MatchValue(value=request.video_id)))
    if request.language:conditions.append(models.FieldCondition(key='language',match=models.MatchValue(value=request.language)))
    time_range=getattr(request,'time_range',None) or {}
    for field,bound,key in [('start_time','gte','start'),('end_time','lte','end')]:
        if time_range.get(key) is not None:
            conditions.append(models.FieldCondition(key=field,range=models.Range(**{bound:time_range[key]})))
    scan_filter=models.Filter(must=conditions)
    using_index=False
    if use_normalized_index and request.filter_type=='text' and request.transcript_contract!='literal':
        candidate_filter=normalized_prefilter(query,request.transcript_contract,models)
        if candidate_filter is not None:
            total=reader.count(collection_name=collection,count_filter=scan_filter,exact=True).count
            versioned=models.Filter(must=conditions+[models.FieldCondition(key='search_contract_version',match=models.MatchValue(value=VERSION))])
            ready=reader.count(collection_name=collection,count_filter=versioned,exact=True).count
            if total==ready:
                # Include unversioned points even after the coverage probes, so
                # a concurrent legacy writer cannot hide its transcript from retrieval.
                unversioned=models.Filter(must_not=[models.FieldCondition(key='search_contract_version',match=models.MatchValue(value=VERSION))])
                safe_candidates=models.Filter(should=[candidate_filter,unversioned])
                scan_filter=models.Filter(must=conditions+[safe_candidates]);using_index=True
    fields=['video_id','video_title','video_filename','youtube_url','language','video_created_at']
    if request.filter_type=='summary':fields+=list(SUMMARY_FIELDS)
    elif request.filter_type=='text':fields+=['text','speaker','diarization_speaker','transcript_speaker','start_time','end_time','timestamp_unit']
    offset=None;scanned=0;rows=[];videos=OrderedDict();exhausted=False;limited=False
    cap=min(request.max_scanned,10000)
    supplied_summary = getattr(request,'summary_candidates',None) if request.filter_type=='summary' else None
    if supplied_summary is not None:
        for item in supplied_summary:
            videos[int(item['video_id'])]={'matched_summary_fields':item.get('matched_summary_fields',[])}
        exhausted=True
    try:
        while scanned<cap and supplied_summary is None:
            points,next_offset=reader.scroll(collection_name=collection,scroll_filter=scan_filter,
                limit=min(500,cap-scanned),offset=offset,with_payload=fields,with_vectors=False)
            scanned+=len(points)
            for point in points:
                payload=point.payload or {}
                if not date_matches(payload,request):continue
                if request.filter_type=='text':
                    # Retain only transcript hits; metadata never supplies evidence.
                    if match_spans(payload.get('text',''),query,request.transcript_contract):
                        if len(rows)<1000:rows.append({'id':str(point.id),**payload})
                        else:limited=True
                elif request.filter_type=='title':
                    score=title_score(payload.get('video_title',''),query)
                    if score:
                        videos[payload['video_id']]={'id':str(point.id),**payload,'score':score,'is_video_only':True,
                            'match_types':['simple_title_filter'],'matched_field':'video_title'}
                        if len(videos)>1000:
                            worst=min(videos,key=lambda vid:(videos[vid]['score'],-int(vid)))
                            del videos[worst];limited=True
                else:
                    hits=[field for field in SUMMARY_FIELDS if title_score(payload.get(field,'') or '',query)>0]
                    if hits and len(videos)<60:videos[payload['video_id']]={'matched_summary_fields':hits}
                    elif hits and payload['video_id'] not in videos:limited=True
            if next_offset is None:exhausted=True;break
            offset=next_offset
    except RetrievalBudgetReached:limited=True
    metadata={'authority':'python_contract_v1','representation':VERSION,'contract':request.transcript_contract if request.filter_type=='text' else request.filter_type,
              'normalized_index_used':using_index,'count_scope':'retained_result_set_after_filters','scanned_segments':scanned,'scope':'eligible_database_summaries' if supplied_summary is not None else 'indexed_payloads_within_scan_budget','exhaustive':exhausted and not limited}
    if request.filter_type=='text':results=transcript_results(rows,query,request.transcript_contract)
    elif request.filter_type=='title':results=sorted(videos.values(),key=lambda r:(-r['score'],str(r['video_id'])))
    else:
        results=[];validation=None
        if videos:
            vector=embedding(query,8)
            summary_filter=models.Filter(must=conditions+[models.FieldCondition(key='video_id',match=models.MatchAny(any=list(videos)))])
            try:
                found=reader.query_points(collection_name=collection,query=vector,query_filter=summary_filter,limit=60,
                    with_payload=['video_id','text','speaker','diarization_speaker','transcript_speaker','start_time','end_time','timestamp_unit','youtube_url','language','video_created_at'],with_vectors=False).points
                if len(found)>=60 or {p.payload.get('video_id') for p in found if p.payload} != set(videos):
                    limited=True
                candidates=[{'id':str(p.id),**(p.payload or {}),'score':p.score} for p in found if valid_interval(p.payload or {}) and date_matches(p.payload or {},request)]
                results,validation=validate_passages(query,candidates,judge)
                for row in results:
                    row['matched_summary_fields']=videos[row['video_id']]['matched_summary_fields'];row['matched_field']='summary_with_transcript_evidence'
            except RetrievalBudgetReached:limited=True
        metadata['passage_validation']=validation
        metadata['summary_candidate_videos']=len(videos)
    grouped=OrderedDict()
    for row in results:grouped.setdefault(row['video_id'],[]).append(row)
    selected=list(grouped)[:request.top_k]
    limited=limited or len(selected)<len(grouped)
    results=[r for vid in selected for r in grouped[vid]]
    metadata['exhaustive']=metadata['exhaustive'] and not limited
    partial=not metadata['exhaustive'] or bool(metadata.get('passage_validation') and metadata['passage_validation']['status']=='partial')
    metadata['status']='partial' if partial else ('completed' if results else 'no_matches')
    metadata['retryable']=bool(metadata.get('passage_validation') and metadata['passage_validation']['retryable'])
    metadata['counts']={'matching_videos':len(selected),'displayed_passages':0 if request.filter_type=='title' else len(results),
        'matched_transcript_occurrences':sum(r['occurrence_count'] for r in results) if request.filter_type=='text' else None}
    return {'results':results,'search_mode':'simple','filter_type':request.filter_type,'search_contract':metadata,'returned':len(results)}


def search_session_binding(request):
    # Literal queries retain case/spacing; changing any contract/filter restarts.
    fields=('query','words','speaker','title','video_id','language','filter_year','filter_month',
            'filter_date','time_range','search_mode','filter_type','transcript_contract','min_score','max_scanned')
    return {key:getattr(request,key,None) for key in fields}


def indexed_text_fields(text):
    return {'search_contract_version':VERSION,'transcript_normalized_v1':title_key(text or '')}


def normalized_prefilter(query,contract,models):
    key=title_key(query)
    if not key:return None
    equivalents=[key]
    if contract=='alias':
        for aliases in CONFIG['aliases'].values():
            values=[title_key(a) for a in aliases]
            if key in values:equivalents=values;break
    return models.Filter(should=[models.FieldCondition(key='transcript_normalized_v1',match=models.MatchText(text=k)) for k in equivalents])
