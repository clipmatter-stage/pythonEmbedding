"""Bounded, scoped summary matches; generated summaries are not spoken evidence."""
import time
from qdrant_client import models
from contextual_search import source_filter
from speaker_identity import FIELDS as IDENTITY_FIELDS, matches_identity
from transcript_search_contract import normalize, normalized_with_spans

SUMMARY_FIELDS=('video_summary','video_summary_english','video_summary_urdu','summary_en','summary_ur')
VIDEO_FIELDS=['video_id','video_title','youtube_url','language','video_created_at','speaker','diarization_speaker','start_time','end_time']


def summary_excerpts(text, terms):
    normalized,mapping=normalized_with_spans(text)
    excerpts=[]
    for term in sorted(terms):
        position=normalized.find(term)
        if position<0:continue
        start=max(0,mapping[position][0]-80)
        end=min(len(text),max(mapping[position+len(term)-1][1]+80,start+240))
        end=min(end,start+320)
        if any(start<previous['end']and end>previous['start']for previous in excerpts):continue
        excerpts.append({'start':start,'end':end,'text':text[start:end]})
        if len(excerpts)==3:break
    return excerpts


def summary_candidates(base, request, queries, video_ids=None, indexed_fields=()):
    started=time.monotonic();deadline=started+3
    indexed=tuple(field for field in SUMMARY_FIELDS if field in indexed_fields)
    fallback=tuple(field for field in SUMMARY_FIELDS if field not in indexed)
    phases=[(indexed,True),(fallback,False)]
    prepared=[(q,normalize(q.replace('_',' ')))for q in queries]
    prepared=[(q,value,set(value.split()))for q,value in prepared if value]
    maximum=min(getattr(request,'max_scanned',10000),10000)
    scanned=0;videos={};scores={};warnings=[];searched=[];complete=True
    for fields,use_index in phases:
        if not fields:continue
        if scanned>=maximum or time.monotonic()>=deadline:
            complete=False;warnings.append('summary_retrieval_bounded');break
        filters=source_filter(request,video_ids)
        if use_index:
            filters.must.append(models.Filter(should=[models.FieldCondition(key=field,
                match=models.MatchText(text=query))for field in fields for query in queries]))
        offset=None;phase_complete=False;searched.extend(fields)
        while scanned<maximum and time.monotonic()<deadline:
            try:
                points,offset=base.client.scroll(collection_name=base.collection,scroll_filter=filters,
                    limit=min(200,maximum-scanned),offset=offset,
                    with_payload=[*IDENTITY_FIELDS,*VIDEO_FIELDS,*fields],with_vectors=False,timeout=3)
            except Exception:
                warnings.append('summary_retrieval_unavailable');break
            scanned+=len(points);interrupted=False
            for point in points:
                if time.monotonic()>=deadline:interrupted=True;break
                row=point.payload or{};identity=row.get('video_id')
                if identity is None:continue
                if request.speaker and not matches_identity(request.speaker,row):continue
                if request.title and not base.title_matches(request.title,row.get('video_title')or''):continue
                matches=[];best=0.
                for field in fields:
                    if time.monotonic()>=deadline:interrupted=True;break
                    text=row.get(field)
                    if not isinstance(text,str)or not text.strip():continue
                    # Video summaries repeat on source segments. Normalize each
                    # distinct field value once within this bounded lookup.
                    cache_key=(field,text)
                    if cache_key not in scores:
                        value=normalize(text.replace('_',' '));tokens=set(value.split())
                        hits=[(q,1. if wanted==value else .9,terms)for q,wanted,terms in prepared if terms.issubset(tokens)]
                        score=max((score for _,score,_ in hits),default=0.)
                        scores[cache_key]=(score,{'field':field,'matched_queries':[q for q,_,_ in hits],
                            'excerpts':summary_excerpts(text,set().union(*(terms for _,_,terms in hits)))if hits else []})
                    score,evidence=scores[cache_key]
                    if score and score>=request.min_score:matches.append(evidence);best=max(best,score)
                if not matches:continue
                if identity not in videos:
                    videos[identity]={'id':f'summary:{identity}','video_id':identity,
                        'video_title':row.get('video_title',''),'youtube_url':row.get('youtube_url',''),
                        'language':row.get('language'),'video_created_at':row.get('video_created_at'),
                        'text':'','is_video_only':True,'score':best,'score_type':'summary_lexical_match',
                        'matched_field':matches[0]['field'],'matched_fields':[],
                        'matched_summary_fields':[],'match_types':['summary_match'],
                        'evidence_verified':False,'transcript_verified':False,'evidence_source':'summary',
                        'speaker_identity_status':'not_established_by_summary','summary_evidence':[]}
                result=videos[identity];result['score']=max(result['score'],best)
                for evidence in matches:
                    if evidence not in result['summary_evidence']:
                        # At most one bounded excerpt set per matched field/video.
                        if evidence['field']in result['matched_summary_fields']:continue
                        result['summary_evidence'].append(evidence)
                        result['matched_summary_fields'].append(evidence['field'])
                        result['matched_fields'].append(evidence['field'])
            if interrupted:break
            if offset is None:phase_complete=True;break
            if not points:break
        if not phase_complete:complete=False;warnings.append('summary_retrieval_bounded')
    return sorted(videos.values(),key=lambda r:(-r['score'],str(r['video_id']))),{
        'source':'video_summary','method':'indexed_lexical_with_bounded_fallback'if indexed and fallback else 'indexed_lexical'if indexed else 'bounded_lexical_scan',
        'queries':queries,'indexed_fields':list(indexed),'searched_fields':searched,
        'scanned_segments':scanned,'matching_videos':len(videos),'exhaustive':complete,
        'warnings':sorted(set(warnings)),'elapsed_seconds':round(time.monotonic()-started,3)}


def merge_summary_candidates(results, summaries, limit):
    """Attach summary evidence without replacing verified transcript passages."""
    by_video={row['video_id']:row for row in summaries};out=[];seen=set()
    for row in results:
        seen.add(row['video_id']);result=dict(row);summary=by_video.get(row['video_id'])
        if summary:
            result.update(summary_evidence=summary['summary_evidence'],matched_summary_fields=summary['matched_summary_fields'],
                matched_fields=list(dict.fromkeys([*row.get('matched_fields',['text']),*summary['matched_fields']])),
                match_types=list(dict.fromkeys([*row.get('match_types',[]),'summary_match'])))
            sources=[]
            if row.get('title_evidence'):sources.append('title')
            sources.append('summary')
            if row.get('transcript_verified'):sources.append('transcript')
            result['evidence_source']='_and_'.join(sources)
        out.append(result)
    out.extend(row for row in summaries if row['video_id']not in seen)
    # Preserve verified-first ordering; metadata relevance is not a model score.
    return out[:limit]
