"""Scoped lexical title candidates; a title never establishes spoken evidence."""
import time
from qdrant_client import models
from contextual_search import source_filter
from transcript_search_contract import normalize

TITLE_FIELDS=['video_id','video_title','youtube_url','language','video_created_at','speaker','diarization_speaker','start_time','end_time']


def title_candidates(base, request, queries, video_ids=None, indexed=False):
    started=time.monotonic();deadline=started+3
    filters=source_filter(request,video_ids)
    if indexed:
        filters.must.append(models.Filter(should=[models.FieldCondition(
            key='video_title',match=models.MatchText(text=query))for query in queries]))
    offset=None;scanned=0;videos={};complete=False;warnings=[]
    prepared=[(q,normalize(q.replace('_',' ')))for q in queries]
    prepared=[(q,value,set(value.split()))for q,value in prepared if value]
    scores={}
    maximum=min(getattr(request,'max_scanned',10000),10000)
    while scanned<maximum and time.monotonic()<deadline:
        points,offset=base.client.scroll(collection_name=base.collection,scroll_filter=filters,
            limit=min(500,maximum-scanned),offset=offset,with_payload=TITLE_FIELDS,with_vectors=False,timeout=3)
        scanned+=len(points)
        for point in points:
            row=point.payload or{};identity=row.get('video_id');title=row.get('video_title')or''
            if identity is None or not title:continue
            if request.speaker and not base.speaker_matches(request.speaker,f"{row.get('speaker','')} {row.get('diarization_speaker','')}"):continue
            if request.title and not base.title_matches(request.title,title):continue
            if title not in scores:
                normalized=normalize(title.replace('_',' '));tokens=set(normalized.split())
                matched=[(q,1. if value==normalized else .9)for q,value,wanted in prepared if wanted.issubset(tokens)]
                scores[title]=(max((score for _,score in matched),default=0),[q for q,_ in matched])
            score,matched_queries=scores[title]
            if score<request.min_score or not score:continue
            result={'id':f'title:{identity}','video_id':identity,'video_title':title,
                'youtube_url':row.get('youtube_url',''),'language':row.get('language'),
                'video_created_at':row.get('video_created_at'),'text':'','is_video_only':True,
                'score':score,'score_type':'title_lexical_match','matched_field':'video_title',
                'matched_fields':['video_title'],'match_types':['title_match'],
                'evidence_verified':False,'transcript_verified':False,
                'evidence_source':'title','speaker_identity_status':'not_established_by_title',
                'title_evidence':{'title':title,'matched_queries':matched_queries}}
            if identity not in videos or score>videos[identity]['score']:videos[identity]=result
        if offset is None:complete=True;break
        if not points:break
    if not complete:warnings.append('title_retrieval_bounded')
    return sorted(videos.values(),key=lambda r:(-r['score'],str(r['video_id']))),{
        'source':'video_title','method':'indexed_lexical'if indexed else 'bounded_lexical_scan',
        'queries':queries,'scanned_segments':scanned,'matching_videos':len(videos),
        'exhaustive':complete,'warnings':warnings,'elapsed_seconds':round(time.monotonic()-started,3)}


def merge_title_candidates(passages,titles,limit):
    by_video={row['video_id']:row for row in titles};out=[];verified=set()
    for row in passages:
        result={**row,'evidence_verified':True,'transcript_verified':True,'evidence_source':'transcript'}
        verified.add(row['video_id'])
        if row['video_id']in by_video:
            result.update(evidence_source='title_and_transcript',matched_fields=['text','video_title'],
                          title_evidence=by_video[row['video_id']]['title_evidence'])
            result['match_types']=[*row.get('match_types',[]),'title_match']
        out.append(result)
    out.extend(row for row in titles if row['video_id']not in verified)
    # Verified passage ranks and title ranks have different meanings. Keep
    # verified passages first, then title-only videos; never compare their scores.
    return out[:limit]
