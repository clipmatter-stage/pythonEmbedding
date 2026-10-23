"""Read-only model evaluation; run explicitly in Railway, never in unit tests.

Uses synthetic controls plus the previously authorized video-592 source for
the exact query 'namaz ki talqeen'. No cache, index or deployed file is changed.
Output contains source hashes and verdicts, never transcript excerpts.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import time

from openai import OpenAI
from qdrant_client import QdrantClient, models
from redis import Redis

from grounded_search import FIELDS, contextual_candidate
from grounded_validation import candidate_id, provider_judge, validate_grounded, VERSION, SYSTEM
from speaker_identity import view


def live_reference():
    client = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=10)
    collection = os.getenv('SEGMENTS_COLLECTION', 'video_transcript_segments')
    identity = 'e3da037a-0ac0-5717-bcb3-f3938ae77f61'
    points = client.retrieve(collection, ids=[identity], with_payload=True, with_vectors=False)
    if len(points) != 1:
        raise RuntimeError('Live reference point missing')
    focus = {'id': str(points[0].id), **view(points[0].payload or {})}
    if (focus.get('video_id') != 592 or focus.get('processing_status') != 'completed'
            or focus.get('approval_status') != 'approved' or focus.get('is_archived') is not False):
        raise RuntimeError('Live reference is not eligible')
    conditions = [models.FieldCondition(key='video_id', match=models.MatchValue(value=592)),
        models.FieldCondition(key='start_time', range=models.Range(gte=max(0, focus['start_time']-20), lte=focus['end_time']+20))]
    for key, value in [('processing_status', 'completed'), ('approval_status', 'approved'), ('is_archived', False)]:
        conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    neighbors, offset = client.scroll(collection, scroll_filter=models.Filter(must=conditions),
        limit=100, with_payload=FIELDS, with_vectors=False)
    if offset is not None:
        raise RuntimeError('Reference context truncated')
    result = contextual_candidate(focus, [{'id':str(p.id), **view(p.payload or {})} for p in neighbors])
    client.close()
    return result


def frozen_website_batch(reference):
    """Reconstruct the actual six-passage batch of a cached failed request."""
    redis = Redis.from_url(os.getenv('SEARCH_CACHE_REDIS_URL') or os.environ['REDIS_URL'])
    sessions=[]
    for index, key in enumerate(redis.scan_iter(match='contextual:v8:session:*', count=100)):
        if index >= 500:break
        data=json.loads(redis.get(key) or '{}');params=data.get('query_params',{})
        if params.get('query','').strip().casefold()!='namaz ki talqeen':continue
        trace=params.get('passage_validation',{}).get('trace',[])
        if any(t.get('candidate_id')==candidate_id(reference) and not t.get('accepted') for t in trace):
            sessions.append((data.get('timestamp',0),trace))
    if sessions:
        trace=max(sessions,key=lambda item:item[0])[1]
        index=next(i for i,t in enumerate(trace)if t.get('candidate_id')==candidate_id(reference))
        selected=trace[index//6*6:index//6*6+6]
    else:
        fixture=pathlib.Path(__file__).with_name('fixtures')/'grounded_website_batch.json'
        selected=json.loads(fixture.read_text())['sources']
    client=QdrantClient(url=os.environ['QDRANT_URL'],api_key=os.environ['QDRANT_API_KEY'],timeout=10)
    collection=os.getenv('SEGMENTS_COLLECTION','video_transcript_segments');rows=[]
    for entry in selected:
        point=client.retrieve(collection,ids=[entry['point_id']],with_payload=True,with_vectors=False)[0]
        focus={'id':str(point.id),**view(point.payload or{})}
        conditions=[models.FieldCondition(key='video_id',match=models.MatchValue(value=focus['video_id'])),
            models.FieldCondition(key='start_time',range=models.Range(gte=max(0,focus['start_time']-20),lte=focus['end_time']+20))]
        for key,value in [('processing_status','completed'),('approval_status','approved'),('is_archived',False)]:
            conditions.append(models.FieldCondition(key=key,match=models.MatchValue(value=value)))
        neighbors,offset=client.scroll(collection,scroll_filter=models.Filter(must=conditions),limit=100,with_payload=FIELDS,with_vectors=False)
        if offset is not None:raise RuntimeError('Frozen batch context truncated')
        passage=contextual_candidate(focus,[{'id':str(p.id),**view(p.payload or{})}for p in neighbors])
        if candidate_id(passage)!=entry['candidate_id']:raise RuntimeError('Frozen website source changed')
        rows.append(passage)
    client.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--baseline', action='store_true')
    arguments = parser.parse_args()
    if not 2 <= arguments.rounds <= 10:
        raise ValueError('Use 2-10 fixed, non-adaptive repetitions')
    reference = live_reference()
    website_batch = frozen_website_batch(reference)
    cases = json.loads(pathlib.Path(__file__).with_name('fixtures').joinpath('grounded_role_cases.json').read_text())['cases']
    original = 'namaz ki talqeen'
    groups = collections.defaultdict(list)
    for index, case in enumerate(cases):
        row = {'id':'synthetic_'+case['id'], 'video_id':100000+index, 'text':case['text'],
               'start_time':1, 'end_time':15, 'segment_ids':['synthetic_'+case['id']]}
        groups[case['query']].append((row, case['relevant'], case['id']))
    groups[original].append((reference, True, 'live_592'))
    model = os.getenv('SEARCH_JUDGE_MODEL', os.getenv('OPENAI_TEXT_MODEL', 'gpt-6-luna'))
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    arms = [('updated', provider_judge(client, model, os.getenv('GROUNDING_CITATION_MODE', 'span_id')))]
    if arguments.baseline:
        spec = importlib.util.spec_from_file_location('baseline_grounding', '/app/grounded_validation.py')
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        arms.insert(0, ('deployed_baseline', baseline.provider_judge(client, model, os.getenv('GROUNDING_CITATION_MODE', 'span_id'))))
    output = {'model':model, 'rules_version':VERSION, 'system_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'live_reference_id':candidate_id(reference), 'live_reference_text_sha256':hashlib.sha256(reference['text'].encode()).hexdigest(),
        'website_batch_sources':[{'point_id':str(row['id']),'candidate_id':candidate_id(row)}for row in website_batch],
        'rounds':arguments.rounds, 'arms':{}}
    for label, judge in arms:
        runs = []; total = collections.Counter()
        # Five distinct, reproducible batch arrangements. Do not retry a valid
        # negative judgment until it becomes positive or drop failed runs.
        for query, entries in groups.items():
            for repeat in range(arguments.rounds if query == original else 1):
                ordered = list(entries); random.Random(1729+repeat).shuffle(ordered)
                expected = {candidate_id(row):(yes, name) for row, yes, name in ordered}
                started = time.monotonic()
                try:
                    passed, metadata = validate_grounded(query, [row for row,_,_ in ordered], judge,
                        batch_size=6, max_calls=4, deadline_seconds=40, concurrency=2)
                    positive = {candidate_id(row) for row in passed}
                    trace = {row['candidate_id']:row for row in metadata['trace']}
                    errors = [name for key,(yes,name) in expected.items() if (key in positive) != yes]
                    for key,(yes,name) in expected.items():
                        total['true_positive' if yes and key in positive else 'false_negative' if yes else
                              'false_positive' if key in positive else 'true_negative'] += 1
                    live = next((trace.get(key) for key,(_,name) in expected.items() if name == 'live_592'), None)
                    run = {'query':query, 'repeat':repeat, 'status':metadata['status'],
                        'mismatches':errors, 'live_592':live, 'provider_calls':metadata['provider_calls']}
                    if metadata['status']=='partial':total['incomplete_runs'] += 1
                except Exception as exc:
                    total['failed_runs'] += 1
                    run = {'query':query, 'repeat':repeat, 'error_type':type(exc).__name__}
                run['seconds'] = round(time.monotonic()-started, 3); runs.append(run)
                print(json.dumps({'arm':label, **run}, ensure_ascii=False), flush=True)
        for repeat in range(arguments.rounds):
            ordered=list(website_batch);random.Random(592+repeat).shuffle(ordered)
            started=time.monotonic()
            try:
                passed,metadata=validate_grounded(original,ordered,judge,batch_size=6,max_calls=2,deadline_seconds=35,concurrency=1)
                accepted=any(candidate_id(row)==candidate_id(reference)for row in passed)
                total['website_live_positive' if accepted else 'website_live_missed']+=1
                if metadata['status']=='partial':total['incomplete_runs']+=1
                run={'query':original,'batch':'frozen_website','repeat':repeat,'live_592_accepted':accepted,
                    'status':metadata['status'],'provider_calls':metadata['provider_calls'],
                    'live_592':[t for t in metadata['trace']if t['candidate_id']==candidate_id(reference)]}
            except Exception as exc:
                total['failed_runs']+=1;run={'batch':'frozen_website','repeat':repeat,'error_type':type(exc).__name__}
            run['seconds']=round(time.monotonic()-started,3);runs.append(run)
            print(json.dumps({'arm':label,**run},ensure_ascii=False),flush=True)
        output['arms'][label] = {'counts':dict(total), 'runs':runs}
    print('EVALUATION_SUMMARY='+json.dumps(output, ensure_ascii=False), flush=True)
    updated = output['arms']['updated']['counts']
    return int(any(updated.get(key, 0) for key in ['false_negative','false_positive','failed_runs','incomplete_runs','website_live_missed']))


if __name__ == '__main__':
    raise SystemExit(main())
