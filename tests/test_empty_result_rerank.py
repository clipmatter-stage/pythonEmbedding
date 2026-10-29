"""Fix A: one extra judged batch from the Urdu-script lane after a clean strict no-match.

Mocked Qdrant (in-memory) and a deterministic judge; no provider calls.
"""
import copy,json,os,pathlib,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient,models
from fast_search import FastSemanticSearch,FastSearchSettings
import grounded_search
from grounded_search import GroundedSearch,snapshot_key,urdu_lane_candidates
from grounded_validation import candidate_id
from test_fast_search import request

ROMAN='Gaza aur hathyar pohanchana'
QUERIES=[ROMAN,'غزہ تک ہتھیار پہنچانا','Gaza tak hathyar pohanchana','Delivering weapons to Gaza']
MATCH='غزہ کے لوگوں تک ہتھیار پہنچانا ہماری ذمہ داری ہے'
LIVE_LIMITS={'GROUNDING_INITIAL_CANDIDATES':'8','GROUNDING_MAX_CANDIDATES':'16'}
VOLATILE={'elapsed_seconds','total_seconds','retrieval_seconds','context_seconds','validation_seconds',
          'title_seconds','summary_seconds','diagnostic_id','elapsed_ms'}

def stable(value):
    """Drop wall-clock and random diagnostic fields before a byte comparison."""
    if isinstance(value,dict):return {k:stable(v)for k,v in value.items()if k not in VOLATILE}
    if isinstance(value,list):return [stable(v)for v in value]
    return value

class EmptyResultRerankTests(unittest.TestCase):
    def setUp(self):
        self.client=QdrantClient(':memory:')
        self.client.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
        self.calls=0;self.judged=[]
        env=patch.dict(os.environ,LIVE_LIMITS);env.start();self.addCleanup(env.stop)
        os.environ.pop('GROUNDED_EMPTY_RESULT_URDU_RERANK',None)

    def tearDown(self):self.client.close()

    def judge(self,query,rows,timeout):
        self.calls+=1;out=[]
        for row in rows:
            self.judged.append(row['video_id'])
            yes='ہتھیار' in row['text'] and 'غزہ' in row['text']
            quotes=['ہتھیار']if yes else []
            out.append({'candidate_id':candidate_id(row),'score':.9 if yes else .2,'relevant':yes,'complete':yes,
                        'roles_supported':yes,'relationship_supported':yes,'quotes':quotes,
                        'reason':'relevant'if yes else 'related_only'})
        return {'judgments':out}

    def engine(self,rows,queries,cache=None):
        base=FastSemanticSearch(self.client,'source',lambda q:[1.,0.],lambda q:{'semantic_query':q},
            lambda q,v:True,lambda q,v:True,settings=FastSearchSettings(candidate_limit=100))
        cache={}if cache is None else cache
        engine=GroundedSearch(base,self.judge,'fixture-model',lambda:'epoch',cache.get,
            lambda k,v,t:cache.update({k:v}),('same-model',2))
        base.search=lambda request,video_ids=None:{'results':copy.deepcopy(rows),'speaker_filter':None,
            'query_decomposition':{},'ranking':{'retrieval_queries':list(queries),'retrieval_warnings':[]}}
        return engine,cache

    def corpus(self,urdu_lane,relevant_in_strict=False,match_position=22,total=24):
        """`total` one-passage videos ranked by the fused score; 874 sits at `match_position`."""
        rows=[];videos=[v for v in range(1,total+1)if v!=874][:total-1]
        videos.insert(match_position-1,874)
        for position,video in enumerate(videos,1):
            text=MATCH if video==874 else f'غزہ کے بارے میں عام گفتگو نمبر {position}'
            if relevant_in_strict and position==3:text='غزہ میں ہتھیار پہنچانے کی بات ہو رہی ہے'
            point=position
            self.client.upsert('source',[models.PointStruct(id=point,vector=[1.,0.],payload={
                'video_id':video,'text':text,'speaker':'Unverified Label','transcript_speaker':'A',
                'start_time':10.,'end_time':18.,'processing_status':'completed','approval_status':'approved',
                'is_archived':False})])
            lanes=[{'lane':0,'kind':'semantic','rank':position,'depth':0,'weight':1.}]
            if video in urdu_lane:lanes.append({'lane':1,'kind':'semantic','rank':urdu_lane.index(video)+1,'depth':0,'weight':1.})
            score=1-position/100
            rows.append({'id':point,'video_id':video,'text':text,'speaker':'Unverified Label','transcript_speaker':'A',
                'start_time':10.,'end_time':18.,'score':score,'retrieval_score':score,'segment_ids':[point],
                'retrieval_lanes':lanes,'match_count':1})
        return rows

    def run_search(self,rows,queries=QUERIES,query=ROMAN,flag=None,cache=None):
        self.calls=0;self.judged=[]
        env={}if flag is None else {'GROUNDED_EMPTY_RESULT_URDU_RERANK':flag}
        with patch.dict(os.environ,env):
            engine,cache=self.engine(rows,queries,cache)
            result=engine.search(request(query=query,min_score=0.))
        return result,self.calls,cache

    # (a) Roman Gaza: 874 is passage #22 in the fused ranking but #1 in the Urdu lane.
    def test_roman_gaza_match_beyond_strict_cut_is_judged_from_urdu_lane(self):
        rows=self.corpus(urdu_lane=[874,5,6])
        strict,strict_calls,_=self.run_search(rows,flag='false')
        self.assertEqual([],strict['results'])
        self.assertNotIn(874,self.judged)
        self.assertNotIn('empty_result_rerank',strict['metadata']['passage_validation'])
        result,calls,cache=self.run_search(rows)
        self.assertEqual([874],[r['video_id']for r in result['results']])
        info=result['metadata']['passage_validation']['empty_result_rerank']
        self.assertEqual(('completed',1,1),(info['status'],info['accepted'],info['lane_index']))
        self.assertGreater(calls,strict_calls)
        self.assertLessEqual(calls,6)  # Same total provider-call budget as the strict pass.
        self.assertEqual('completed',result['metadata']['passage_validation']['status'])
        self.assertTrue(any(k.startswith('grounded:')for k in cache))  # Non-empty -> saved as usual.

    def test_extra_pass_never_rejudges_strict_candidates(self):
        rows=self.corpus(urdu_lane=[1,2,874])
        self.run_search(rows)
        self.assertEqual(1,self.judged.count(1));self.assertEqual(1,self.judged.count(874))

    # (b) Any strict result: no extra pass, identical output and call count.
    def test_query_with_strict_result_never_runs_extra_pass(self):
        rows=self.corpus(urdu_lane=[874],relevant_in_strict=True)
        off,off_calls,_=self.run_search(rows,flag='false')
        on,on_calls,_=self.run_search(rows)
        self.assertTrue(on['results']);self.assertEqual(off_calls,on_calls)
        self.assertNotIn(874,self.judged)
        self.assertNotIn('empty_result_rerank',on['metadata']['passage_validation'])
        self.assertEqual(json.dumps(stable(off),sort_keys=True,ensure_ascii=False,default=str),
                         json.dumps(stable(on),sort_keys=True,ensure_ascii=False,default=str))

    # (c) Short and long queries that already return results are untouched.
    def test_short_and_long_working_queries_are_unchanged(self):
        for query,queries in [('Gaza',['Gaza (gaza; ghaza; غزہ)','غزہ','Gaza']),
                              ('elite and mafias causing public hardship',
                               ['elite and mafias causing public hardship','اشرافیہ اور مافیا عوام کو مشکلات میں ڈال رہے ہیں۔',
                                'Elite aur mafia awam ko mushkilat mein daal rahe hain.'])]:
            with self.subTest(query=query):
                self.client.delete_collection('source')
                self.client.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
                rows=self.corpus(urdu_lane=[874],relevant_in_strict=True)
                off,off_calls,_=self.run_search(rows,queries=queries,query=query,flag='false')
                on,on_calls,_=self.run_search(rows,queries=queries,query=query)
                self.assertTrue(on['results']);self.assertEqual(off_calls,on_calls)
                self.assertEqual(json.dumps(stable(off),sort_keys=True,ensure_ascii=False,default=str),
                                 json.dumps(stable(on),sort_keys=True,ensure_ascii=False,default=str))

    def test_zero_result_without_urdu_expansion_adds_no_provider_call(self):
        rows=self.corpus(urdu_lane=[874])
        off,off_calls,_=self.run_search(rows,queries=[ROMAN,'Delivering weapons to Gaza'],flag='false')
        on,on_calls,_=self.run_search(rows,queries=[ROMAN,'Delivering weapons to Gaza'])
        self.assertEqual(([],off_calls),(on['results'],on_calls))
        self.assertEqual('no_urdu_script_expansion',on['metadata']['passage_validation']['empty_result_rerank']['reason'])
        self.assertEqual('no_matches',on['metadata']['passage_validation']['status'])

    def test_kill_switch_ignores_saved_results_found_only_by_extra_pass(self):
        rows=self.corpus(urdu_lane=[874])
        first,_,cache=self.run_search(rows)
        self.assertTrue(first['results'])
        replay,calls,_=self.run_search(rows,cache=cache)
        self.assertEqual(0,calls);self.assertTrue(replay['metadata']['passage_validation']['cache_hit'])
        killed,calls,_=self.run_search(rows,flag='false',cache=cache)
        self.assertEqual([],killed['results']);self.assertGreater(calls,0)

    def test_urdu_lane_order_skips_judged_and_original_lane(self):
        rows=[{'id':i,'retrieval_lanes':[{'lane':1,'kind':'semantic','rank':r,'depth':0}]}for i,r in [(1,3),(2,1),(3,2)]]
        lane,picked=urdu_lane_candidates(rows,['غزہ','غزہ تک','Gaza'],{'2'},2)
        self.assertEqual((1,[3,1]),(lane,[r['id']for r in picked]))
        self.assertEqual((None,[]),urdu_lane_candidates(rows,['غزہ','Gaza'],set(),2))

class CarryOverTests(unittest.TestCase):
    """(d) Saved non-empty results from the previous rules fingerprint keep being served."""
    def setUp(self):
        self.client=QdrantClient(':memory:')
        self.client.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
        self.calls=0
        self.client.upsert('source',[models.PointStruct(id=1,vector=[1.,0.],payload={'video_id':874,'text':MATCH,
            'speaker':'Unverified Label','transcript_speaker':'A','start_time':1.,'end_time':5.,
            'processing_status':'completed','approval_status':'approved','is_archived':False})])
        self.cache={}
        base=FastSemanticSearch(self.client,'source',lambda q:[1.,0.],lambda q:{'semantic_query':q},
            lambda q,v:True,lambda q,v:True,settings=FastSearchSettings(candidate_limit=100))
        self.engine=GroundedSearch(base,self.judge,'fixture-model',lambda:'epoch',lambda k:self.cache.get(k),
            lambda k,v,t:self.cache.update({k:v}),('same-model',2))
    def tearDown(self):self.client.close()
    def judge(self,query,rows,timeout):
        self.calls+=1
        return {'judgments':[{'candidate_id':candidate_id(r),'score':.9,'relevant':True,'complete':True,
            'roles_supported':True,'relationship_supported':True,'quotes':['ہتھیار پہنچانا'],'reason':'relevant'}for r in rows]}
    def saved_under(self,rules,query,result):
        params=vars(query).copy()
        for key in ['cursor','exact_cursor','search_session_id','batch_size']:params.pop(key,None)
        initial=int(os.getenv('GROUNDING_INITIAL_CANDIDATES','12'));maximum=int(os.getenv('GROUNDING_MAX_CANDIDATES','24'))
        self.cache={snapshot_key(self.engine,'epoch',params,None,initial,maximum,rules_fingerprint=rules):result}

    def test_previous_fingerprint_snapshot_is_served_without_rejudgment(self):
        query=request(query='غزہ اور ہتھیار پہنچانا',min_score=0.)
        with patch('grounded_search.search_rules_fingerprint',return_value='old-rules'):
            previous=self.engine.search(query)
        self.assertTrue(previous['results'])
        self.saved_under('old-rules',query,previous);calls=self.calls
        with patch('grounded_search.search_rules_fingerprint',return_value='new-rules'),\
             patch('grounded_search.carried_rules_fingerprints',return_value=['old-rules']):
            result=self.engine.search(query)
        self.assertEqual(calls,self.calls);self.assertEqual(previous['results'],result['results'])
        self.assertTrue(result['metadata']['passage_validation']['cache_hit'])
        self.assertEqual(2,len(self.cache))  # Copied under the new key; the old key stays for rollback.

    def test_empty_previous_snapshot_is_not_carried(self):
        query=request(query='غزہ اور ہتھیار پہنچانا',min_score=0.)
        self.saved_under('old-rules',query,{'results':[],'metadata':{'passage_validation':{'provider_calls':1}},'ranking':{'timings':{}}})
        with patch('grounded_search.search_rules_fingerprint',return_value='new-rules'),\
             patch('grounded_search.carried_rules_fingerprints',return_value=['old-rules']):
            result=self.engine.search(query)
        self.assertGreater(self.calls,0);self.assertTrue(result['results'])

    def test_carry_over_is_bound_to_the_reviewed_rules(self):
        migration=json.loads((pathlib.Path(grounded_search.__file__).parent/'search_snapshot_migration.json').read_text())
        carry=migration['persistent_carry_over']
        # Fails on any later search-rule edit: re-review the carry-over, then update the fingerprint.
        self.assertEqual(grounded_search.search_rules_fingerprint(),carry['compatible_rules_fingerprint'])
        self.assertIn('2605f5a769c97483c8250bf66347de5ef27c1c0e5d816be00a48b9913e34b2cc',carry['previous_rules_fingerprints'])
        self.assertEqual(carry['previous_rules_fingerprints'],grounded_search.carried_rules_fingerprints())
        with patch('grounded_search.search_rules_fingerprint',return_value='edited-rules'):
            self.assertEqual([],grounded_search.carried_rules_fingerprints())

if __name__=='__main__':unittest.main()
