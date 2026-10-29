"""Fix B: "related, not verified" fallback after the strict pass and the Fix A pass both end empty.

Mocked Qdrant (in-memory) and a deterministic judge; no provider or network calls.
"""
import copy,json,os,pathlib,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient,models
from fast_search import FastSemanticSearch,FastSearchSettings
import grounded_search
from grounded_search import GroundedSearch,snapshot_key,FALLBACK_FLAG,is_related_fallback
from grounded_validation import candidate_id
from test_fast_search import request

LIVE_LIMITS={'GROUNDING_INITIAL_CANDIDATES':'8','GROUNDING_MAX_CANDIDATES':'16'}
VOLATILE={'elapsed_seconds','total_seconds','retrieval_seconds','context_seconds','validation_seconds',
          'title_seconds','summary_seconds','diagnostic_id','elapsed_ms'}
LIVE_FP='4ca8868d866d8dd457d78d84ea2817e220ad4a1268fca49fbe039d046fa47b35'
PRE_FIX_A_FP='2605f5a769c97483c8250bf66347de5ef27c1c0e5d816be00a48b9913e34b2cc'
MAKKAH='Agreements made in Makkah between political parties and the government'
MAKKAH_EXP=[MAKKAH,'مکہ میں سیاسی جماعتوں اور حکومت کے درمیان معاہدے','Makkah mein siyasi jamaaton aur hukumat ke darmiyan muahiday']
STUDENTS='Students being encouraged to pray'
STUDENTS_UR='طلبہ کو نماز کی طرف'
STUDENTS_EXP=['طلبہ کو نماز کی طرف','Talaba ko namaz ki taraf','Students being encouraged to pray']
HAFIZ_UR='حافظ نعیم نوجوانوں کو نماز کی تلقین'
HAFIZ_EXP=[HAFIZ_UR,'Hafiz Naeem nojawano ko namaz ki talqeen','Hafiz Naeem urging youth to pray']

def stable(value):
    if isinstance(value,dict):return {k:stable(v)for k,v in value.items()if k not in VOLATILE}
    if isinstance(value,list):return [stable(v)for v in value]
    return value

def dump(value):return json.dumps(stable(value),sort_keys=True,ensure_ascii=False,default=str)

class Base(unittest.TestCase):
    def setUp(self):
        self.client=QdrantClient(':memory:')
        self.client.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
        self.calls=0;self.verdicts={}
        env=patch.dict(os.environ,LIVE_LIMITS);env.start();self.addCleanup(env.stop)
        for name in ['GROUNDED_EMPTY_RESULT_URDU_RERANK','GROUNDED_RELATED_UNVERIFIED_FALLBACK']:os.environ.pop(name,None)
    def tearDown(self):self.client.close()

    def judge(self,query,rows,timeout):
        """verdicts: video_id -> (score, reason, accepted). Default: unrelated, 0.05."""
        self.calls+=1;out=[]
        for row in rows:
            score,reason,yes=self.verdicts.get(row['video_id'],(.05,'related_only',False))
            out.append({'candidate_id':candidate_id(row),'score':score,'relevant':yes,'complete':yes,
                'roles_supported':yes,'relationship_supported':yes,'quotes':[row['text'][:12]]if yes else [],
                'reason':reason,'asr_interpretation':None})
        return {'judgments':out}

    def corpus(self,texts):
        """texts: list of (video_id, text) in fused-rank order; one passage per entry."""
        rows=[]
        for position,(video,text) in enumerate(texts,1):
            self.client.upsert('source',[models.PointStruct(id=position,vector=[1.,0.],payload={
                'video_id':video,'text':text,'speaker':'Unverified Label','transcript_speaker':'A',
                'start_time':10.*position,'end_time':10.*position+8,'processing_status':'completed',
                'approval_status':'approved','is_archived':False})])
            score=1-position/100
            rows.append({'id':position,'video_id':video,'text':text,'speaker':'Unverified Label','transcript_speaker':'A',
                'start_time':10.*position,'end_time':10.*position+8,'score':score,'retrieval_score':score,
                'segment_ids':[position],'retrieval_lanes':[{'lane':0,'kind':'semantic','rank':position,'depth':0,'weight':1.}],
                'match_count':1})
        return rows

    def run_search(self,rows,query,queries,fallback=None,rerank=None,cache=None):
        self.calls=0
        env={}
        if fallback is not None:env['GROUNDED_RELATED_UNVERIFIED_FALLBACK']=fallback
        if rerank is not None:env['GROUNDED_EMPTY_RESULT_URDU_RERANK']=rerank
        cache={}if cache is None else cache
        with patch.dict(os.environ,env):
            base=FastSemanticSearch(self.client,'source',lambda q:[1.,0.],lambda q:{'semantic_query':q},
                lambda q,v:True,lambda q,v:True,settings=FastSearchSettings(candidate_limit=100))
            engine=GroundedSearch(base,self.judge,'fixture-model',lambda:'epoch',cache.get,
                lambda k,v,t:cache.update({k:v}),('same-model',2))
            base.search=lambda request,video_ids=None:{'results':copy.deepcopy(rows),'speaker_filter':None,
                'query_decomposition':{},'ranking':{'retrieval_queries':list(queries),'retrieval_warnings':[]}}
            result=engine.search(request(query=query,min_score=0.))
        return result,self.calls,cache

    def assert_fallback(self,result,videos):
        self.assertEqual(videos,[r['video_id']for r in result['results']])
        for row in result['results']:
            self.assertEqual('related_unverified',row['verification_status'])
            self.assertFalse(row['evidence_verified']);self.assertFalse(row['transcript_verified'])
            self.assertEqual(['related_unverified'],row['match_types'])
            self.assertNotIn('passage_evidence',row)
        validation=result['metadata']['passage_validation']
        self.assertEqual(FALLBACK_FLAG,result['fallback'])
        self.assertEqual(FALLBACK_FLAG,validation['fallback'])
        self.assertEqual(FALLBACK_FLAG,result['ranking']['fallback'])
        self.assertFalse(result['ranking']['evidence_verified'])
        self.assertEqual('no_matches',validation['status'])  # Truthful: nothing was verified.
        self.assertEqual(len(videos),validation['related_unverified_count'])

    def assert_not_saved(self,cache):
        self.assertFalse([k for k in cache if k.startswith('grounded:')])

def filler(n,start=5000):return [(start+i,f'عام سیاسی گفتگو نمبر {i}')for i in range(n)]

class RelatedFallbackTests(Base):
    # Long Makkah query: 1087 judged 0.45 insufficient_evidence -> shown as related_unverified.
    def test_long_makkah_insufficient_evidence_is_returned_unverified(self):
        rows=self.corpus([(1087,'مکہ میں جماعتوں کے درمیان معاہدہ ہوا تھا'),*filler(12)])
        self.verdicts={1087:(.45,'insufficient_evidence',False)}
        off,off_calls,_=self.run_search(rows,MAKKAH,MAKKAH_EXP,fallback='false')
        on,on_calls,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP)
        self.assertEqual([],off['results']);self.assertNotIn('fallback',off)
        self.assert_fallback(on,[1087])
        self.assertEqual(.45,on['results'][0]['score']);self.assertEqual('insufficient_evidence',on['results'][0]['judge_reason'])
        self.assertEqual(off_calls,on_calls)  # Uses the existing judgments only: no extra judge call.
        self.assertIn('empty_result_rerank',on['metadata']['passage_validation'])  # Fix A ran first.
        self.assert_not_saved(cache)

    def test_students_english_and_urdu_cap_three_videos_best_passage_each(self):
        texts=[(588,'طلبہ نماز کے بارے میں بات'),(588,'طلبہ کی تعلیم اور نماز'),(601,'نوجوان اور عبادت'),
               (700,'اساتذہ طلبہ کو نصیحت'),(702,'نماز کی اہمیت'),(800,'کرکٹ کا میچ'),(900,'نماز نہ پڑھنے کی بات'),*filler(6)]
        rows=self.corpus(texts)
        self.verdicts={588:(.55,'related_only',False),601:(.42,'specificity',False),700:(.48,'role_mismatch',False),
                       702:(.41,'insufficient_evidence',False),800:(.30,'related_only',False),900:(.6,'negation',False)}
        for query in [STUDENTS,STUDENTS_UR]:
            with self.subTest(query=query):
                off,off_calls,_=self.run_search(rows,query,STUDENTS_EXP,fallback='false')
                on,on_calls,cache=self.run_search(rows,query,STUDENTS_EXP)
                self.assertEqual([],off['results'])
                # 588 .55 > 700 .48 > 601 .42; 702 dropped by the cap, 800 below 0.4, 900 negation (.6) never shown.
                self.assert_fallback(on,[588,700,601])
                self.assertEqual(1,sum(r['video_id']==588 for r in on['results']))
                self.assertEqual(off_calls,on_calls);self.assert_not_saved(cache)

    def test_hafiz_naeem_urdu_1105_live_scores_stay_empty(self):
        # Live judgments for 1105 were 0.22 role_mismatch and 0.16: below 0.4, so nothing is shown.
        rows=self.corpus([(1105,'حافظ نعیم نوجوانوں سے خطاب'),(588,'نوجوان اور نماز'),*filler(10)])
        self.verdicts={1105:(.22,'role_mismatch',False),588:(.16,'related_only',False)}
        off,off_calls,_=self.run_search(rows,HAFIZ_UR,HAFIZ_EXP,fallback='false')
        on,on_calls,cache=self.run_search(rows,HAFIZ_UR,HAFIZ_EXP)
        self.assertEqual([],on['results']);self.assertNotIn('fallback',on)
        self.assertEqual((dump(off),off_calls),(dump(on),on_calls))
        self.assert_not_saved(cache)

    def test_hafiz_naeem_urdu_1105_at_threshold_is_returned_unverified(self):
        rows=self.corpus([(1105,'حافظ نعیم نوجوانوں سے خطاب'),*filler(10)])
        for score,shown in [(.39,[]),(.40,[1105])]:
            with self.subTest(score=score):
                self.verdicts={1105:(score,'role_mismatch',False)}
                on,_,_=self.run_search(rows,HAFIZ_UR,HAFIZ_EXP)
                if shown:self.assert_fallback(on,shown)
                else:self.assertEqual([],on['results']);self.assertNotIn('fallback',on)

    def test_invalid_reason_or_negation_never_shown(self):
        rows=self.corpus([(10,'الف'),(11,'ب'),*filler(6)])
        self.verdicts={10:(.6,'negation',False),11:(.6,'relevant',False)}  # Negation, or 'relevant' reason but not accepted.
        on,_,_=self.run_search(rows,MAKKAH,MAKKAH_EXP)
        self.assertEqual([],on['results']);self.assertNotIn('fallback',on)

    # Guards: queries with >= 1 verified result are byte-identical with the switch on or off.
    def test_short_and_long_verified_queries_unchanged(self):
        cases=[('Gaza',['Gaza (gaza; ghaza; غزہ)','غزہ','Gaza']),
               ('elite and mafias causing public hardship',['elite and mafias causing public hardship',
                'اشرافیہ اور مافیا عوام کو مشکلات میں ڈال رہے ہیں۔','Elite aur mafia awam ko mushkilat mein daal rahe hain.'])]
        for query,queries in cases:
            with self.subTest(query=query):
                self.client.delete_collection('source')
                self.client.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
                rows=self.corpus([(38,'غزہ پر بات چیت'),(874,'غزہ کے لوگ'),(6,'اشرافیہ'),*filler(10)])
                # One verified result plus related-only judgments that WOULD qualify for the fallback.
                self.verdicts={38:(.9,'relevant',True),874:(.6,'related_only',False),6:(.5,'specificity',False)}
                off,off_calls,_=self.run_search(rows,query,queries,fallback='false')
                on,on_calls,cache=self.run_search(rows,query,queries)
                self.assertEqual([38],[r['video_id']for r in on['results']])
                self.assertEqual((dump(off),off_calls),(dump(on),on_calls))
                self.assertNotIn('fallback',on);self.assertNotIn('verification_status',on['results'][0])
                self.assertTrue(on['results'][0]['transcript_verified'])
                self.assertTrue([k for k in cache if k.startswith('grounded:')])  # Verified -> saved as before.

    def test_kill_switch_off_restores_empty_answer(self):
        rows=self.corpus([(1087,'مکہ میں معاہدہ'),*filler(8)])
        self.verdicts={1087:(.45,'insufficient_evidence',False)}
        for value in ['false','0','off','no']:
            with self.subTest(value=value):
                off,_,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP,fallback=value)
                self.assertEqual([],off['results']);self.assertNotIn('fallback',off)
                self.assertNotIn('fallback',off['metadata']['passage_validation'])
                self.assertNotIn('fallback',off['ranking']);self.assert_not_saved(cache)

    def test_fallback_runs_after_strict_pass_when_fix_a_is_off(self):
        rows=self.corpus([(1087,'مکہ میں معاہدہ'),*filler(8)])
        self.verdicts={1087:(.45,'insufficient_evidence',False)}
        on,_,_=self.run_search(rows,MAKKAH,MAKKAH_EXP,rerank='false')
        self.assert_fallback(on,[1087]);self.assertNotIn('empty_result_rerank',on['metadata']['passage_validation'])

class FallbackCacheTests(Base):
    def test_unverified_never_saved_and_later_verified_not_blocked(self):
        rows=self.corpus([(1087,'مکہ میں معاہدہ'),*filler(8)])
        self.verdicts={1087:(.45,'insufficient_evidence',False)}
        first,first_calls,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP)
        self.assert_fallback(first,[1087]);self.assert_not_saved(cache)
        self.assertFalse([k for k in cache if k.startswith('grounded_judgment:')])  # Rejections never cached.
        again,calls,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP,cache=cache)
        self.assertEqual(first_calls,calls);self.assertFalse(again['metadata']['passage_validation']['cache_hit'])
        self.verdicts={1087:(.9,'relevant',True)}  # A later run verifies the passage.
        verified,calls,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP,cache=cache)
        self.assertGreater(calls,0);self.assertEqual([1087],[r['video_id']for r in verified['results']])
        self.assertFalse(is_related_fallback(verified));self.assertTrue(verified['results'][0]['transcript_verified'])
        self.assertNotIn('verification_status',verified['results'][0])
        replay,calls,_=self.run_search(rows,MAKKAH,MAKKAH_EXP,cache=cache)
        self.assertEqual(0,calls);self.assertTrue(replay['metadata']['passage_validation']['cache_hit'])
        self.assertEqual([1087],[r['video_id']for r in replay['results']])

    def planted(self,rows,rules,result):
        """Store `result` under the snapshot key the engine would use for MAKKAH with `rules`."""
        engine=GroundedSearch(FastSemanticSearch(self.client,'source',lambda q:[1.,0.],lambda q:{'semantic_query':q},
            lambda q,v:True,lambda q,v:True,settings=FastSearchSettings(candidate_limit=100)),self.judge,'fixture-model',
            lambda:'epoch',{}.get,lambda k,v,t:None,('same-model',2))
        params=vars(request(query=MAKKAH,min_score=0.)).copy()
        for key in ['cursor','exact_cursor','search_session_id','batch_size']:params.pop(key,None)
        return snapshot_key(engine,'epoch',params,None,8,16,rules_fingerprint=rules)

    def test_saved_fallback_snapshot_is_never_replayed_as_verified(self):
        rows=self.corpus([(1087,'مکہ میں معاہدہ'),*filler(8)])
        self.verdicts={1087:(.45,'insufficient_evidence',False)}
        unverified,_,_=self.run_search(rows,MAKKAH,MAKKAH_EXP)
        for rules in [None,LIVE_FP,PRE_FIX_A_FP]:
            with self.subTest(rules=rules):
                key=self.planted(rows,rules,unverified);cache={key:copy.deepcopy(unverified)}
                self.verdicts={1087:(.9,'relevant',True)}
                result,calls,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP,cache=cache)
                self.assertGreater(calls,0)  # Planted unverified snapshot ignored -> re-judged.
                self.assertFalse(is_related_fallback(result));self.assertTrue(result['results'][0]['transcript_verified'])
                self.verdicts={1087:(.45,'insufficient_evidence',False)}

class FixBCarryOverTests(Base):
    def test_carry_over_chain_serves_live_and_pre_fix_a_snapshots(self):
        self.assertEqual([LIVE_FP,PRE_FIX_A_FP],grounded_search.carried_rules_fingerprints())
        rows=self.corpus([(874,'غزہ کے لوگوں تک ہتھیار'),*filler(6)])
        self.verdicts={874:(.9,'relevant',True)}
        saved,_,_=self.run_search(rows,MAKKAH,MAKKAH_EXP)
        self.assertTrue(saved['results'])
        for rules in [LIVE_FP,PRE_FIX_A_FP]:
            with self.subTest(rules=rules):
                key=FallbackCacheTests.planted(self,rows,rules,saved)
                cache={key:copy.deepcopy(saved)}
                result,calls,cache=self.run_search(rows,MAKKAH,MAKKAH_EXP,cache=cache)
                self.assertEqual(0,calls);self.assertTrue(result['metadata']['passage_validation']['cache_hit'])
                self.assertEqual(saved['results'],result['results'])
                self.assertEqual(2,len([k for k in cache if k.startswith('grounded:')]))  # Copied; old key kept.

if __name__=='__main__':unittest.main()
